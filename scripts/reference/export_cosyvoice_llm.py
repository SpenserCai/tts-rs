"""Run the pinned official CosyVoice 3 LM with deterministic greedy decoding."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from common import PINS, environment, offline_reference_runtime, pinned_checkpoint_sha256, reference_runtime, revision, sha256, tensor_document, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt-features", type=Path, required=True)
    parser.add_argument("--text", default="你好。")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--attention-implementation", choices=["default", "eager", "sdpa"], default="default")
    parser.add_argument("--teacher-forced-tokens", nargs="+", type=int,
                        help="Optional fixed continuation for separate cached/full-context diagnostics")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    if revision(source) != PINS["cosyvoice"]["revision"]:
        raise RuntimeError("Official source revision differs from pin")
    if args.max_tokens < 1:
        raise ValueError("max-tokens must be positive")
    runtime = reference_runtime("cosyvoice")
    weights_sha256 = pinned_checkpoint_sha256("cosyvoice", args.model / "llm.pt")
    offline_reference_runtime()
    sys.path.insert(0, str(source))
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from cosyvoice.llm.llm import CosyVoice3LM, Qwen2Encoder
    from cosyvoice.tokenizer.tokenizer import CosyVoice3Tokenizer

    torch.set_num_threads(1)
    # The complete checkpoint already contains the backbone. Construct its
    # architecture on meta instead of downloading redundant pretrained weights;
    # inference still executes the unmodified official Qwen2Encoder methods.
    config = Qwen2Config.from_pretrained(args.model / "CosyVoice-BlankEN", local_files_only=True)
    if args.attention_implementation != "default":
        config._attn_implementation = args.attention_implementation
    with torch.device("meta"):
        encoder = Qwen2Encoder.__new__(Qwen2Encoder)
        torch.nn.Module.__init__(encoder)
        encoder.model = Qwen2ForCausalLM(config)
        model = CosyVoice3LM(896, 896, 6561, encoder,
                             sampling=lambda scores, _history, _sampling: scores.argmax())
    state = torch.load(args.model / "llm.pt", map_location="cpu", mmap=True, weights_only=True)
    model.load_state_dict(state, strict=True, assign=True)
    # RoPE frequencies are nonpersistent buffers, absent from the checkpoint.
    # Recreate them with the original Transformers initializer on CPU.
    rotary = model.llm.model.model.rotary_emb
    model.llm.model.model.rotary_emb = type(rotary)(config, device=torch.device("cpu"))
    model = model.float().eval()
    del state
    tokenizer = CosyVoice3Tokenizer(str(args.model / "CosyVoice-BlankEN"))
    prompt = json.loads(args.prompt_features.read_text())
    prompt_text = "You are a helpful assistant.<|endofprompt|>" + prompt["transcript"]
    tensors = {
        "text_tokens": torch.tensor([tokenizer.encode(args.text)], dtype=torch.uint32),
        "prompt_text_tokens": torch.tensor([tokenizer.encode(prompt_text)], dtype=torch.uint32),
        "prompt_tokens": torch.tensor([prompt["speech_tokens"]], dtype=torch.uint32),
    }
    logits = []

    def capture(_module, _inputs, result):
        if not logits:
            logits.append(result.detach().cpu())

    handle = model.llm_decoder.register_forward_hook(capture)
    target_len = tensors["text_tokens"].shape[-1]
    max_ratio = min(20.0, args.max_tokens / target_len)
    with torch.inference_mode():
        tokens = list(model.inference(
            text=tensors["text_tokens"].long(), text_len=torch.tensor([target_len]),
            prompt_text=tensors["prompt_text_tokens"].long(),
            prompt_text_len=torch.tensor([tensors["prompt_text_tokens"].shape[-1]]),
            prompt_speech_token=tensors["prompt_tokens"].long(),
            prompt_speech_token_len=torch.tensor([tensors["prompt_tokens"].shape[-1]]),
            embedding=torch.zeros(1, 192), sampling=1,
            max_token_text_ratio=max_ratio, min_token_text_ratio=2,
        ))
    handle.remove()
    if not logits or not tokens:
        raise RuntimeError("Official LM did not produce logits and speech tokens")
    metadata = {
        **environment(), "device": "cpu", "dtype": "float32", "torch": torch.__version__,
        "runtime": runtime,
        "source_revision": PINS["cosyvoice"]["revision"],
        "model_revision": PINS["cosyvoice"]["model_revision"],
        "weights_sha256": weights_sha256,
        "prompt_features_sha256": sha256(args.prompt_features),
        "max_tokens": args.max_tokens, "min_token_text_ratio": 2, "max_token_text_ratio": max_ratio,
        "attention_implementation": model.llm.model.config._attn_implementation,
        "scope": "official LM first raw logits and greedy speech tokens; excludes flow and HiFT",
    }
    if args.teacher_forced_tokens:
        if any(token < 0 or token >= 6561 for token in args.teacher_forced_tokens):
            raise ValueError("Teacher-forced speech tokens must be in [0, 6561)")
        with torch.inference_mode():
            text = torch.cat([tensors["prompt_text_tokens"], tensors["text_tokens"]], dim=1).long()
            context = torch.cat([
                model.speech_embedding.weight[model.sos].reshape(1, 1, -1),
                model.llm.model.model.embed_tokens(text),
                model.speech_embedding.weight[model.task_id].reshape(1, 1, -1),
                model.speech_embedding(tensors["prompt_tokens"].long()),
            ], dim=1)
            causal = lambda length: torch.ones(1, length, length, dtype=torch.bool).tril()
            hidden, cache = model.llm.forward_one_step(context, causal(context.shape[1]))
            diagnostics = {"initial_logits": model.llm_decoder(hidden[:, -1])}
            for index, token in enumerate(args.teacher_forced_tokens):
                embedding = model.speech_embedding.weight[token].reshape(1, 1, -1)
                # Preserve the exact upstream inference_wrapper mask contract.
                hidden, cache = model.llm.forward_one_step(embedding, causal(1), cache=cache)
                diagnostics[f"cached_logits_{index}"] = model.llm_decoder(hidden[:, -1])
                context = torch.cat([context, embedding], dim=1)
                hidden, _ = model.llm.forward_one_step(context, causal(context.shape[1]))
                diagnostics[f"full_context_logits_{index}"] = model.llm_decoder(hidden[:, -1])
        write_json(args.output / "teacher-forced.json", tensor_document(diagnostics, {
            **metadata, "teacher_forced_tokens": args.teacher_forced_tokens,
            "scope": "diagnostic comparison of official cached and full-context LM execution",
        }))
    write_json(args.output / "input.json", tensor_document(tensors, metadata))
    write_json(args.output / "reference.json", tensor_document({
        "first_logits": logits[0],
        "generated_tokens": torch.tensor([int(token) for token in tokens], dtype=torch.uint32),
    }, metadata))
    write_json(args.output / "manifest.json", metadata)
    print(f"Exported official LM: {target_len} target text tokens -> {len(tokens)} speech tokens")


if __name__ == "__main__":
    main()
