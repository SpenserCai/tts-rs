"""Run the pinned official CosyVoice flow network on fixed public prompt features."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

from common import PINS, environment, offline_reference_runtime, pinned_checkpoint_sha256, reference_runtime, revision, sha256, tensor_document, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--prompt-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-tokens", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--attention-backend", choices=["default", "math"], default="default")
    parser.add_argument("--layerwise", action="store_true", help="Export first-step DiT layer outputs for numerical diagnosis")
    args = parser.parse_args()
    source = args.source.resolve()
    matcha = source / "third_party" / "Matcha-TTS"
    if revision(source) != PINS["cosyvoice"]["revision"] or revision(matcha) != PINS["cosyvoice"]["matcha_revision"]:
        raise RuntimeError("Official CosyVoice/Matcha revisions differ from the reviewed pins")
    sys.path[:0] = [str(source), str(matcha)]
    runtime = reference_runtime("cosyvoice")
    weights_sha256 = pinned_checkpoint_sha256("cosyvoice", args.weights)
    offline_reference_runtime()
    import torch
    from omegaconf import DictConfig
    from safetensors.torch import save_file
    from cosyvoice.flow.flow import CausalMaskedDiffWithDiT
    from cosyvoice.flow.flow_matching import CausalConditionalCFM
    from cosyvoice.flow.DiT.dit import DiT
    from cosyvoice.transformer.upsample_encoder import PreLookaheadLayer

    torch.set_num_threads(1)
    prompt = json.loads(args.prompt_features.read_text())
    total_tokens = args.prompt_tokens + args.tokens
    if min(args.prompt_tokens, args.tokens) < 1 or total_tokens > len(prompt["speech_tokens"]):
        raise ValueError("Prompt/target token slices exceed the available public feature sequence")
    decoder = CausalConditionalCFM(
        in_channels=240, n_spks=1, spk_emb_dim=80,
        cfm_params=DictConfig({"sigma_min": 1e-6, "solver": "euler", "t_scheduler": "cosine",
                               "training_cfg_rate": 0.2, "inference_cfg_rate": 0.7, "reg_loss_type": "l1"}),
        estimator=DiT(dim=1024, depth=22, heads=16, dim_head=64, ff_mult=2,
                      mel_dim=80, mu_dim=80, spk_dim=80, out_channels=80,
                      static_chunk_size=50, num_decoding_left_chunks=-1),
    )
    model = CausalMaskedDiffWithDiT(
        input_size=80, output_size=80, spk_embed_dim=192, output_type="mel", vocab_size=6561,
        input_frame_rate=25, only_mask_loss=True, token_mel_ratio=2, pre_lookahead_len=3,
        pre_lookahead_layer=PreLookaheadLayer(in_channels=80, channels=1024, pre_lookahead_len=3),
        decoder=decoder,
    ).eval()
    state = torch.load(args.weights, map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict(state, strict=True)
    tensors = {
        "tokens": torch.tensor([prompt["speech_tokens"][args.prompt_tokens:total_tokens]], dtype=torch.uint32),
        "prompt_tokens": torch.tensor([prompt["speech_tokens"][:args.prompt_tokens]], dtype=torch.uint32),
        "prompt_mel": torch.tensor([prompt["mel"][:args.prompt_tokens * 2]], dtype=torch.float32),
        "speaker_embedding": torch.tensor([prompt["speaker_embedding"]], dtype=torch.float32),
    }
    intermediate, handles = {}, []

    def capture(name, result):
        if name not in intermediate:
            intermediate[name] = result.detach().cpu()

    trace_modules = ["pre_lookahead_layer", "spk_embed_affine_layer", "decoder.estimator"]
    if args.layerwise:
        trace_modules.extend(["decoder.estimator.time_embed", "decoder.estimator.input_embed",
                              "decoder.estimator.norm_out"])
        trace_modules.extend(f"decoder.estimator.transformer_blocks.{i}" for i in range(22))
    for name in trace_modules:
        handles.append(model.get_submodule(name).register_forward_hook(
            lambda _module, _inputs, result, key=name: capture(key, result)
        ))
    from torch.nn.attention import SDPBackend, sdpa_kernel
    attention = sdpa_kernel(SDPBackend.MATH) if args.attention_backend == "math" else contextlib.nullcontext()
    with torch.inference_mode(), attention:
        result, _ = model.inference(
            token=tensors["tokens"].long(), token_len=torch.tensor([args.tokens]),
            prompt_token=tensors["prompt_tokens"].long(), prompt_token_len=torch.tensor([args.prompt_tokens]),
            prompt_feat=tensors["prompt_mel"], prompt_feat_len=torch.tensor([args.prompt_tokens * 2]),
            embedding=tensors["speaker_embedding"], streaming=False, finalize=True,
        )
    for handle in handles:
        handle.remove()
    args.output.mkdir(parents=True, exist_ok=True)
    save_file({"noise": model.decoder.rand_noise.contiguous()}, str(args.output / "flow-noise.safetensors"))
    metadata = {
        **environment(), "model": "cosyvoice3-flow", "device": "cpu", "dtype": "float32",
        "torch": torch.__version__, "source_revision": PINS["cosyvoice"]["revision"],
        "model_revision": PINS["cosyvoice"]["model_revision"],
        "runtime": runtime,
        "matcha_revision": PINS["cosyvoice"]["matcha_revision"],
        "weights_sha256": weights_sha256, "input_features_sha256": sha256(args.prompt_features),
        "flow_noise_sha256": sha256(args.output / "flow-noise.safetensors"),
        "timesteps": 10, "streaming": False, "finalize": True,
        "attention_backend": args.attention_backend, "layerwise_trace": args.layerwise,
        "scope": "official token-conditioned flow; excludes LLM and HiFT",
        "randomness": "exact official decoder.rand_noise tensor exported for Candle replay",
    }
    write_json(args.output / "input.json", tensor_document(tensors, metadata))
    write_json(args.output / "reference.json", tensor_document({"mel": result}, metadata))
    write_json(args.output / "intermediates.json", tensor_document(intermediate, metadata))
    write_json(args.output / "manifest.json", metadata)
    print(f"Exported official flow: {args.tokens} target tokens -> {result.shape[-1]} mel frames")


if __name__ == "__main__":
    main()
