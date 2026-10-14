use std::{fs::File, io::Read, path::Path};

use sha2::{Digest, Sha256};

use crate::{ModelKind, Result};

/// Hash content, not paths or timestamps, so prompts survive model relocation.
pub(crate) fn model_fingerprint(path: &Path, kind: ModelKind) -> Result<String> {
    let names: &[&str] = match kind {
        ModelKind::Qwen3 => &[
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "vocab.json",
            "merges.txt",
            "tokenizer_config.json",
            "speech_tokenizer/config.json",
            "speech_tokenizer/model.safetensors",
        ],
        ModelKind::Cosyvoice3 => &[
            "model-config.json",
            "llm.safetensors",
            "flow.safetensors",
            "hift.safetensors",
            "tokenizer.json",
            "flow-noise.safetensors",
            "source-noise.safetensors",
            "campplus.onnx",
            "speech_tokenizer_v3.onnx",
        ],
    };
    let mut hasher = Sha256::new();
    hasher.update(b"candle-tts-checkpoint-v1\0");
    hasher.update(kind.to_string());
    let mut buffer = vec![0; 1024 * 1024];
    for name in names {
        hasher.update(name.as_bytes());
        let file_path = path.join(name);
        if !file_path.exists() {
            hasher.update([0]);
            continue;
        }
        hasher.update([1]);
        let mut file = File::open(file_path)?;
        hasher.update(file.metadata()?.len().to_le_bytes());
        loop {
            let count = file.read(&mut buffer)?;
            if count == 0 {
                break;
            }
            hasher.update(&buffer[..count]);
        }
    }
    Ok(format!("{:x}", hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizer_contents_participate_in_checkpoint_identity() {
        let root = tempfile::tempdir().unwrap();
        let first = model_fingerprint(root.path(), ModelKind::Qwen3).unwrap();
        std::fs::write(root.path().join("vocab.json"), b"{\"a\":0}").unwrap();
        let second = model_fingerprint(root.path(), ModelKind::Qwen3).unwrap();
        assert_ne!(first, second);
        std::fs::write(root.path().join("vocab.json"), b"{\"b\":0}").unwrap();
        let third = model_fingerprint(root.path(), ModelKind::Qwen3).unwrap();
        assert_ne!(second, third);
    }
}
