use std::{fs::File, io::Read, path::Path};

use sha2::{Digest, Sha256};

use crate::{Error, ModelKind, Result};

fn names(kind: ModelKind) -> &'static [&'static str] {
    match kind {
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
    }
}

/// Incremental checkpoint identity shared by file and explicit-byte loaders.
///
/// Process every path returned by [`Self::file_names`] in order. Begin each
/// present file with its exact byte length, then feed its bytes with [`Self::update`].
/// Begin absent files with `None`. Large files never need a contiguous buffer.
pub struct CheckpointHasher {
    hasher: Sha256,
    names: &'static [&'static str],
    index: usize,
    remaining: u64,
}

impl CheckpointHasher {
    /// Start a checkpoint digest for one model family.
    pub fn new(kind: ModelKind) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(b"candle-tts-checkpoint-v1\0");
        hasher.update(kind.to_string());
        Self {
            hasher,
            names: names(kind),
            index: 0,
            remaining: 0,
        }
    }

    /// Canonical relative paths, including optional files which must be marked absent.
    pub fn file_names(&self) -> &'static [&'static str] {
        self.names
    }

    /// Begin the next file; `None` explicitly records that the file is absent.
    pub fn begin_file(&mut self, name: &str, length: Option<u64>) -> Result<()> {
        if self.remaining != 0 || self.names.get(self.index).copied() != Some(name) {
            return Err(Error::InvalidInput(
                "checkpoint files must be complete and in canonical order".into(),
            ));
        }
        self.hasher.update(name.as_bytes());
        match length {
            None => self.hasher.update([0]),
            Some(length) => {
                self.hasher.update([1]);
                self.hasher.update(length.to_le_bytes());
                self.remaining = length;
            }
        }
        self.index += 1;
        Ok(())
    }

    /// Add a bounded chunk from the current file.
    pub fn update(&mut self, bytes: &[u8]) -> Result<()> {
        let length = u64::try_from(bytes.len())
            .map_err(|_| Error::InvalidInput("chunk length overflow".into()))?;
        if self.index == 0 || length > self.remaining {
            return Err(Error::InvalidInput(
                "checkpoint chunk exceeds the declared file length".into(),
            ));
        }
        self.hasher.update(bytes);
        self.remaining -= length;
        Ok(())
    }

    /// Complete the digest, rejecting missing or truncated files.
    pub fn finish(self) -> Result<String> {
        if self.index != self.names.len() || self.remaining != 0 {
            return Err(Error::InvalidInput(
                "checkpoint digest is incomplete".into(),
            ));
        }
        Ok(format!("{:x}", self.hasher.finalize()))
    }
}

/// Hash content, not paths or timestamps, so prompts survive model relocation.
pub(crate) fn model_fingerprint(path: &Path, kind: ModelKind) -> Result<String> {
    let mut hasher = CheckpointHasher::new(kind);
    let mut buffer = vec![0; 1024 * 1024];
    for name in hasher.file_names() {
        let file_path = path.join(name);
        if !file_path.exists() {
            hasher.begin_file(name, None)?;
            continue;
        }
        let mut file = File::open(file_path)?;
        hasher.begin_file(name, Some(file.metadata()?.len()))?;
        loop {
            let count = file.read(&mut buffer)?;
            if count == 0 {
                break;
            }
            hasher.update(&buffer[..count])?;
        }
    }
    hasher.finish()
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

    #[test]
    fn chunked_identity_matches_files_and_rejects_truncation() {
        let root = tempfile::tempdir().unwrap();
        std::fs::write(root.path().join("config.json"), b"abcdef").unwrap();
        let mut hasher = CheckpointHasher::new(ModelKind::Qwen3);
        assert!(hasher.begin_file("vocab.json", None).is_err());
        hasher.begin_file("config.json", Some(6)).unwrap();
        hasher.update(b"abc").unwrap();
        assert!(hasher.begin_file("model.safetensors", None).is_err());
        assert!(hasher.update(b"toolong").is_err());
        hasher.update(b"def").unwrap();
        for name in &hasher.file_names()[1..] {
            hasher.begin_file(name, None).unwrap();
        }
        assert_eq!(
            hasher.finish().unwrap(),
            model_fingerprint(root.path(), ModelKind::Qwen3).unwrap()
        );
        assert!(CheckpointHasher::new(ModelKind::Qwen3).finish().is_err());
    }
}
