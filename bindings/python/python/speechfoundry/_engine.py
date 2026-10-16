"""Thin ownership wrapper; inference and audio validation remain in Rust."""

from __future__ import annotations

from typing import Sequence

from ._native import Audio
from ._backends import load_backend


class Engine:
    """A local model session; calls on one native engine are serialized."""

    __slots__ = ("_engine", "_backend")

    def __init__(self) -> None:
        raise TypeError("Use Engine.load() to create a model session")

    @classmethod
    def load(
        cls,
        model_dir: str,
        model: str,
        device: str = "cpu",
        *,
        backend: str | None = None,
    ) -> Engine:
        """Load a model on a device. Specify a build only when CUDA selection is ambiguous."""
        selected, module = load_backend(device, backend)
        native = module.Engine.load(model_dir, model, device)
        result = object.__new__(cls)
        result._engine, result._backend = native, selected
        return result

    @property
    def backend(self) -> str:
        """The selected installed build identifier, not a claim of model parity."""
        return self._backend

    def extract_voice(
        self,
        samples: Sequence[float],
        sample_rate: int,
        transcript: str | None = None,
    ) -> str:
        """Extract checkpoint-specific voice conditioning JSON."""
        return self._engine.extract_voice(samples, sample_rate, transcript)

    def synthesize(
        self,
        text: str,
        voice_prompt_json: str,
        options_json: str | None = None,
    ) -> Audio:
        """Return public Audio regardless of which native extension owns the engine."""
        audio = self._engine.synthesize(text, voice_prompt_json, options_json)
        return audio if isinstance(audio, Audio) else Audio(audio.samples, audio.sample_rate)
