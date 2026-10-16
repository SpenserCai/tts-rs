from typing import Sequence

__version__: str
_backend_api: int
_backends: list[str]

class Audio:
    def __init__(self, samples: Sequence[float], sample_rate: int) -> None: ...
    @property
    def samples(self) -> list[float]: ...
    @property
    def sample_rate(self) -> int: ...
    @property
    def duration_seconds(self) -> float: ...
    def resample(self, sample_rate: int) -> Audio: ...

class Engine:
    @staticmethod
    def load(model_dir: str, model: str, device: str = "cpu") -> Engine: ...
    def extract_voice(
        self,
        samples: Sequence[float],
        sample_rate: int,
        transcript: str | None = None,
    ) -> str: ...
    def synthesize(
        self,
        text: str,
        voice_prompt_json: str,
        options_json: str | None = None,
    ) -> Audio: ...

def validate_voice_prompt(voice_prompt_json: str) -> str: ...
