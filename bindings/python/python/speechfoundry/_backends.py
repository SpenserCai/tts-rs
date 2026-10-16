"""Metadata-only discovery; never import an accelerator to probe its availability."""

from importlib import import_module, metadata
import re
from threading import Lock

from . import _native

_GROUP = "speechfoundry.backends"
_CUDA = re.compile(r"cu[0-9]{3}")
_DEVICE = re.compile(r"(cuda|metal)(?::[0-9]+)?")
_lock = Lock()
_cuda_backend = None


def _entries():
    entries = {}
    for entry in metadata.entry_points(group=_GROUP):
        if entry.name != "metal" and _CUDA.fullmatch(entry.name) is None:
            raise RuntimeError(f"Invalid SpeechFoundry backend registration: {entry.name}")
        expected = f"_speechfoundry_{entry.name}._native"
        if entry.value != expected or entry.name in entries:
            raise RuntimeError(f"Conflicting SpeechFoundry backend registration: {entry.name}")
        if entry.dist is None or entry.dist.metadata["Name"] != f"speechfoundry-{entry.name}":
            raise RuntimeError(f"Unexpected distribution for backend {entry.name}")
        entries[entry.name] = entry
    return entries


def installed_backends() -> list[str]:
    """Return installed build identifiers, without loading or certifying GPU runtimes."""
    return sorted(set(_native._backends) | set(_entries()))


def _validate(module, family):
    if getattr(module, "__version__", None) != _native.__version__ or getattr(module, "_backend_api", None) != 1:
        raise RuntimeError("SpeechFoundry native backend version/API mismatch; install matching packages")
    if family not in getattr(module, "_backends", ()):
        raise RuntimeError(f"The selected backend was not compiled for {family}")
    return module


def load_backend(device, backend=None):
    """Resolve one explicit device, rejecting ambiguity and native ABI mixing."""
    global _cuda_backend
    if device == "cpu":
        if backend not in (None, "cpu"):
            raise ValueError("CPU execution uses backend='cpu'")
        return "cpu", _validate(_native, "cpu")
    match = _DEVICE.fullmatch(device) if isinstance(device, str) else None
    if match is None:
        raise ValueError("device must be cpu, cuda:N, or metal:N")
    family = match[1]
    entries = _entries()
    candidates = [name for name in entries if (name.startswith("cu") if family == "cuda" else name == "metal")]
    if family in _native._backends:
        candidates.append(family)
    candidates = sorted(set(candidates))
    if backend is None:
        if len(candidates) != 1:
            detail = ", ".join(candidates) if candidates else "none installed"
            raise RuntimeError(f"Select one {family} backend with backend=... ({detail}); install a matching preview if needed")
        backend = candidates[0]
    if backend not in candidates:
        raise ValueError(f"Backend {backend!r} is not installed for {family}")
    entry = entries.get(backend) if backend != family or family not in _native._backends else None
    if entry is not None and entry.dist.version != _native.__version__.replace("-rc.", "rc"):
        raise RuntimeError(f"Backend {backend} version mismatch; install SpeechFoundry {_native.__version__}")
    with _lock:
        if family == "cuda":
            if _cuda_backend is not None and _cuda_backend != backend:
                raise RuntimeError("Use a separate process for a different CUDA build; native library ABIs are not isolated")
            # Keep the reservation even after failure: native imports can partially load libraries.
            _cuda_backend = backend
        try:
            module = import_module(entry.value) if entry is not None else _native
        except (ImportError, OSError) as error:
            raise RuntimeError(f"Cannot load {backend}: check its native runtime dependencies; no CPU fallback") from error
        return backend, _validate(module, family)
