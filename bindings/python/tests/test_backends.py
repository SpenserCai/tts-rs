"""Wrapper contracts. Mock native backends do not establish GPU execution."""

import types
import unittest
from unittest.mock import Mock, patch

from speechfoundry import Audio, Engine, __version__, installed_backends
from speechfoundry import _backends


def registration(name, version=None):
    return types.SimpleNamespace(
        name=name, value=f"_speechfoundry_{name}._native",
        dist=types.SimpleNamespace(version=version or __version__.replace("-rc.", "rc"),
                                  metadata={"Name": f"speechfoundry-{name}"}),
    )


def native(family="cuda", version=__version__, api=1):
    result = types.SimpleNamespace(__version__=version, _backend_api=api,
                                   _backends=["cpu", family], Engine=Mock())
    result.Engine.load.return_value.synthesize.return_value = types.SimpleNamespace(
        samples=[0.0, 0.25], sample_rate=24000)
    result.Engine.load.return_value.extract_voice.return_value = "voice-json"
    return result


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(patch.stopall)
        patch.object(_backends, "_cuda_backend", None).start()
        self.entries = patch.object(_backends.metadata, "entry_points", return_value=[]).start()
        self.load = patch.object(_backends, "import_module", return_value=native()).start()

    def test_cpu_never_probes_or_imports_installed_accelerators(self):
        self.entries.side_effect = AssertionError("CPU must not enumerate GPU packages")
        self.assertEqual(_backends.load_backend("cpu")[0], "cpu")
        self.load.assert_not_called()

    def test_discovery_only_reads_metadata(self):
        self.entries.return_value = [registration("cu124"), registration("metal")]
        self.assertEqual(installed_backends(), ["cpu", "cu124", "metal"])
        self.load.assert_not_called()

    def test_one_backend_is_selected_and_public_audio_is_preserved(self):
        self.entries.return_value = [registration("cu124")]
        engine = Engine.load("checkpoint", "qwen3", "cuda:0")
        self.assertEqual(engine.backend, "cu124")
        self.assertEqual(engine.extract_voice([0.0], 24000), "voice-json")
        speech = engine.synthesize("text", "voice-json")
        self.assertIsInstance(speech, Audio)
        self.assertEqual(speech.samples, [0.0, 0.25])
        self.load.assert_called_once_with("_speechfoundry_cu124._native")

    def test_missing_and_multiple_backends_fail_before_native_import(self):
        for entries in ([], [registration("cu118"), registration("cu124")]):
            self.entries.return_value = entries
            with self.assertRaisesRegex(RuntimeError, "Select one"):
                Engine.load("checkpoint", "qwen3", "cuda:0")
        self.load.assert_not_called()

    def test_explicit_selection_does_not_import_other_versions(self):
        self.entries.return_value = [registration("cu118"), registration("cu124")]
        Engine.load("checkpoint", "qwen3", "cuda:1", backend="cu124")
        self.load.assert_called_once_with("_speechfoundry_cu124._native")
        with self.assertRaisesRegex(RuntimeError, "separate process"):
            Engine.load("checkpoint", "qwen3", "cuda:0", backend="cu118")

    def test_distribution_version_checked_before_import(self):
        self.entries.return_value = [registration("cu124", "0.0.1")]
        with self.assertRaisesRegex(RuntimeError, "version mismatch"):
            Engine.load("checkpoint", "qwen3", "cuda:0")
        self.load.assert_not_called()

    def test_compiled_version_protocol_and_feature_are_verified(self):
        self.entries.return_value = [registration("cu124")]
        for module in (native(version="0.0.1"), native(api=99), native(family="metal")):
            self.load.return_value = module
            with self.assertRaises(RuntimeError):
                Engine.load("checkpoint", "qwen3", "cuda:0")

    def test_load_failure_has_no_cpu_fallback_and_retains_abi_guard(self):
        self.entries.return_value = [registration("cu118"), registration("cu124")]
        self.load.side_effect = ImportError("CUDA dependency missing")
        with self.assertRaisesRegex(RuntimeError, "no CPU fallback"):
            Engine.load("checkpoint", "qwen3", "cuda:0", backend="cu124")
        with self.assertRaisesRegex(RuntimeError, "separate process"):
            Engine.load("checkpoint", "qwen3", "cuda:0", backend="cu118")
        self.assertEqual(_backends.load_backend("cpu")[0], "cpu")

    def test_metal_uses_the_same_entry_point(self):
        self.entries.return_value = [registration("metal")]
        self.load.return_value = native(family="metal")
        self.assertEqual(Engine.load("checkpoint", "qwen3", "metal:0").backend, "metal")

    def test_invalid_device_selection_or_registration_rejected(self):
        for device, backend in (("gpu", None), ("cuda:-1", None), ("cpu", "cu124"), ("metal:0", "cu124")):
            with self.assertRaises(ValueError):
                _backends.load_backend(device, backend)
        entry = registration("cu124")
        entry.value = "unrelated.module"
        self.entries.return_value = [entry]
        with self.assertRaisesRegex(RuntimeError, "registration"):
            installed_backends()
        self.load.assert_not_called()

    def test_builtin_source_build_can_use_its_compiled_accelerator(self):
        with patch.object(_backends._native, "_backends", ["cpu", "cuda"]):
            selected, module = _backends.load_backend("cuda:0")
            self.assertEqual(selected, "cuda")
            self.assertIs(module, _backends._native)
        self.load.assert_not_called()


class InstalledPackageTests(unittest.TestCase):
    def test_real_installed_backend_discovery_does_not_load_gpu_libraries(self):
        with patch.object(_backends, "import_module", side_effect=AssertionError("unexpected native import")):
            self.assertIn("cpu", installed_backends())

    def test_public_engine_cannot_be_constructed_without_loading(self):
        with self.assertRaisesRegex(TypeError, "Engine.load"):
            Engine()


if __name__ == "__main__":
    unittest.main()
