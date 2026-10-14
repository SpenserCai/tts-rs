"""Installed-wheel contract tests; no model downloads or network access."""

import json
import math
import unittest

from candle_tts import Audio, Engine, validate_voice_prompt


class BindingSmokeTests(unittest.TestCase):
    def test_audio_round_trip(self):
        audio = Audio([0.0, 0.25, -0.25, 0.0], 24000)
        self.assertEqual(audio.samples, [0.0, 0.25, -0.25, 0.0])
        self.assertEqual(audio.sample_rate, 24000)
        self.assertAlmostEqual(audio.duration_seconds, 4 / 24000)
        self.assertEqual(audio.resample(24000).samples, audio.samples)

    def test_invalid_audio_is_exception(self):
        with self.assertRaises(ValueError):
            Audio([0.0], 0)
        with self.assertRaises(ValueError):
            Audio([math.nan], 24000)

    def test_invalid_model_rejected_before_loading(self):
        with self.assertRaises(ValueError):
            Engine.load("does-not-exist", "not-a-model")

    def test_invalid_prompt_is_exception(self):
        with self.assertRaises(ValueError):
            validate_voice_prompt("{}")

    def test_oversized_prompt_is_rejected_before_decoding(self):
        with self.assertRaisesRegex(ValueError, "limit"):
            validate_voice_prompt("{}" + " " * (16 * 1024 * 1024))

    def test_prompt_envelope_round_trip(self):
        prompt = {
            "schema_version": 1,
            "model": "qwen3",
            "model_fingerprint": "a" * 64,
            "format": "test-envelope-v1",
            "payload": {},
        }
        self.assertEqual(json.loads(validate_voice_prompt(json.dumps(prompt))), prompt)
        prompt["schema_version"] = 999
        with self.assertRaises(ValueError):
            validate_voice_prompt(json.dumps(prompt))


if __name__ == "__main__":
    unittest.main()
