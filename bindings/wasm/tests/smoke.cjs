const assert = require('node:assert/strict')
const { test } = require('node:test')
const {
  Audio,
  capabilities,
  validateVoicePrompt,
  validateSynthesisOptions,
  QwenSpeakerEncoder,
} = require('../pkg/tts_wasm.js')

test('Wasm audio preserves PCM and metadata', () => {
  const pcm = new Float32Array([0, 0.25, -0.25, 0])
  const audio = new Audio(pcm, 24000)
  const same = audio.resample(24000)
  try {
    assert.deepEqual(audio.samples, pcm)
    assert.equal(audio.sampleRate, 24000)
    assert.equal(audio.durationSeconds, 4 / 24000)
    assert.deepEqual(same.samples, pcm)
  } finally {
    same.free()
    audio.free()
  }
})

test('Wasm rejects invalid PCM and malformed envelopes', () => {
  assert.throws(() => new Audio(new Float32Array([NaN]), 24000))
  assert.throws(() => new Audio(new Float32Array([0]), 0))
  assert.throws(() => validateVoicePrompt('{}'))
  assert.throws(() => validateSynthesisOptions('{"top_p":0}'))
})

test('Wasm options use the shared defaults and capabilities are explicit', () => {
  assert.equal(JSON.parse(validateSynthesisOptions('{}')).seed, 0)
  const supported = JSON.parse(capabilities())
  assert.equal(supported.audio_resampling, true)
  assert.equal(supported.speech_synthesis, false)
  assert.equal(supported.voice_extraction, typeof QwenSpeakerEncoder === 'function')
})

test('Wasm rejects oversized JSON before decoding', () => {
  assert.throws(() => validateSynthesisOptions('{}' + ' '.repeat(65536)), /limit/)
  assert.throws(() => validateVoicePrompt('{}' + ' '.repeat(16 * 1024 * 1024)), /limit/)
})

test('Wasm validates the versioned prompt envelope without claiming payload inference', () => {
  const prompt = {
    schema_version: 1,
    model: 'qwen3',
    model_fingerprint: 'a'.repeat(64),
    format: 'test-envelope-v1',
    payload: {},
  }
  assert.deepEqual(JSON.parse(validateVoicePrompt(JSON.stringify(prompt))), prompt)
  prompt.schema_version = 999
  assert.throws(() => validateVoicePrompt(JSON.stringify(prompt)))
})
