const assert = require('node:assert/strict')
const { test } = require('node:test')
const { createHash } = require('node:crypto')
const wasm = require('../pkg-memory64-node/speechfoundry_wasm.js')

test('memory64 exposes the actual Qwen synthesis API and finite PCM', () => {
  const capabilities = JSON.parse(wasm.capabilities())
  assert.equal(capabilities.address_bits, 64)
  assert.equal(capabilities.speech_synthesis, true)
  assert.deepEqual(capabilities.models, ['qwen3'])
  assert.equal(typeof wasm.QwenModelBuilder, 'function')
  assert.equal(typeof wasm.QwenModel.prototype.synthesize, 'function')
  const audio = new wasm.Audio(new Float32Array([0.125, -0.25]), 24000)
  try { assert.deepEqual([...audio.samples], [0.125, -0.25]) } finally { audio.free() }
  assert.throws(() => new wasm.Audio(new Float32Array([NaN]), 24000))
})

test('memory64 hashes fragmented files with the native identity protocol', () => {
  const hasher = new wasm.CheckpointHasher()
  const expected = createHash('sha256').update('candle-tts-checkpoint-v1\0').update('qwen3')
  for (const [index, name] of JSON.parse(hasher.fileNames()).entries()) {
    expected.update(name)
    if (index === 0) {
      const length = Buffer.alloc(8); length.writeBigUInt64LE(6n)
      expected.update(Buffer.from([1])).update(length).update('abcdef')
      hasher.beginFile(name, 6n)
      hasher.update(new TextEncoder().encode('abc'))
      assert.throws(() => hasher.skipFile('model.safetensors'))
      hasher.update(new TextEncoder().encode('def'))
    } else { hasher.skipFile(name); expected.update(Buffer.from([0])) }
  }
  assert.equal(hasher.finish(), expected.digest('hex'))
})

test('memory64 rejects incomplete and malformed checkpoint metadata', () => {
  let hasher = new wasm.CheckpointHasher()
  assert.throws(() => hasher.finish())
  assert.throws(() => new wasm.QwenModelBuilder('{}', '{}'))
})


test('memory64 accepts official-style byte tokenizers and rejects malformed tensor uploads', () => {
  const builder = wasm.QwenModelBuilder.fromVocabAndMerges('{}', '{"a":0,"b":1,"ab":2,"<|endoftext|>":3}', '#version: 0.2\na b\n')
  try {
    assert.throws(() => builder.addTensor('wrong', 'weight', 'F32', '[1]', new Uint8Array(4)))
    assert.throws(() => builder.addTensor('model', 'weight', 'F32', '[2]', new Uint8Array(4)))
    builder.addTensor('model', 'weight', 'F32', '[1]', new Uint8Array(4))
    assert.throws(() => builder.addTensor('model', 'weight', 'F32', '[1]', new Uint8Array(4)))
  } finally { builder.free() }
})
