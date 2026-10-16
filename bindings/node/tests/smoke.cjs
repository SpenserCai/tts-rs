const assert = require('node:assert/strict')
const { test } = require('node:test')
const { loadEngine, resampleAudio, validateVoicePrompt } = require('../index.js')

test('ESM named imports match the CommonJS facade', async () => {
  const esm = await import('../index.js')
  assert.equal(esm.loadEngine, loadEngine)
  assert.equal(esm.resampleAudio, resampleAudio)
  assert.equal(typeof esm.Engine, 'function')
  assert(esm.installedBackends().includes('cpu'))
})

test('facade rejects invalid device selection asynchronously', async () => {
  await assert.rejects(loadEngine('does-not-exist', 'qwen3', 'cuda:-1'), /device/)
  await assert.rejects(loadEngine('does-not-exist', 'qwen3', 'cpu', { backend: 'cu124' }), /CPU/)
})

test('native audio round trip uses Float32Array', async () => {
  const pcm = new Float32Array([0, 0.25, -0.25, 0])
  const result = await resampleAudio(pcm, 24000, 24000)
  assert(result.samples instanceof Float32Array)
  assert.deepEqual(result.samples, pcm)
  assert.equal(result.sampleRate, 24000)
})

test('invalid audio rejects its Promise', async () => {
  await assert.rejects(resampleAudio(new Float32Array([NaN]), 24000, 24000))
  await assert.rejects(resampleAudio(new Float32Array([0]), 0, 24000))
})

test('invalid model rejects its Promise before accessing a checkpoint', async () => {
  await assert.rejects(loadEngine('does-not-exist', 'not-a-model'), /model/)
})

test('invalid prompt rejects its Promise', async () => {
  await assert.rejects(validateVoicePrompt('{}'))
})

test('oversized prompt rejects before decoding', async () => {
  await assert.rejects(validateVoicePrompt('{}' + ' '.repeat(16 * 1024 * 1024)), /limit/)
})

test('valid prompt envelope round trips and unknown schema rejects', async () => {
  const prompt = {
    schema_version: 1,
    model: 'qwen3',
    model_fingerprint: 'a'.repeat(64),
    format: 'test-envelope-v1',
    payload: {},
  }
  assert.deepEqual(JSON.parse(await validateVoicePrompt(JSON.stringify(prompt))), prompt)
  prompt.schema_version = 999
  await assert.rejects(validateVoicePrompt(JSON.stringify(prompt)))
})
