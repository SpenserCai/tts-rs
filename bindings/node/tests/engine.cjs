// Wrapper behavior only. No model or GPU inference is simulated as validation.
const assert = require('node:assert/strict')
const { test } = require('node:test')
const { version } = require('../package.json')
const calls = []
const samples = new Float32Array([0, 0.25])
const nativeEngine = {
  async extractVoice(...args) { calls.push(args); return 'voice-json' },
  async synthesize(...args) { calls.push(args); return { samples, sampleRate: 24000 } },
}
require.cache[require.resolve('../_native.js')] = { loaded: true, exports: {
  backendInfo: () => ({ version, api: 1, cuda: false, metal: false }),
  async loadEngine(...args) { calls.push(args); return nativeEngine },
} }
const { Engine, loadEngine } = require('../index.js')

test('one public Engine owns a selected native session and forwards arguments', async () => {
  const engine = await loadEngine('checkpoint', 'qwen3')
  assert(engine instanceof Engine)
  assert.equal(engine.backend, 'cpu')
  assert.equal(await engine.extractVoice(samples, 24000, 'reference'), 'voice-json')
  assert.deepEqual(await engine.synthesize('text', 'voice-json', '{}'), { samples, sampleRate: 24000 })
  assert.deepEqual(calls, [['checkpoint', 'qwen3', 'cpu'], [samples, 24000, 'reference'], ['text', 'voice-json', '{}']])
  assert.throws(() => new Engine(), /loadEngine/)
})
