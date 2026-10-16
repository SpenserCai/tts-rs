// Explicit real-weight parity probe. The default smoke suite never downloads models.
// Usage: node tests/qwen.cjs /path/to/speaker-fixture /path/to/generated/pkg
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')

const fixtureDir = path.resolve(process.argv[2] || '')
const packageDir = path.resolve(process.argv[3] || path.join(__dirname, '../pkg-qwen'))
if (!process.argv[2]) throw new Error('A speaker fixture directory is required')
const { QwenSpeakerEncoder, capabilities } = require(path.join(packageDir, 'speechfoundry_wasm.js'))
assert.equal(JSON.parse(capabilities()).voice_extraction, true)

const readJson = (name) => JSON.parse(fs.readFileSync(path.join(fixtureDir, name), 'utf8'))
const input = readJson('input.json')
const reference = readJson('reference.json')
const config = fs.readFileSync(path.join(fixtureDir, 'config.json'), 'utf8')
const weights = fs.readFileSync(path.join(fixtureDir, 'speaker.safetensors'))
const encoder = new QwenSpeakerEncoder(config, new Uint8Array(weights))

try {
  const inputTensor = Object.values(input.tensors)[0]
  const referenceTensor = Object.values(reference.tensors)[0]
  const frames = inputTensor.shape.at(-1)
  const actual = encoder.forwardMel(new Float32Array(inputTensor.data), frames)
  const expected = referenceTensor.data
  assert.equal(actual.length, expected.length)
  let maxAbsoluteError = 0
  for (let index = 0; index < actual.length; index++) {
    assert(Number.isFinite(actual[index]), `non-finite embedding at ${index}`)
    const error = Math.abs(actual[index] - expected[index])
    maxAbsoluteError = Math.max(maxAbsoluteError, error)
    assert(error <= 2e-4 + 2e-4 * Math.abs(expected[index]), `mismatch at ${index}: ${error}`)
  }
  // Exercise the audio frontend as well as the exact intermediate comparison.
  const pcm = Float32Array.from({ length: 12000 }, (_, index) =>
    0.1 * Math.sin(2 * Math.PI * 220 * index / 24000))
  const fromPcm = encoder.extractEmbedding(pcm, 24000)
  assert.equal(fromPcm.length, actual.length)
  assert(fromPcm.every(Number.isFinite))
  console.log(JSON.stringify({
    status: 'passed',
    backend: 'wasm32-cpu',
    component: 'qwen-speaker-encoder',
    elements: actual.length,
    max_absolute_error: maxAbsoluteError,
    atol: 2e-4,
    rtol: 2e-4,
    pcm_frontend_finite: true,
  }, null, 2))
} finally {
  encoder.free()
}
