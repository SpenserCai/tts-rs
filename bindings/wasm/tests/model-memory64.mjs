/** Explicit, opt-in real checkpoint synthesis probe. No downloads or private inputs. */
import { openAsBlob } from 'node:fs'
import { access, readFile, writeFile } from 'node:fs/promises'
import path from 'node:path'
import init, * as bindings from '../pkg-memory64/speechfoundry_wasm.js'
import { loadQwenModel } from '../loader.mjs'
await init({ module_or_path: await readFile(new URL('../pkg-memory64/speechfoundry_wasm_bg.wasm', import.meta.url)) })
const [directory, inputPath, outputPath] = process.argv.slice(2)
if (!directory || !inputPath || !outputPath) throw new Error('Usage: node model-memory64.mjs MODEL INPUT.json OUTPUT.json')
const hasher = new bindings.CheckpointHasher()
const names = JSON.parse(hasher.fileNames()); hasher.free()
const files = {}
for (const name of names) {
  const file = path.join(directory, name)
  try { await access(file) } catch (error) { if (error.code === 'ENOENT') continue; throw error }
  files[name] = await openAsBlob(file)
}
const input = JSON.parse(await readFile(inputPath, 'utf8'))
let count = 0
const started = performance.now()
const model = await loadQwenModel(bindings, files, { onProgress(event) {
  if (event.stage === 'tensor' && ++count % 100 === 0) console.log(JSON.stringify({ tensors: count, rss: process.memoryUsage().rss }))
  if (event.stage === 'ready') console.log('Model ready')
} })
const loaded = performance.now()
try {
  const prompt = model.extractVoice(new Float32Array(input.samples), input.sample_rate, input.transcript)
  const conditioned = performance.now()
  const audio = model.synthesize(input.text, prompt, JSON.stringify(input.options))
  try {
    const samples = audio.samples
    if (!samples.length || !samples.every(Number.isFinite)) throw new Error('Model produced empty or nonfinite PCM')
    const result = {
      schema_version: 1,
      metadata: { implementation: 'speechfoundry-wasm-memory64', address_bits: 64, sample_rate: audio.sampleRate,
        fingerprint: model.modelFingerprint, load_seconds: (loaded - started) / 1000,
        extraction_seconds: (conditioned - loaded) / 1000, synthesis_seconds: (performance.now() - conditioned) / 1000,
        max_rss_kib: process.resourceUsage().maxRSS },
      tensors: { waveform: { shape: [samples.length], dtype: 'float32', data: Array.from(samples) } },
    }
    await writeFile(outputPath, JSON.stringify(result))
    console.log(JSON.stringify(result.metadata))
  } finally { audio.free() }
} finally { model.free() }
