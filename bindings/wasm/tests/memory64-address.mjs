import assert from 'node:assert/strict'
import { test } from 'node:test'
import { readFile } from 'node:fs/promises'
import init, { Audio } from '../pkg-memory64/speechfoundry_wasm.js'

test('Rust allocation and JS typed-array transfer cross the 4 GiB boundary', async () => {
  const exports = await init({ module_or_path: await readFile(new URL('../pkg-memory64/speechfoundry_wasm_bg.wasm', import.meta.url)) })
  const boundary = 2 ** 32
  // Reserve virtual pages without touching their contents: resident memory stays small.
  const pages = Math.ceil((boundary + 65536 - exports.memory.buffer.byteLength) / 65536)
  exports.memory.grow(BigInt(pages))
  const count = 1024 * 1024
  const audio = new Audio(new Float32Array(count).fill(0.25), 24000)
  try {
    assert.ok(audio.__wbg_ptr > boundary, 'the actual Rust object must reside above 4 GiB')
    const samples = audio.samples
    assert.equal(samples.length, count)
    assert.equal(samples[0], 0.25)
    assert.equal(samples[count - 1], 0.25)
  } finally { audio.free() }
})
