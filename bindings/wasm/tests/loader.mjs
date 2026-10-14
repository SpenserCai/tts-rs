import assert from 'node:assert/strict'
import { test } from 'node:test'
import { tensorDirectory } from '../loader.mjs'

function checkpoint(header, data = new Uint8Array(8)) {
  const encoded = new TextEncoder().encode(JSON.stringify(header))
  const size = new Uint8Array(8)
  new DataView(size.buffer).setBigUint64(0, BigInt(encoded.length), true)
  return new Blob([size, encoded, data])
}

test('tensor directory keeps offsets separate from bounded byte transfers', async () => {
  const result = await tensorDirectory(checkpoint({ weight: { dtype: 'F32', shape: [2], data_offsets: [0, 8] } }))
  assert.equal(result.tensors[0].end, 8)
  assert.equal(result.tensors[0].name, 'weight')
})

test('tensor directory rejects malformed ranges, gaps, overlap and unsupported types', async () => {
  for (const spec of [
    { dtype: 'F32', shape: [3], data_offsets: [0, 8] },
    { dtype: 'F32', shape: [2], data_offsets: [1, 9] },
    { dtype: 'F32', shape: [2], data_offsets: [0, 9] },
    { dtype: 'F32', shape: [Number.MAX_SAFE_INTEGER, 100], data_offsets: [0, 8] },
    { dtype: 'U8', shape: [8], data_offsets: [0, 8] },
  ]) await assert.rejects(tensorDirectory(checkpoint({ weight: spec })))
  await assert.rejects(tensorDirectory(checkpoint({
    a: { dtype: 'F32', shape: [2], data_offsets: [0, 8] },
    b: { dtype: 'F32', shape: [2], data_offsets: [0, 8] },
  })))
})
