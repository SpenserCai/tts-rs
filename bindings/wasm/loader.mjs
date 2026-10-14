/** Load explicitly supplied checkpoint files into memory64, one tensor at a time.
 * Files are Blob/File objects. This module does not fetch URLs or read host paths.
 * Run it in a Worker. Model tensor transfers are bounded to 2 GiB; hash chunks to
 * 8 MiB, so no multi-gigabyte checkpoint ArrayBuffer is required.
 */
const HASH_CHUNK = 8 * 1024 * 1024
const MAX_TENSOR = 2 * 1024 * 1024 * 1024
const WIDTH = { BF16: 2, F16: 2, F32: 4 }
const decoder = new TextDecoder('utf-8', { fatal: true })

async function bytes(file, start, length) {
  if (!Number.isSafeInteger(start) || !Number.isSafeInteger(length) || start < 0 || length < 0 || start + length > file.size) {
    throw new RangeError('Invalid checkpoint byte range')
  }
  const result = new Uint8Array(await file.slice(start, start + length).arrayBuffer())
  if (result.byteLength !== length) throw new Error('Truncated checkpoint file')
  return result
}

async function text(file, limit) {
  if (!file || file.size > limit) throw new Error('Missing or oversized checkpoint metadata')
  return decoder.decode(await bytes(file, 0, file.size))
}

/** Parse and validate a safetensors directory without allocating its data payload. */
export async function tensorDirectory(file) {
  const prefix = await bytes(file, 0, 8)
  const headerLength64 = new DataView(prefix.buffer, prefix.byteOffset, 8).getBigUint64(0, true)
  if (headerLength64 > 16n * 1024n * 1024n) throw new Error('Safetensors header exceeds 16 MiB')
  const headerLength = Number(headerLength64)
  const header = JSON.parse(decoder.decode(await bytes(file, 8, headerLength)))
  if (!header || Array.isArray(header) || typeof header !== 'object') throw new Error('Invalid safetensors header')
  const tensors = Object.entries(header).filter(([name]) => name !== '__metadata__').map(([name, spec]) => {
    if (!Object.hasOwn(WIDTH, spec.dtype) || !Array.isArray(spec.shape) || !spec.shape.length || spec.shape.length > 8 || !spec.shape.every(n => Number.isSafeInteger(n) && n > 0)) throw new Error(`Unsupported tensor metadata: ${name}`)
    const length = spec.shape.reduce((n, dim) => n * BigInt(dim), BigInt(WIDTH[spec.dtype]))
    if (length > BigInt(MAX_TENSOR)) throw new Error(`Tensor transfer exceeds 2 GiB: ${name}`)
    if (!Array.isArray(spec.data_offsets) || spec.data_offsets.length !== 2 || !spec.data_offsets.every(n => Number.isSafeInteger(n) && n >= 0)) throw new Error(`Invalid offsets: ${name}`)
    const [start, end] = spec.data_offsets
    if (BigInt(end) - BigInt(start) !== length) throw new Error(`Tensor shape/length mismatch: ${name}`)
    return { name, dtype: spec.dtype, shape: spec.shape, start, end }
  }).sort((a, b) => a.start - b.start)
  let cursor = 0
  for (const tensor of tensors) {
    if (tensor.start !== cursor) throw new Error('Safetensors offsets overlap or leave gaps')
    cursor = tensor.end
  }
  if (cursor + headerLength + 8 !== file.size) throw new Error('Safetensors payload length mismatch')
  return { tensors, dataStart: headerLength + 8 }
}

/**
 * @param {object} bindings Initialized generated memory64 module exports.
 * @param {Record<string, Blob>} files Original checkpoint files by relative name.
 * @param {{ onProgress?: (event: object) => void }} options
 * @returns {Promise<object>} A QwenModel; the caller owns it and must call free().
 */
export async function loadQwenModel(bindings, files, { onProgress = () => {} } = {}) {
  const capabilities = JSON.parse(bindings.capabilities())
  if (capabilities.address_bits !== 64 || !capabilities.speech_synthesis) throw new Error('Qwen synthesis requires the memory64 package')
  for (const name of ['config.json', 'model.safetensors', 'speech_tokenizer/model.safetensors']) {
    if (!Object.hasOwn(files, name)) throw new Error(`Missing checkpoint file: ${name}`)
  }
  const config = await text(files['config.json'], 64 * 1024)
  let builder = Object.hasOwn(files, 'tokenizer.json')
    ? new bindings.QwenModelBuilder(config, await text(files['tokenizer.json'], 32 * 1024 * 1024))
    : bindings.QwenModelBuilder.fromVocabAndMerges(config,
      await text(files['vocab.json'], 32 * 1024 * 1024),
      await text(files['merges.txt'], 16 * 1024 * 1024),
      Object.hasOwn(files, 'tokenizer_config.json') ? await text(files['tokenizer_config.json'], 1024 * 1024) : undefined)
  let hasher = new bindings.CheckpointHasher()
  try {
    for (const name of JSON.parse(hasher.fileNames())) {
      const file = Object.hasOwn(files, name) ? files[name] : undefined
      if (!file) { hasher.skipFile(name); continue }
      hasher.beginFile(name, BigInt(file.size))
      for (let start = 0; start < file.size; start += HASH_CHUNK) {
        hasher.update(await bytes(file, start, Math.min(HASH_CHUNK, file.size - start)))
      }
      onProgress({ stage: 'hashed', file: name })
    }
    const consumedHasher = hasher
    hasher = undefined
    const fingerprint = consumedHasher.finish()
    for (const [name, section] of [['model.safetensors', 'model'], ['speech_tokenizer/model.safetensors', 'codec']]) {
      const file = files[name]
      const { tensors, dataStart } = await tensorDirectory(file)
      for (const tensor of tensors) {
        builder.addTensor(section, tensor.name, tensor.dtype, JSON.stringify(tensor.shape), await bytes(file, dataStart + tensor.start, tensor.end - tensor.start))
        onProgress({ stage: 'tensor', file: name, tensor: tensor.name })
      }
    }
    const consumed = builder
    builder = undefined
    const model = consumed.finish(fingerprint)
    onProgress({ stage: 'ready', fingerprint })
    return model
  } finally {
    hasher?.free()
    builder?.free()
  }
}
