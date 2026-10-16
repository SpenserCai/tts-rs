'use strict'

const { readFileSync } = require('node:fs')

// These are installed build identifiers, not hardware availability probes.
const builds = { cuda: ['cu118', 'cu124', 'cu128'], metal: ['metal'] }

function createBackendLoader(native, {
  version = require('./package.json').version,
  platform = process.platform,
  arch = process.arch,
  resolve = require.resolve,
  read = (path) => JSON.parse(readFileSync(path, 'utf8')),
  load = require,
} = {}) {
  let cudaBackend
  function validate(binding, family) {
    if (typeof binding.backendInfo !== 'function') {
      throw new Error('SpeechFoundry native backend protocol missing; install matching packages')
    }
    const info = binding.backendInfo()
    if (info.version !== version || info.api !== 1) {
      throw new Error('SpeechFoundry native backend version/API mismatch; install matching packages')
    }
    if (family !== 'cpu' && !info[family]) {
      throw new Error(`The selected backend was not compiled for ${family}`)
    }
    return binding
  }
  validate(native, 'cpu')
  const compiled = native.backendInfo()

  function entries(family) {
    const result = new Map()
    for (const id of builds[family]) {
      const name = `@spensercai/speechfoundry-${id}-${platform}-${arch}`
      let path
      try {
        path = resolve(`${name}/package.json`)
      } catch (error) {
        if (error.code === 'MODULE_NOT_FOUND') continue
        throw error
      }
      const metadata = read(path)
      if (metadata.name !== name || metadata.speechfoundryBackend !== id ||
          !metadata.os?.includes(platform) || !metadata.cpu?.includes(arch)) {
        throw new Error(`Invalid SpeechFoundry backend registration: ${name}`)
      }
      result.set(id, { name, version: metadata.version })
    }
    return result
  }

  function installedBackends() {
    return [...new Set(['cpu', ...Object.keys(builds).flatMap(family => [
      ...(compiled[family] ? [family] : []), ...entries(family).keys(),
    ])])].sort()
  }

  function select(device, backend) {
    if (device === 'cpu') {
      if (backend !== undefined && backend !== 'cpu') throw new TypeError("CPU execution uses backend='cpu'")
      return { id: 'cpu', binding: native }
    }
    const match = typeof device === 'string' && /^(cuda|metal)(?::[0-9]+)?$/.exec(device)
    if (!match) throw new TypeError('device must be cpu, cuda:N, or metal:N')
    const family = match[1]
    const candidates = entries(family)
    if (compiled[family]) candidates.set(family, null)
    if (backend === undefined) {
      if (candidates.size !== 1) {
        throw new Error(`Select one ${family} backend with options.backend (${[...candidates.keys()].join(', ') || 'none installed'}); install a matching preview if needed`)
      }
      backend = candidates.keys().next().value
    }
    if (!candidates.has(backend)) throw new TypeError(`Backend ${backend} is not installed for ${family}`)
    const entry = candidates.get(backend)
    if (entry && entry.version !== version) {
      throw new Error(`Backend ${backend} version mismatch; install SpeechFoundry ${version}`)
    }
    if (family === 'cuda') {
      if (cudaBackend !== undefined && cudaBackend !== backend) {
        throw new Error('Use a separate process for a different CUDA build; native library ABIs are not isolated')
      }
      // Retain the reservation after failures: loading can partially initialize native libraries.
      cudaBackend = backend
    }
    let binding
    try {
      binding = entry ? load(entry.name) : native
    } catch (cause) {
      throw new Error(`Cannot load ${backend}: check its native runtime dependencies; no CPU fallback`, { cause })
    }
    return { id: backend, binding: validate(binding, family) }
  }
  return { installedBackends, select }
}

module.exports = { createBackendLoader }
