const assert = require('node:assert/strict')
const { test } = require('node:test')
const { createBackendLoader } = require('../backends.cjs')

const { version } = require('../package.json')
const binary = (cuda = false, metal = false, revision = version, api = 1) => ({
  backendInfo: () => ({ version: revision, api, cuda, metal }),
})

function fixture(ids = [], binding = binary(true), overrides = {}) {
  const loaded = []
  const packages = new Map(ids.map(id => {
    const name = `@spensercai/speechfoundry-${id}-linux-x64`
    return [`${name}/package.json`, { name, version, speechfoundryBackend: id, os: ['linux'], cpu: ['x64'] }]
  }))
  const loader = createBackendLoader(binary(), {
    version, platform: 'linux', arch: 'x64',
    resolve: (name) => {
      if (packages.has(name)) return name
      throw Object.assign(new Error('not installed'), { code: 'MODULE_NOT_FOUND' })
    },
    read: name => packages.get(name),
    load: name => { loaded.push(name); return binding },
    ...overrides,
  })
  return { loader, loaded, packages }
}

test('CPU does not probe accelerator packages', () => {
  const { loader, loaded } = fixture([], binary(), { resolve() { throw new Error('must not probe') } })
  assert.equal(loader.select('cpu').id, 'cpu')
  assert.deepEqual(loaded, [])
})

test('installed build discovery is metadata-only', () => {
  const { loader, loaded } = fixture(['cu118', 'cu124'])
  assert.deepEqual(loader.installedBackends(), ['cpu', 'cu118', 'cu124'])
  assert.deepEqual(loaded, [])
})

test('only the single selected CUDA backend loads', () => {
  const { loader, loaded } = fixture(['cu124'])
  assert.equal(loader.select('cuda:0').id, 'cu124')
  assert.deepEqual(loaded, ['@spensercai/speechfoundry-cu124-linux-x64'])
})

test('missing and ambiguous backends fail without loading', () => {
  for (const ids of [[], ['cu118', 'cu124']]) {
    const { loader, loaded } = fixture(ids)
    assert.throws(() => loader.select('cuda:0'), /Select one/)
    assert.deepEqual(loaded, [])
  }
})

test('explicit CUDA choice forbids later ABI mixing', () => {
  const { loader, loaded } = fixture(['cu118', 'cu124'])
  assert.equal(loader.select('cuda:1', 'cu124').id, 'cu124')
  assert.throws(() => loader.select('cuda:0', 'cu118'), /separate process/)
  assert.equal(loader.select('cpu').id, 'cpu')
  assert.equal(loaded.length, 1)
})

test('distribution version mismatch fails before import', () => {
  const { loader, loaded, packages } = fixture(['cu124'])
  packages.values().next().value.version = '0.0.1'
  assert.throws(() => loader.select('cuda:0'), /version mismatch/)
  assert.deepEqual(loaded, [])
})

test('native identity, protocol and compiled feature are checked', () => {
  for (const binding of [binary(true, false, '0.0.1'), binary(true, false, version, 99), binary()]) {
    assert.throws(() => fixture(['cu124'], binding).loader.select('cuda:0'), /mismatch|not compiled/)
  }
})

test('load errors do not fall back to CPU or another CUDA build', () => {
  const { loader } = fixture(['cu118', 'cu124'], undefined, { load() { throw new Error('libcuda missing') } })
  assert.throws(() => loader.select('cuda:0', 'cu124'), /no CPU fallback/)
  assert.throws(() => loader.select('cuda:0', 'cu118'), /separate process/)
  assert.equal(loader.select('cpu').id, 'cpu')
})

test('Metal selection uses the same loader', () => {
  assert.equal(fixture(['metal'], binary(false, true)).loader.select('metal:0').id, 'metal')
})

test('invalid devices, selections and registrations reject', () => {
  const { loader, packages } = fixture(['cu124'])
  for (const [device, id] of [['gpu'], ['cuda:-1'], ['cpu', 'cu124'], ['metal:0', 'cu124']]) {
    assert.throws(() => loader.select(device, id), /device|backend|Backend/)
  }
  packages.values().next().value.speechfoundryBackend = 'cu118'
  assert.throws(() => loader.installedBackends(), /registration/)
})

test('source builds can use their compiled accelerator', () => {
  const loader = createBackendLoader(binary(true), { version, resolve() {
    throw Object.assign(new Error('missing'), { code: 'MODULE_NOT_FOUND' })
  } })
  assert.equal(loader.select('cuda:0').id, 'cuda')
})
