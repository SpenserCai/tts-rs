'use strict'

const native = require('./_native.js')
const { createBackendLoader } = require('./backends.cjs')
const loader = createBackendLoader(native)
const session = Symbol('SpeechFoundry session')

class Engine {
  #engine
  #backend

  constructor(token, engine, backend) {
    if (token !== session) throw new TypeError('Use loadEngine() to create a model session')
    this.#engine = engine
    this.#backend = backend
  }

  get backend() { return this.#backend }

  extractVoice(samples, sampleRate, transcript) {
    return this.#engine.extractVoice(samples, sampleRate, transcript)
  }

  synthesize(text, voicePromptJson, optionsJson) {
    return this.#engine.synthesize(text, voicePromptJson, optionsJson)
  }
}

async function loadEngine(modelDir, model, device = 'cpu', options = {}) {
  const { id, binding } = loader.select(device, options.backend)
  return new Engine(session, await binding.loadEngine(modelDir, model, device), id)
}

// Explicit exports preserve CommonJS and Node's ESM named-import discovery.
exports.Engine = Engine
exports.loadEngine = loadEngine
exports.installedBackends = loader.installedBackends
exports.resampleAudio = native.resampleAudio
exports.validateVoicePrompt = native.validateVoicePrompt
exports.version = require('./package.json').version
