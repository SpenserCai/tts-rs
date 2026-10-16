import { Engine, loadEngine, installedBackends, resampleAudio, validateVoicePrompt, version } from '../index'
import type { AudioOutput } from '../index'

async function contract(pcm: Float32Array) {
  const engine: Engine = await loadEngine('models/qwen-base', 'qwen3', 'cuda:0', { backend: 'cu124' })
  const prompt: string = await engine.extractVoice(pcm, 24000, null)
  const audio: AudioOutput = await engine.synthesize('Hello.', prompt, null)
  const rate: number = audio.sampleRate
  const samples: Float32Array = audio.samples
  const builds: string[] = installedBackends()
  const backend: string = engine.backend
  const normalized: string = await validateVoicePrompt(prompt)
  const resampled: AudioOutput = await resampleAudio(samples, rate, 24000)
  return { builds, backend, normalized, resampled, version }
}

void contract
