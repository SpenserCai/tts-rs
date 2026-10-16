import type { AudioOutput } from './_native'
export type { AudioOutput } from './_native'

/** Optional build selection when several CUDA previews are installed. */
export interface LoadOptions { backend?: string }

/** A local model session. Native work runs on libuv workers. */
export declare class Engine {
  private constructor()
  readonly backend: string
  extractVoice(samples: Float32Array, sampleRate: number, transcript?: string | null): Promise<string>
  synthesize(text: string, voicePromptJson: string, optionsJson?: string | null): Promise<AudioOutput>
}

export declare function loadEngine(modelDir: string, model: string, device?: string, options?: LoadOptions): Promise<Engine>
/** Installed builds only; does not load or certify a GPU runtime. */
export declare function installedBackends(): string[]
export { resampleAudio, validateVoicePrompt } from './_native'
export declare const version: string
