import * as tf from "@tensorflow/tfjs";
import { decodeToMono } from "./audio";
import {
  loadDspAssets,
  melDbToDisplayGrid,
  melSpectrogramDb,
  segmentWaveform,
  type DspAssets,
} from "./dsp";
import type { ClassifyResponse } from "./types";

/**
 * In-browser genre classification. The model runs client-side via TensorFlow.js,
 * so there is no backend: audio never leaves the machine and there is nothing to
 * deploy, wake up, or pay for.
 *
 * The log-mel pipeline in ./dsp mirrors librosa (verified against a fixture in
 * lib/__tests__/dsp.parity.test.ts); the model itself is a tfjs GraphModel
 * converted from best_model.keras.
 */

export class ClassifyError extends Error {
  code: string;
  constructor(code: string, message: string) {
    super(message);
    this.code = code;
    this.name = "ClassifyError";
  }
}

let modelPromise: Promise<tf.GraphModel> | null = null;

/** Loads (once) the converted GraphModel. */
export function loadModel(basePath = "/model"): Promise<tf.GraphModel> {
  if (!modelPromise) {
    modelPromise = tf.loadGraphModel(`${basePath}/model.json`).catch(() => {
      modelPromise = null; // let a later attempt retry
      throw new ClassifyError("model_unavailable", "Couldn't load the model. Reload and try again.");
    });
  }
  return modelPromise;
}

/** Warm the model + DSP assets so the first classification isn't the slow one. */
export async function warmUp(): Promise<void> {
  await Promise.all([loadModel(), loadDspAssets()]);
}

export interface ClassifyDeps {
  model?: tf.GraphModel;
  assets?: DspAssets;
}

/**
 * Classify mono audio already at the model's sample rate.
 * Split out from `classify` so it can be exercised without the Web Audio API.
 */
export async function classifySamples(
  y: Float32Array,
  deps: ClassifyDeps = {},
): Promise<ClassifyResponse> {
  const assets = deps.assets ?? (await loadDspAssets());
  const { meta } = assets;

  const segments = segmentWaveform(y, meta.samples_per_segment);
  if (segments.length === 0) {
    throw new ClassifyError("clip_too_short", "Clips need to be at least 3 seconds.");
  }

  const model = deps.model ?? (await loadModel());

  // Unnormalized log-mel per segment: needed both as model input (after
  // normalizing) and to pick the loudest segment for the display spectrogram.
  const melDbs = segments.map((seg) => melSpectrogramDb(seg, assets));

  try {
    const probs = tf.tidy(() => {
      const batch = tf.stack(
        melDbs.map((m) => tf.sub(m, meta.norm_mean).div(meta.norm_std).expandDims(-1)),
      ) as tf.Tensor4D; // [n_seg, n_mels, frames, 1]
      const out = model.predict(batch) as tf.Tensor2D; // [n_seg, 10]
      return out.mean(0) as tf.Tensor1D; // average over the clip's segments
    });
    const probsArr = Array.from(await probs.data());
    probs.dispose();

    // Loudest segment (max mean dB), matching how the reference pipeline chose
    // the spectrogram to display.
    const means = await Promise.all(melDbs.map((m) => m.mean().data()));
    const loudest = means.reduce((best, v, i) => (v[0] > means[best][0] ? i : best), 0);
    const grid = melDbToDisplayGrid(
      (await melDbs[loudest].data()) as Float32Array,
      meta.n_mels,
      meta.frames,
    );

    const top = probsArr.indexOf(Math.max(...probsArr));
    return {
      genre: meta.genres[top],
      confidence: probsArr[top],
      probabilities: meta.genres.map((genre, i) => ({ genre, prob: probsArr[i] })),
      spectrogram: { bands: meta.n_mels, frames: meta.frames, data: grid },
      meta: {
        segments: segments.length,
        duration_s: Math.round((y.length / meta.sr) * 100) / 100,
        sample_rate: meta.sr,
      },
    };
  } finally {
    melDbs.forEach((m) => m.dispose());
  }
}

/** Classify an audio file (or recorded blob) entirely in the browser. */
export async function classify(input: Blob): Promise<ClassifyResponse> {
  const assets = await loadDspAssets();
  let samples: Float32Array;
  try {
    samples = await decodeToMono(input, assets.meta.sr);
  } catch {
    throw new ClassifyError("unreadable_audio", "Couldn't read that file — try a WAV or MP3.");
  }
  return classifySamples(samples, { assets });
}
