import * as tf from "@tensorflow/tfjs";

/**
 * Browser-side reproduction of the training pipeline's log-mel spectrogram:
 *
 *   librosa.feature.melspectrogram(y, sr, n_fft, hop_length, n_mels)
 *   librosa.power_to_db(mel, ref=np.max)
 *
 * The mel filterbank and analysis window are NOT re-derived here — they are
 * exported verbatim from librosa by scripts/export_dsp_assets.py and shipped as
 * static assets, so the only thing this code has to get right is the framing,
 * the FFT, and the dB conversion. Parity is asserted against a librosa fixture
 * in lib/__tests__/dsp.parity.test.ts.
 */

export interface DspMeta {
  sr: number;
  n_fft: number;
  hop_length: number;
  n_mels: number;
  frames: number;
  samples_per_segment: number;
  mel_filters_shape: [number, number];
  window_length: number;
  center: boolean;
  pad_mode: string;
  power: number;
  top_db: number;
  amin: number;
  norm_mean: number;
  norm_std: number;
  genres: string[];
}

export interface DspAssets {
  meta: DspMeta;
  /** (n_mels, 1 + n_fft/2) Slaney mel filterbank, straight from librosa. */
  melFilters: tf.Tensor2D;
  /** Periodic Hann window of length n_fft, straight from librosa. */
  window: tf.Tensor1D;
}

async function fetchFloat32(url: string): Promise<Float32Array> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`failed to load ${url}`);
  return new Float32Array(await res.arrayBuffer());
}

let assetsPromise: Promise<DspAssets> | null = null;

/** Loads (once) the librosa-exported DSP assets. */
export function loadDspAssets(basePath = "/model"): Promise<DspAssets> {
  if (!assetsPromise) {
    assetsPromise = (async () => {
      const [metaRes, filters, window] = await Promise.all([
        fetch(`${basePath}/dsp.json`),
        fetchFloat32(`${basePath}/mel_filters.bin`),
        fetchFloat32(`${basePath}/hann_window.bin`),
      ]);
      if (!metaRes.ok) throw new Error("failed to load dsp.json");
      const meta = (await metaRes.json()) as DspMeta;
      const [nMels, nBins] = meta.mel_filters_shape;
      return {
        meta,
        melFilters: tf.tensor2d(filters, [nMels, nBins]),
        window: tf.tensor1d(window),
      };
    })();
  }
  return assetsPromise;
}

/** Test seam: build assets from raw buffers instead of fetching them. */
export function makeDspAssets(
  meta: DspMeta,
  melFilters: Float32Array,
  window: Float32Array,
): DspAssets {
  const [nMels, nBins] = meta.mel_filters_shape;
  return {
    meta,
    melFilters: tf.tensor2d(melFilters, [nMels, nBins]),
    window: tf.tensor1d(window),
  };
}

/**
 * One 3-second segment of mono 22.05 kHz audio -> log-mel spectrogram
 * (n_mels x frames), matching librosa's `power_to_db(melspectrogram(...), ref=max)`.
 */
export function melSpectrogramDb(segment: Float32Array, assets: DspAssets): tf.Tensor2D {
  const { meta, melFilters, window } = assets;
  return tf.tidy(() => {
    // librosa's stft uses center=True: pad n_fft/2 on both sides. This librosa
    // version pads with zeros (pad_mode="constant"), which is what we mirror.
    const pad = Math.floor(meta.n_fft / 2);
    const padded = tf.pad(tf.tensor1d(segment), [[pad, pad]]);

    // Framing + FFT. Supplying librosa's own window removes any periodic-vs-
    // symmetric Hann ambiguity.
    const spec = tf.signal.stft(
      padded as tf.Tensor1D,
      meta.n_fft,
      meta.hop_length,
      meta.n_fft,
      () => window.clone(),
    ); // [frames, 1 + n_fft/2] complex

    // Power spectrogram |X|^2 (librosa's melspectrogram default power=2.0).
    const power = tf.square(tf.abs(spec)) as tf.Tensor2D; // [frames, bins]

    // mel = filterbank @ power^T  ->  [n_mels, frames]
    const mel = tf.matMul(melFilters, power, false, true) as tf.Tensor2D;

    // power_to_db(mel, ref=np.max):
    //   S_db = 10*log10(max(amin, S)) - 10*log10(max(amin, ref)), ref = S.max()
    //   S_db = maximum(S_db, S_db.max() - top_db)
    const amin = tf.scalar(meta.amin);
    const ref = tf.maximum(tf.max(mel), amin);
    const log10 = (x: tf.Tensor) => tf.div(tf.log(x), Math.log(10));
    const dbUnclamped = tf.sub(
      tf.mul(log10(tf.maximum(mel, amin)), 10),
      tf.mul(log10(ref), 10),
    );
    const floor = tf.sub(tf.max(dbUnclamped), meta.top_db);
    return tf.maximum(dbUnclamped, floor) as tf.Tensor2D;
  });
}

/** Split a mono waveform into non-overlapping full segments (drops the tail). */
export function segmentWaveform(y: Float32Array, samplesPerSegment: number): Float32Array[] {
  const n = Math.floor(y.length / samplesPerSegment);
  const out: Float32Array[] = [];
  for (let i = 0; i < n; i++) {
    out.push(y.subarray(i * samplesPerSegment, (i + 1) * samplesPerSegment));
  }
  return out;
}

/** Rescale a log-mel grid to uint8 0..255 for display ("what the model sees"). */
export function melDbToDisplayGrid(melDb: Float32Array, rows: number, cols: number): number[][] {
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of melDb) {
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  const span = hi - lo + 1e-9;
  const grid: number[][] = [];
  for (let r = 0; r < rows; r++) {
    const row = new Array<number>(cols);
    for (let c = 0; c < cols; c++) {
      row[c] = Math.round(((melDb[r * cols + c] - lo) / span) * 255);
    }
    grid.push(row);
  }
  return grid;
}
