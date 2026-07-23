// @vitest-environment node
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import * as tf from "@tensorflow/tfjs";
import { makeDspAssets, melSpectrogramDb, type DspMeta } from "../dsp";

/**
 * The browser must reproduce librosa's log-mel spectrogram, or predictions
 * drift. scripts/export_dsp_assets.py writes a fixture: one real 3s segment and
 * the reference `power_to_db(melspectrogram(...), ref=max)` librosa produced for
 * it. This asserts our TS pipeline matches that reference numerically.
 */

const root = process.cwd(); // vitest runs from web/
const f32 = (p: string) => {
  const b = readFileSync(p);
  return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4);
};

const meta = JSON.parse(
  readFileSync(resolve(root, "public/model/dsp.json"), "utf8"),
) as DspMeta;
const melFilters = f32(resolve(root, "public/model/mel_filters.bin"));
const window = f32(resolve(root, "public/model/hann_window.bin"));
const segment = f32(resolve(root, "../scripts/parity_fixture/segment.bin"));
const referenceMelDb = f32(resolve(root, "../scripts/parity_fixture/mel_db.bin"));

// Slow on purpose: tfjs's pure-JS CPU backend runs the 130-frame FFT here.
// In the browser this path uses WebGL and is fast.
test("log-mel spectrogram matches librosa's reference", { timeout: 180_000 }, async () => {
  await tf.ready();
  const assets = makeDspAssets(meta, melFilters, window);
  const out = melSpectrogramDb(segment, assets);

  expect(out.shape).toEqual([meta.n_mels, meta.frames]);

  const got = await out.data();
  out.dispose();
  expect(got.length).toBe(referenceMelDb.length);

  let maxAbsDiff = 0;
  for (let i = 0; i < got.length; i++) {
    const d = Math.abs(got[i] - referenceMelDb[i]);
    if (d > maxAbsDiff) maxAbsDiff = d;
  }
  // Reported so a regression shows its magnitude, not just a pass/fail.
  console.log(`log-mel max abs diff vs librosa: ${maxAbsDiff.toExponential(3)} dB`);

  // float32 FFT differences only; anything larger means the framing, window,
  // padding, or dB conversion has drifted from librosa.
  expect(maxAbsDiff).toBeLessThan(0.05);
});
