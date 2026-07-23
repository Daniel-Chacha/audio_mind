// @vitest-environment node
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import * as tf from "@tensorflow/tfjs";
import { makeDspAssets, type DspMeta } from "../dsp";
import { classifySamples } from "../classifier";

/**
 * End-to-end parity: run the real browser pipeline (librosa-equivalent log-mel
 * -> normalize -> converted tfjs GraphModel) over a real audio segment in Node,
 * and compare against the probabilities the original Keras model produced for
 * the same segment (recorded by scripts/export_dsp_assets.py).
 *
 * This is the check that the app actually predicts what the trained model
 * predicts — not just that the spectrogram looks right.
 */

const root = process.cwd(); // vitest runs from web/
const f32 = (p: string) => {
  const b = readFileSync(p);
  return new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4);
};

const meta = JSON.parse(
  readFileSync(resolve(root, "public/model/dsp.json"), "utf8"),
) as DspMeta;
const fixture = JSON.parse(
  readFileSync(resolve(root, "../scripts/parity_fixture/meta.json"), "utf8"),
) as { keras_probs: number[]; keras_top: string };
const segment = f32(resolve(root, "../scripts/parity_fixture/segment.bin"));

/** Load the converted GraphModel straight off disk (no HTTP, no tfjs-node). */
async function loadModelFromDisk(): Promise<tf.GraphModel> {
  const dir = resolve(root, "public/model");
  const modelJson = JSON.parse(readFileSync(resolve(dir, "model.json"), "utf8"));
  const specs = modelJson.weightsManifest.flatMap(
    (g: { weights: unknown[] }) => g.weights,
  );
  const buffers = modelJson.weightsManifest.flatMap((g: { paths: string[] }) =>
    g.paths.map((p: string) => {
      const b = readFileSync(resolve(dir, p));
      return new Uint8Array(b.buffer, b.byteOffset, b.byteLength);
    }),
  );
  const total = buffers.reduce((n: number, b: Uint8Array) => n + b.byteLength, 0);
  const weightData = new Uint8Array(total);
  let off = 0;
  for (const b of buffers) {
    weightData.set(b, off);
    off += b.byteLength;
  }
  return tf.loadGraphModel({
    load: async () => ({
      modelTopology: modelJson.modelTopology,
      weightSpecs: specs,
      weightData: weightData.buffer,
      format: modelJson.format,
      generatedBy: modelJson.generatedBy,
      convertedBy: modelJson.convertedBy,
      signature: modelJson.signature,
      userDefinedMetadata: modelJson.userDefinedMetadata,
    }),
  } as tf.io.IOHandler);
}

test("browser pipeline reproduces the Keras prediction", { timeout: 300_000 }, async () => {
  await tf.ready();
  const assets = makeDspAssets(
    meta,
    f32(resolve(root, "public/model/mel_filters.bin")),
    f32(resolve(root, "public/model/hann_window.bin")),
  );
  const model = await loadModelFromDisk();

  const result = await classifySamples(segment, { model, assets });

  // Same winning genre as Keras.
  expect(result.genre).toBe(fixture.keras_top);
  expect(result.meta.segments).toBe(1);
  expect(result.probabilities).toHaveLength(10);

  // Probabilities sum to 1 and match Keras closely.
  const sum = result.probabilities.reduce((a, p) => a + p.prob, 0);
  expect(sum).toBeCloseTo(1, 4);

  let maxAbsDiff = 0;
  result.probabilities.forEach((p, i) => {
    const d = Math.abs(p.prob - fixture.keras_probs[i]);
    if (d > maxAbsDiff) maxAbsDiff = d;
  });
  console.log(
    `probability max abs diff vs Keras: ${maxAbsDiff.toExponential(3)} ` +
      `(${result.genre} @ ${(result.confidence * 100).toFixed(2)}%)`,
  );
  expect(maxAbsDiff).toBeLessThan(2e-3);

  // The display spectrogram is a real 128x130 uint8 grid.
  expect(result.spectrogram.data).toHaveLength(meta.n_mels);
  expect(result.spectrogram.data[0]).toHaveLength(meta.frames);
});
