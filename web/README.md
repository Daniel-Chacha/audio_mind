# AudioMind — web

Next.js frontend for AudioMind, a music-genre classifier. Drop in an audio clip
(record 3 seconds from the mic, or click one of the 10 bundled sample clips) and
the app renders the predicted genre, a confidence bar per genre, and the
mel-spectrogram the model actually saw.

**Inference runs in the browser.** The CNN ships as a TensorFlow.js GraphModel in
`public/model/`, so there's no API to run — no server, no CORS, no env vars, and
the audio never leaves the machine. See the [repo root README](../README.md) for
how the browser reproduces the training pipeline.

## Prerequisites

- Node.js 20+ (Next.js 16 / React 19)

## Setup & running

```bash
npm install
npm run dev     # http://localhost:3000
```

Nothing else to start.

## Testing & building

```bash
npm run test    # vitest — unit, component, and the librosa/Keras parity tests
npm run build   # production build (next build)
npm run start   # serve the production build
npm run lint    # eslint
```

The two parity tests in `lib/__tests__/` are the important ones: they assert the
browser's log-mel spectrogram matches librosa, and that the whole pipeline
reproduces the original Keras model's probabilities. They run on tfjs's pure-JS
CPU backend, so they're slow (~1 min) — in the browser this path uses WebGL.

## App structure

- `components/AnalyzerConsole.tsx` — top-level view; owns the
  upload/record/analyze/result state machine (`lib/machine.ts`)
- `components/Dropzone.tsx`, `components/SamplePicker.tsx`,
  `hooks/useRecorder.ts` — file upload, the one-click sample clips, and 3 s mic
  recording
- `components/AnalyzingView.tsx`, `components/PredictionView.tsx`,
  `components/ConfidenceBars.tsx`, `components/SpectrogramCanvas.tsx` —
  in-flight animation, result rendering, per-genre confidence bars, and the
  "what the model sees" spectrogram canvas
- `lib/classifier.ts` — loads the GraphModel and classifies a clip
- `lib/dsp.ts` — the librosa-equivalent log-mel pipeline
- `lib/audio.ts` — Web Audio decoding/resampling to mono 22.05 kHz

## Generated assets

`public/model/` is generated — the converted model plus librosa's mel filterbank,
analysis window, and normalization stats. Don't hand-edit it; regenerate with
`scripts/export_dsp_assets.py` (see [`DEPLOY.md`](../DEPLOY.md)).
