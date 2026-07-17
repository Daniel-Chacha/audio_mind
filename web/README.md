# AudioMind — web

Next.js frontend for AudioMind, a music-genre classifier. Drop in an audio
clip (or record 3 seconds from the mic) and the app sends it to the AudioMind
serving API, then renders the predicted genre, a confidence bar per genre,
and the mel-spectrogram the model actually saw.

This app is the UI half of a two-service project — see the [repo root
README](../README.md) for how it fits together with `serving/`.

## Prerequisites

- Node.js 20+ (Next.js 16 / React 19)
- The [serving API](../serving/README.md) running locally for live
  predictions (not required to run `npm run dev` or `npm run build`
  themselves — only for the app to get real responses instead of network
  errors)

## Setup

```bash
cd web
npm install
```

## Running

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). By default the app
talks to the serving API at `http://localhost:8000` (see `NEXT_PUBLIC_API_BASE`
below), so start that service first — see
[`serving/README.md`](../serving/README.md) — or you'll see a "Couldn't reach
the model" error when you try to classify a clip.

## Testing & building

```bash
npm run test    # vitest — component/unit tests, no serving API needed
npm run build   # production build (next build)
npm run start   # serve the production build
npm run lint    # eslint
```

## Configuration

| Env var | Default | Purpose |
| --- | --- | --- |
| `NEXT_PUBLIC_API_BASE` | `http://localhost:8000` | Base URL of the AudioMind serving API. Set this in `web/.env.local` (gitignored) to point at a different host/port, e.g. when the serving API runs elsewhere. |

The default lives in `lib/api.ts`, so the app works out of the box against a
locally running serving API with no `.env.local` file required; only set one
if you need a non-default API URL.

## App structure

- `components/AnalyzerConsole.tsx` — top-level view; owns the
  upload/record/analyze/result state machine (`lib/machine.ts`)
- `components/Dropzone.tsx`, `hooks/useRecorder.ts` — file upload and 3s mic
  recording
- `components/AnalyzingView.tsx`, `components/PredictionView.tsx`,
  `components/ConfidenceBars.tsx`, `components/SpectrogramCanvas.tsx` —
  in-flight animation, result rendering, per-genre confidence bars, and the
  "what the model sees" spectrogram canvas
- `lib/api.ts` — the fetch call to `POST /classify` on the serving API

## Known limitation

Predictions are only as good as `serving/norm_stats.json`. It now holds the
real training-set mean/std exported from the Colab notebook (see
[`serving/README.md`](../serving/README.md)); the request/response pipeline
and UI are fully wired and correct, but real end-to-end prediction accuracy
hasn't yet been visually validated against a live model in a browser.
