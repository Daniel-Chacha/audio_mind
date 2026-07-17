# AudioMind Frontend — Design Spec

**Date:** 2026-07-17
**Status:** Approved visual direction; ready for implementation planning.
**Prototype:** [Interactive Studio/DAW-dark prototype](https://claude.ai/code/artifact/37105396-16d6-462a-b9e8-e6fb485aecfb) (private)

A web frontend for the **AudioMind** music-genre classifier: a user hands the app an
audio clip (upload or 3-second mic recording) and gets back the predicted GTZAN genre,
a confidence breakdown across all 10 genres, and the mel spectrogram the model actually
consumed. Built as a portfolio/demo piece — the look and motion carry the impact.

---

## 1. Goal & scope

- **In scope:** a single-screen "analyzer console" that classifies one clip at a time,
  wired to serve real predictions from `best_model.keras`.
- **Deliverable ordering:** the visual prototype (done, approved) → this spec → implementation plan → build.
- **Success:** dropping a real `.wav`/`.mp3` returns the *same* genre and probabilities the
  notebook's `classify_clip()` produces for that file (parity is the correctness bar), inside
  the approved Studio/DAW interface.

**Out of scope (Phase 2, noted in §10):** classification history, batch upload, an
"about the model" page, user accounts, persistence.

---

## 2. Architecture

Two services. The language gap is real — the model + `librosa` preprocessing are Python,
the UI is JavaScript — so a thin Python serving layer owns everything model-related and the
Next.js app stays a pure client of it.

```text
┌─────────────────────┐        multipart POST /classify        ┌──────────────────────────┐
│  Next.js frontend   │  ───────  audio file (wav/mp3)  ──────▶ │  FastAPI serving layer   │
│  (App Router)       │                                         │  • librosa preprocessing │
│  • upload / record  │ ◀──────  JSON: genre, probs[10],  ───── │  • best_model.keras      │
│  • state machine    │          spectrogram, meta              │  • train mean/std        │
│  • canvas spectro   │                                         └──────────────────────────┘
│  • confidence bars  │
└─────────────────────┘
```

**Why a separate FastAPI service (not a Next.js Python route or in-browser TF.js):** the
preprocessing must match training bit-for-bit, and `librosa`'s mel/STFT + `power_to_db`
are painful to reproduce in JS. Keeping Python authoritative removes an entire class of
"prediction is subtly wrong" bugs. The frontend never touches the model.

### Repository layout (monorepo)

```text
audio_mind/
├── index.ipynb              # existing: training notebook (source of truth for preprocessing)
├── best_model.keras         # existing: trained CNN
├── serving/                 # NEW — FastAPI service
│   ├── app.py               #   routes: GET /health, POST /classify
│   ├── inference.py         #   load model + stats; preprocess; predict (mirrors classify_clip)
│   ├── norm_stats.json      #   { "mean": <float>, "std": <float> }  ← exported from notebook
│   ├── requirements.txt     #   fastapi, uvicorn, librosa, tensorflow, python-multipart, numpy
│   └── tests/
│       └── test_parity.py   #   API output == notebook classify_clip() for a fixture clip
└── web/                     # NEW — Next.js app
    ├── app/                 #   App Router: single page
    ├── components/          #   Dropzone, Analyzer, SpectrogramCanvas, ConfidenceBars, ...
    ├── lib/                 #   api client, magma colormap, recording, tokens
    └── ...
```

---

## 3. The serving layer (FastAPI)

### 3.1 Preprocessing parity — the critical contract

The endpoint MUST reproduce the notebook's `classify_clip()` (cell 22) exactly. Constants
(notebook cell 3):

| Constant      | Value   |
| ------------- | ------- |
| `SR`          | 22050   |
| `DURATION`    | 3 s     |
| `N_FFT`       | 2048    |
| `HOP_LENGTH`  | 512     |
| `N_MELS`      | 128     |
| segment shape | (128, 130) |
| model input   | (N, 128, 130, 1) |
| `GENRES`      | blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock (index order fixed) |

Pipeline, per request:

1. `y, _ = librosa.load(file, sr=22050)` — mono; decodes wav/mp3/flac (needs `ffmpeg`/`audioread`).
2. `sps = 22050 * 3 = 66150`; `n_seg = len(y) // sps`. If `n_seg == 0` → clip too short → error (§3.4).
3. For each of the `n_seg` non-overlapping segments:
   `mel = librosa.feature.melspectrogram(y=seg, sr=22050, n_fft=2048, hop_length=512, n_mels=128)`
   → `mel_db = librosa.power_to_db(mel, ref=np.max)` → shape (128, 130).
4. Stack → `X` shape `(n_seg, 128, 130, 1)`.
5. **Normalize:** `X = (X - mean) / std` using the **train** scalars.
6. `probs = model.predict(X).mean(axis=0)` — average softmax across the clip's segments.
7. `genre = GENRES[probs.argmax()]`.

> **Non-negotiable dependency — `mean` & `std`.** These two floats were computed on the
> training set only (notebook cell 14) and saved into `gtzan_features.npz`. That file does
> **not** exist locally (Colab-only; `*.npz` is gitignored). **Required setup step:** in the
> notebook, export the two scalars to `serving/norm_stats.json`, e.g.
> `json.dump({"mean": float(mean), "std": float(std)}, open("serving/norm_stats.json","w"))`.
> The service loads this at startup and fails fast if it's missing. Skipping this = every
> prediction is wrong.

### 3.2 `POST /classify`

**Request:** `multipart/form-data`, field `file` = the audio blob (wav/mp3/flac).

**Response:** `200 application/json`

```jsonc
{
  "genre": "hiphop",                       // top-1 label
  "confidence": 0.873,                      // probs[argmax]
  "probabilities": [                        // ALL 10, in fixed GENRES order
    {"genre": "blues", "prob": 0.011}, ...  // frontend sorts for display
  ],
  "spectrogram": {                          // for "what the model sees"
    "bands": 128,
    "frames": 130,
    "data": [[...], ...]                    // loudest segment's mel_db, min-max rescaled
                                            // to uint8 0..255, row-major (128 rows × 130 cols)
  },
  "meta": { "segments": 10, "duration_s": 30.0, "sample_rate": 22050 }
}
```

Notes:
- `probabilities` is returned in fixed genre-index order; the **frontend** sorts descending
  and flags the winner. Keeps the API dumb and the ordering logic in one place.
- `spectrogram.data` is the loudest (or first) segment's `mel_db`, min-max rescaled to `0..1`
  so the frontend renders the *real* model input through the magma colormap — replacing the
  prototype's synthetic spectrogram. Sent as a compact array (quantized to `uint8` 0–255 to
  keep payload small: 128×130 ≈ 16 KB).

### 3.3 `GET /health`

`200 { "status": "ok", "model_loaded": true, "genres": [...] }` — used by the frontend to
show the "model ready" LED and to fail loudly if the service is down.

### 3.4 Error handling

| Case                         | Status | Body                                                        |
| ---------------------------- | ------ | ----------------------------------------------------------- |
| Clip shorter than 3 s        | 422    | `{"error": "clip_too_short", "message": "Clips need to be at least 3 seconds."}` |
| Unreadable / not audio       | 422    | `{"error": "unreadable_audio", "message": "Couldn't read that file — try a WAV or MP3."}` |
| No file field                | 400    | `{"error": "no_file", "message": "Attach an audio file."}`  |
| Model/stats missing at boot  | 503    | `{"error": "model_unavailable", "message": "The model isn't loaded."}` |

Model + stats load **once at startup**, not per request. CORS enabled for the web origin.

---

## 4. The frontend (Next.js)

Executes the approved prototype as a real app. App Router, TypeScript, one page.

### 4.1 State machine

`idle → (uploading | recording) → analyzing → result` with a parallel `error` state
reachable from any request. This mirrors the prototype's panel transitions and the top
signal-chain strip (`waveform → mel spectrogram → cnn → genre`).

```text
idle ──upload/drop──▶ analyzing ──200──▶ result ──"analyze another"──▶ idle
  └────record 3s────▶ analyzing ──4xx/5xx──▶ error ──retry──▶ idle
```

The prototype fakes `analyzing` with a timer. The real app runs the spectrogram build as a
**client-side animation over the actual request**: the magma columns paint/scan while
`POST /classify` is in flight, then snap to the API-returned spectrogram on response.
(Design choice: the build animation is illustrative motion, not a literal frame-by-frame
feed from the server — the API returns one representative segment, not a live stream.)

### 4.2 Component breakdown (each a single-purpose unit)

| Component            | Job | Depends on |
| -------------------- | --- | ---------- |
| `AnalyzerConsole`    | owns the state machine, renders the active panel | the pieces below |
| `Dropzone`           | drag/drop + Upload button + Record button; emits a File/Blob | `useRecorder` |
| `useRecorder` (hook) | mic capture via `MediaRecorder`, 3s countdown, returns a Blob | browser API |
| `AnalyzingView`      | magma build animation + mono frame readout + progress meter | `SpectrogramCanvas`, `magma` |
| `SpectrogramCanvas`  | draws a 128×130 mel grid (or synthetic during build) through the magma colormap | `lib/magma` |
| `PredictionView`     | genre verdict + confidence + "what the model sees" + reset | `ConfidenceBars`, `SpectrogramCanvas` |
| `ConfidenceBars`     | sorts 10 probs desc, animates bars, highlights winner | — |
| `SignalChain`        | the top pipeline strip; highlights the active stage | state |
| `lib/api`            | `classify(file) → response`; typed; maps error codes to messages | fetch |
| `lib/magma`          | `magma(t) → [r,g,b]`; the shared colormap (UI accent + spectrogram both derive from it) | — |

Rationale for the split: canvas drawing (`SpectrogramCanvas`), audio capture (`useRecorder`),
and network (`lib/api`) are the three things most likely to change or break, so each is
isolated behind a small interface and testable on its own.

### 4.3 Mic recording

`useRecorder` uses `navigator.mediaDevices.getUserMedia({audio:true})` + `MediaRecorder`,
captures ~3 s (matching the model's minimum window), stops, and yields a `Blob` sent to
`/classify` identically to an uploaded file. Handle permission-denied → `error` state with a
clear message. The prototype's countdown + red input meter is the visual for this.

### 4.4 Real spectrogram rendering

On `200`, decode `spectrogram.data` (128×130, 0–255) and draw it via `SpectrogramCanvas`
through `magma(t)`. This is the parity payoff: the "what the model sees" image is the
model's actual input, not a decoration.

### 4.5 Design tokens (from the approved prototype)

Committed **dark, single-theme** — a deliberate choice (a DAW is dark), not an omission.

```text
--bg #0B0D14   --panel #141824   --line #242A3A   --text #E9EBF3   --dim #99A0B6
signal gradient (magma-derived): #7B4DFF → #E8368F → #FF8A3C → #FCD34D
semantic: --ready #38E8B0   --rec #FF3B57
type: mono = ui-monospace/SF Mono/JetBrains Mono (brand voice: wordmark, readouts, verdict, numbers)
      sans = system-ui stack (body copy)
```

Reserve the full gradient for the spectrogram and the winning genre; everything else stays
quiet. Carry over from the prototype: orchestrated load-in, the spectrogram scan build,
staggered confidence-bar growth, hover readouts — all gated behind `prefers-reduced-motion`,
with visible keyboard focus and real `<button>` semantics.

---

## 5. End-to-end data flow

```text
user drops clip
  → Dropzone emits File
  → AnalyzerConsole → analyzing; AnalyzingView starts magma build
  → lib/api.classify(File)  ──POST /classify──▶  FastAPI
        load(sr=22050) → 3s segments → mel → power_to_db(ref=max) → (X-mean)/std
        → model.predict().mean(axis=0) → probs[10] + representative spectrogram
  ◀── JSON ──
  → PredictionView: verdict + ConfidenceBars(sorted) + real SpectrogramCanvas
```

---

## 6. Testing strategy

- **API parity test (most important):** `serving/tests/test_parity.py` runs a fixture clip
  through the endpoint and asserts `probabilities` match `classify_clip()` from the notebook
  within tolerance. This is the guard that the service and the model agree.
- **API unit:** short-clip → 422, non-audio → 422, response schema shape.
- **Frontend component:** `ConfidenceBars` sorts + flags winner correctly; `lib/api` maps
  each error code to the right message; state machine transitions (idle→analyzing→result,
  and →error on failure).
- **Frontend interaction:** upload path and record path both reach `result`; reduced-motion
  path renders the final state without animation.

---

## 7. Error handling & edge cases (summary)

- Clip < 3 s, unreadable file, no file → typed 4xx with user-facing messages (§3.4), surfaced
  in the frontend `error` state with a retry back to `idle`.
- Service down / network error → `error` state; the "model ready" LED reflects `/health`.
- Mic permission denied → `error` state with guidance.
- Large files → cap accepted size client-side; only the first N segments are needed anyway.

---

## 8. Copy (audio-engineering register)

Eyebrow `NEURAL GENRE ANALYSIS`; dropzone "Drop a track to classify" / "or record a
3-second clip"; buttons "Upload audio", "Record 3s"; analyzing "extracting mel spectrogram";
result labels "prediction", "confidence · all 10 genres", "what the model sees";
reset "Analyze another". Errors explain + fix, never apologize.

---

## 9. Assumptions & open decisions

- **Deployment target** is unspecified. Assume local/dev first (`uvicorn` + `next dev`),
  containerize later. Not blocking the design.
- **Representative spectrogram** = loudest segment. (Alternative: first segment, or per-segment
  mean.) Loudest is the most visually characteristic; revisit if it looks noisy.
- The `analyzing` animation is illustrative, decoupled from server timing (§4.1).

---

## 10. Phase 2 (later, not now)

Classification history, batch upload, an "about the model / confusion matrix" page (great
portfolio material — reuses the notebook's confusion matrix), shareable result links.
