# AudioMind Frontend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Next.js frontend that classifies an audio clip into a GTZAN genre by calling a FastAPI service that serves `best_model.keras` with the notebook's exact librosa preprocessing.

**Architecture:** Two services in one repo. `serving/` is a FastAPI app that owns the model + preprocessing and exposes `POST /classify` and `GET /health`. `web/` is a Next.js (App Router, TypeScript) app that captures/uploads audio, calls the API, and renders the approved Studio/DAW-dark interface. The frontend never touches the model; Python stays authoritative for anything that must match training.

**Tech Stack:** Python 3.12 · FastAPI · uvicorn · TensorFlow/Keras · librosa · soundfile · numpy · pytest — and — Next.js (App Router) · TypeScript · CSS Modules · Vitest · Testing Library · HTML Canvas · MediaRecorder.

## Global Constraints

- **Preprocessing constants (verbatim from notebook cell 3, must match training exactly):** `SR=22050`, `DURATION=3`, `N_FFT=2048`, `HOP_LENGTH=512`, `N_MELS=128`, `SAMPLES_PER_SEGMENT = 22050*3 = 66150`, segment shape `(128,130)`, model input `(N,128,130,1)`.
- **Genre order (fixed, index = model output index):** `["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]`.
- **Normalization is train-only mean/std**, exported from the notebook to `serving/norm_stats.json`. The service must load these and fail fast if missing. Never invent them.
- **Preprocessing pipeline (mirrors `classify_clip`, notebook cell 22):** load@22050 mono → non-overlapping 3s segments → `melspectrogram(n_fft,hop_length,n_mels)` → `power_to_db(ref=np.max)` → `(X-mean)/std` → `model.predict(X).mean(axis=0)`.
- **Design source of truth:** `docs/prototype/audiomind-prototype.html`. Its `<style>` block is the canonical CSS (tokens, layout, motion); component tasks port the relevant CSS from it rather than re-deriving pixels. Its `<script>` is the reference for canvas drawing, the magma colormap, and interaction behavior.
- **Design tokens:** `--bg #0B0D14`, `--panel #141824`, `--line #242A3A`, `--text #E9EBF3`, `--dim #99A0B6`; signal gradient `#7B4DFF → #E8368F → #FF8A3C → #FCD34D`; `--ready #38E8B0`, `--rec #FF3B57`. Dark, single-theme (deliberate). Mono = brand voice (wordmark, readouts, verdict, numbers); system sans = body.
- **Accessibility floor (every UI task):** real `<button>` semantics, visible `:focus-visible`, `prefers-reduced-motion` disables the build animation / bar transitions / ticker.
- **Frequent commits:** one commit per completed task minimum.

---

## Phase A — Serving layer (FastAPI)

Directory: `serving/`. Uses its own virtualenv (the repo `venv/` is the Colab-kernel env and lacks TF/librosa). `best_model.keras` lives at repo root and is loaded by path.

### Task A1: Serving scaffold, dependencies, and norm-stats export

**Files:**
- Create: `serving/requirements.txt`
- Create: `serving/constants.py`
- Create: `serving/norm_stats.json` (see step 3 — value comes from the notebook)
- Create: `serving/README.md`
- Test: `serving/tests/test_model_loads.py`

**Interfaces:**
- Produces: `serving/constants.py` exposing `SR, DURATION, N_FFT, HOP_LENGTH, N_MELS, SAMPLES_PER_SEGMENT, GENRES, MODEL_PATH, STATS_PATH`.

- [ ] **Step 1: Write `serving/requirements.txt`**

```text
fastapi==0.115.*
uvicorn[standard]==0.32.*
tensorflow==2.17.*
librosa==0.10.*
soundfile==0.12.*
numpy==1.26.*
python-multipart==0.0.*
pytest==8.*
httpx==0.27.*
```

- [ ] **Step 2: Write `serving/constants.py`**

```python
from pathlib import Path

SR = 22050
DURATION = 3
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
SAMPLES_PER_SEGMENT = SR * DURATION  # 66150

GENRES = ["blues", "classical", "country", "disco", "hiphop",
          "jazz", "metal", "pop", "reggae", "rock"]

_ROOT = Path(__file__).resolve().parent
MODEL_PATH = _ROOT.parent / "best_model.keras"
STATS_PATH = _ROOT / "norm_stats.json"
```

- [ ] **Step 3: Create `serving/norm_stats.json` (export from the notebook)**

The train mean/std were computed in notebook cell 14 and saved only into `gtzan_features.npz` (Colab-only). Run this once in the notebook (after cell 14, or after reloading the npz) and copy the file into `serving/`:

```python
import json
json.dump({"mean": float(mean), "std": float(std)},
          open("norm_stats.json", "w"), indent=2)
# then download norm_stats.json from the Colab kernel into serving/
```

If the notebook isn't runnable right now, create a placeholder so the plumbing/tests proceed, and replace before any real prediction is trusted:

```json
{ "mean": 0.0, "std": 1.0, "_placeholder": true }
```

- [ ] **Step 4: Set up the venv and install**

```bash
cd serving
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

- [ ] **Step 5: Write the failing test `serving/tests/test_model_loads.py`**

```python
from pathlib import Path
import tensorflow as tf
from serving.constants import MODEL_PATH, GENRES

def test_model_file_exists():
    assert Path(MODEL_PATH).exists(), f"missing model at {MODEL_PATH}"

def test_model_loads_with_expected_io():
    model = tf.keras.models.load_model(MODEL_PATH)
    assert model.input_shape == (None, 128, 130, 1)
    assert model.output_shape == (None, len(GENRES))  # 10
```

- [ ] **Step 6: Run and verify it passes** (model is present at repo root)

Run: `cd serving && . .venv/bin/activate && python -m pytest tests/test_model_loads.py -v`
Expected: 2 passed. (Run pytest from the `serving` parent as a module path, i.e. `python -m pytest` with `serving` importable — add `serving/__init__.py` and `serving/tests/__init__.py` if needed.)

- [ ] **Step 7: Write `serving/README.md`** documenting: purpose, venv setup, `norm_stats.json` requirement (with the export snippet from step 3), and `uvicorn serving.app:app --reload` run command.

- [ ] **Step 8: Commit**

```bash
git add serving/
git commit -m "feat(serving): scaffold FastAPI service, constants, model-load test"
```

---

### Task A2: Preprocessing — waveform to log-mel segments

**Files:**
- Create: `serving/preprocess.py`
- Test: `serving/tests/test_preprocess.py`

**Interfaces:**
- Produces:
  - `class ClipTooShortError(ValueError)`
  - `waveform_to_segments(y: np.ndarray) -> np.ndarray` → shape `(n_seg, 128, 130, 1)`, dtype float32, **unnormalized** log-mel dB. Raises `ClipTooShortError` when `len(y) < SAMPLES_PER_SEGMENT`.

- [ ] **Step 1: Write the failing tests `serving/tests/test_preprocess.py`**

```python
import numpy as np
import pytest
from serving.preprocess import waveform_to_segments, ClipTooShortError
from serving.constants import SR, SAMPLES_PER_SEGMENT

def _tone(seconds, freq=220.0):
    t = np.linspace(0, seconds, int(SR * seconds), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)

def test_shape_for_two_segments():
    X = waveform_to_segments(_tone(6.0))          # exactly 2 segments
    assert X.shape == (2, 128, 130, 1)
    assert X.dtype == np.float32

def test_partial_tail_is_dropped():
    X = waveform_to_segments(_tone(7.5))          # 2 full segments + 1.5s tail
    assert X.shape[0] == 2

def test_too_short_raises():
    with pytest.raises(ClipTooShortError):
        waveform_to_segments(_tone(1.0))

def test_values_are_finite_db():
    X = waveform_to_segments(_tone(3.0))
    assert np.isfinite(X).all()
    assert X.max() <= 0.0 + 1e-3                   # power_to_db(ref=max) → dB <= 0
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest serving/tests/test_preprocess.py -v`
Expected: FAIL — `ModuleNotFoundError: serving.preprocess`.

- [ ] **Step 3: Implement `serving/preprocess.py`**

```python
import numpy as np
import librosa
from serving.constants import (
    SR, N_FFT, HOP_LENGTH, N_MELS, SAMPLES_PER_SEGMENT,
)

class ClipTooShortError(ValueError):
    """Raised when a clip has no full 3-second segment."""

def waveform_to_segments(y: np.ndarray) -> np.ndarray:
    n_seg = len(y) // SAMPLES_PER_SEGMENT
    if n_seg == 0:
        raise ClipTooShortError("clip shorter than one 3s segment")
    segs = []
    for i in range(n_seg):
        seg = y[i * SAMPLES_PER_SEGMENT:(i + 1) * SAMPLES_PER_SEGMENT]
        mel = librosa.feature.melspectrogram(
            y=seg, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)   # (128, 130)
        segs.append(mel_db.astype(np.float32))
    X = np.stack(segs)[..., np.newaxis]                 # (n_seg, 128, 130, 1)
    return X
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest serving/tests/test_preprocess.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add serving/preprocess.py serving/tests/test_preprocess.py
git commit -m "feat(serving): waveform to log-mel segment preprocessing"
```

---

### Task A3: Classifier — load, normalize, predict

**Files:**
- Create: `serving/inference.py`
- Test: `serving/tests/test_inference.py`

**Interfaces:**
- Consumes: `waveform_to_segments` (A2), `ClipTooShortError` (A2), constants (A1).
- Produces:
  - `load_norm_stats(path) -> tuple[float, float]` (mean, std).
  - `normalize(X, mean, std) -> np.ndarray`.
  - `class Classifier` with `ready: bool`, `predict(y: np.ndarray) -> np.ndarray` (returns probs shape `(10,)`, sums to 1), and `segments(y) -> np.ndarray` (the unnormalized `(n_seg,128,130,1)` for reuse by the response builder).

- [ ] **Step 1: Write failing tests `serving/tests/test_inference.py`**

```python
import json
import numpy as np
import pytest
from serving.inference import load_norm_stats, normalize, Classifier
from serving.constants import GENRES, MODEL_PATH, SR

def _tone(seconds, freq=220.0):
    t = np.linspace(0, seconds, int(SR * seconds), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)

def test_load_norm_stats(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"mean": 3.0, "std": 2.0}))
    assert load_norm_stats(p) == (3.0, 2.0)

def test_normalize_applies_formula():
    X = np.array([[5.0]], dtype=np.float32)
    out = normalize(X, mean=1.0, std=2.0)
    assert out[0, 0] == pytest.approx(2.0)

def test_predict_returns_probability_vector():
    clf = Classifier(MODEL_PATH, mean=0.0, std=1.0)
    probs = clf.predict(_tone(6.0))
    assert probs.shape == (len(GENRES),)
    assert probs.sum() == pytest.approx(1.0, abs=1e-4)
    assert (probs >= 0).all()
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest serving/tests/test_inference.py -v`
Expected: FAIL — `ModuleNotFoundError: serving.inference`.

- [ ] **Step 3: Implement `serving/inference.py`**

```python
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from serving.preprocess import waveform_to_segments

def load_norm_stats(path) -> tuple[float, float]:
    d = json.loads(Path(path).read_text())
    return float(d["mean"]), float(d["std"])

def normalize(X: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (X - mean) / std

class Classifier:
    def __init__(self, model_path, mean: float, std: float):
        self.model = tf.keras.models.load_model(model_path)
        self.mean = mean
        self.std = std
        self.ready = True

    def segments(self, y: np.ndarray) -> np.ndarray:
        return waveform_to_segments(y)                 # (n_seg,128,130,1) unnormalized

    def predict(self, y: np.ndarray) -> np.ndarray:
        X = normalize(self.segments(y), self.mean, self.std)
        probs = self.model.predict(X, verbose=0).mean(axis=0)   # avg over segments
        return probs.astype(np.float64)
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest serving/tests/test_inference.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add serving/inference.py serving/tests/test_inference.py
git commit -m "feat(serving): Classifier with normalization and segment-averaged predict"
```

---

### Task A4: Response builder + representative spectrogram

**Files:**
- Modify: `serving/inference.py` (add `representative_spectrogram` and `build_result`)
- Test: `serving/tests/test_response.py`

**Interfaces:**
- Consumes: `Classifier.segments`, `Classifier.predict`, `GENRES`.
- Produces:
  - `representative_spectrogram(segments: np.ndarray) -> list[list[int]]` — the **loudest** segment (max mean dB), min-max rescaled to `uint8` 0..255, returned as 128×130 nested ints.
  - `Classifier.classify(y: np.ndarray) -> dict` — the full `/classify` response body (genre, confidence, probabilities[10], spectrogram, meta).

- [ ] **Step 1: Write failing tests `serving/tests/test_response.py`**

```python
import numpy as np
import pytest
from serving.inference import representative_spectrogram, Classifier
from serving.constants import GENRES, MODEL_PATH, SR

def _tone(seconds, freq=220.0):
    t = np.linspace(0, seconds, int(SR * seconds), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)

def test_representative_spectrogram_shape_and_range():
    segs = np.random.rand(3, 128, 130, 1).astype(np.float32)
    grid = representative_spectrogram(segs)
    assert len(grid) == 128 and len(grid[0]) == 130
    flat = [v for row in grid for v in row]
    assert min(flat) >= 0 and max(flat) <= 255
    assert all(isinstance(v, int) for v in flat[:5])

def test_classify_result_schema():
    clf = Classifier(MODEL_PATH, mean=0.0, std=1.0)
    r = clf.classify(_tone(6.0))
    assert r["genre"] in GENRES
    assert 0.0 <= r["confidence"] <= 1.0
    assert len(r["probabilities"]) == len(GENRES)
    assert {p["genre"] for p in r["probabilities"]} == set(GENRES)
    assert r["spectrogram"]["bands"] == 128 and r["spectrogram"]["frames"] == 130
    assert r["meta"]["segments"] == 2
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest serving/tests/test_response.py -v`
Expected: FAIL — `ImportError: cannot import name 'representative_spectrogram'`.

- [ ] **Step 3: Extend `serving/inference.py`**

```python
from serving.constants import GENRES, SR, SAMPLES_PER_SEGMENT

def representative_spectrogram(segments: np.ndarray) -> list[list[int]]:
    idx = int(segments.mean(axis=(1, 2, 3)).argmax())   # loudest segment
    grid = segments[idx, :, :, 0]                        # (128,130) dB
    lo, hi = float(grid.min()), float(grid.max())
    scaled = (grid - lo) / (hi - lo + 1e-9)              # 0..1
    return (scaled * 255).astype(np.uint8).astype(int).tolist()

# add as a method on Classifier:
    def classify(self, y: np.ndarray) -> dict:
        segs = self.segments(y)
        X = normalize(segs, self.mean, self.std)
        probs = self.model.predict(X, verbose=0).mean(axis=0).astype(float)
        top = int(probs.argmax())
        return {
            "genre": GENRES[top],
            "confidence": float(probs[top]),
            "probabilities": [
                {"genre": g, "prob": float(p)} for g, p in zip(GENRES, probs)
            ],
            "spectrogram": {
                "bands": 128, "frames": 130,
                "data": representative_spectrogram(segs),
            },
            "meta": {
                "segments": int(segs.shape[0]),
                "duration_s": round(len(y) / SR, 2),
                "sample_rate": SR,
            },
        }
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest serving/tests/test_response.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add serving/inference.py serving/tests/test_response.py
git commit -m "feat(serving): classify() response body with representative spectrogram"
```

---

### Task A5: FastAPI app — routes, errors, CORS

**Files:**
- Create: `serving/app.py`
- Test: `serving/tests/test_app.py`

**Interfaces:**
- Consumes: `Classifier` (A3/A4), `load_norm_stats` (A3), `ClipTooShortError` (A2), constants.
- Produces: ASGI `app` with `GET /health` and `POST /classify` (field name `file`).

- [ ] **Step 1: Write failing tests `serving/tests/test_app.py`**

```python
import io
import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient
from serving.app import app
from serving.constants import SR, GENRES

client = TestClient(app)

def _wav_bytes(seconds, freq=220.0):
    t = np.linspace(0, seconds, int(SR * seconds), endpoint=False)
    y = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    buf = io.BytesIO(); sf.write(buf, y, SR, format="WAV"); buf.seek(0)
    return buf.read()

def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["model_loaded"] is True
    assert body["genres"] == GENRES

def test_classify_returns_prediction():
    r = client.post("/classify", files={"file": ("clip.wav", _wav_bytes(6.0), "audio/wav")})
    assert r.status_code == 200
    assert r.json()["genre"] in GENRES

def test_classify_too_short_is_422():
    r = client.post("/classify", files={"file": ("s.wav", _wav_bytes(1.0), "audio/wav")})
    assert r.status_code == 422
    assert r.json()["error"] == "clip_too_short"

def test_classify_unreadable_is_422():
    r = client.post("/classify", files={"file": ("x.wav", b"not audio", "audio/wav")})
    assert r.status_code == 422
    assert r.json()["error"] == "unreadable_audio"

def test_classify_no_file_is_422():
    assert client.post("/classify").status_code == 422
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest serving/tests/test_app.py -v`
Expected: FAIL — `ModuleNotFoundError: serving.app`.

- [ ] **Step 3: Implement `serving/app.py`**

```python
import io
import numpy as np
import librosa
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from serving.constants import MODEL_PATH, STATS_PATH, SR, GENRES
from serving.inference import Classifier, load_norm_stats
from serving.preprocess import ClipTooShortError

app = FastAPI(title="AudioMind serving")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_mean, _std = load_norm_stats(STATS_PATH)
clf = Classifier(MODEL_PATH, mean=_mean, std=_std)

def _err(status, code, message):
    return JSONResponse(status_code=status, content={"error": code, "message": message})

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": clf.ready, "genres": GENRES}

@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        return _err(400, "no_file", "Attach an audio file.")
    try:
        y, _ = librosa.load(io.BytesIO(data), sr=SR, mono=True)
    except Exception:
        return _err(422, "unreadable_audio", "Couldn't read that file — try a WAV or MP3.")
    try:
        return clf.classify(y)
    except ClipTooShortError:
        return _err(422, "clip_too_short", "Clips need to be at least 3 seconds.")
```

Note: `librosa.load(BytesIO)` reads WAV/FLAC via soundfile. For MP3 support, install `ffmpeg`/`audioread` in the serving environment (documented in `serving/README.md`); the BytesIO path still applies.

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest serving/tests/test_app.py -v`
Expected: 5 passed. (Requires a real `norm_stats.json`; the placeholder from A1 is fine for these schema/shape tests.)

- [ ] **Step 5: Manual smoke test**

```bash
uvicorn serving.app:app --reload --port 8000
# in another shell:
curl -s localhost:8000/health
```
Expected: `{"status":"ok","model_loaded":true,"genres":[...]}`.

- [ ] **Step 6: Commit**

```bash
git add serving/app.py serving/tests/test_app.py serving/README.md
git commit -m "feat(serving): FastAPI /health and /classify with typed errors and CORS"
```

---

## Phase B — Frontend (Next.js)

Directory: `web/`. App Router + TypeScript + CSS Modules + Vitest. Port CSS from `docs/prototype/audiomind-prototype.html` (canonical). API base URL comes from `NEXT_PUBLIC_API_BASE` (default `http://localhost:8000`).

### Task B1: Next.js scaffold, tooling, tokens, shell

**Files:**
- Create: `web/` via create-next-app (App Router, TS, no Tailwind, no `src/`)
- Create: `web/app/globals.css` (tokens + base), `web/app/layout.tsx`, `web/app/page.tsx`
- Create: `web/vitest.config.ts`, `web/vitest.setup.ts`
- Create: `web/.env.local` with `NEXT_PUBLIC_API_BASE=http://localhost:8000`
- Test: `web/app/__tests__/page.test.tsx`

**Interfaces:**
- Produces: a booting app whose landing renders the top rail wordmark `AUDIOMIND`.

- [ ] **Step 1: Scaffold**

```bash
npx create-next-app@latest web --ts --app --eslint --no-tailwind --no-src-dir --import-alias "@/*"
cd web
npm i -D vitest @vitejs/plugin-react @testing-library/react @testing-library/jest-dom jsdom
```

- [ ] **Step 2: Add `web/vitest.config.ts` and `web/vitest.setup.ts`**

```ts
// vitest.config.ts
import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
export default defineConfig({
  plugins: [react()],
  test: { environment: "jsdom", setupFiles: ["./vitest.setup.ts"], globals: true },
  resolve: { alias: { "@": new URL(".", import.meta.url).pathname } },
});
```

```ts
// vitest.setup.ts
import "@testing-library/jest-dom";
```

Add to `web/package.json` scripts: `"test": "vitest run"`, `"test:watch": "vitest"`.

- [ ] **Step 3: Write `web/app/globals.css`** — port the `:root` token block, base `body`, and utility classes from the prototype's `<style>` (the `--bg … --rec`, `--signal`, `--mono`, `--sans`, and body background rules). Keep it dark, single-theme.

- [ ] **Step 4: Write `web/app/layout.tsx`**

```tsx
import "./globals.css";
export const metadata = { title: "AudioMind — Neural Genre Analysis" };
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (<html lang="en"><body>{children}</body></html>);
}
```

- [ ] **Step 5: Write `web/app/page.tsx`** (temporary shell; replaced in B10)

```tsx
export default function Home() {
  return (<main><header><span className="wordmark">AUDIOMIND</span></header></main>);
}
```

- [ ] **Step 6: Write failing test `web/app/__tests__/page.test.tsx`**

```tsx
import { render, screen } from "@testing-library/react";
import Home from "../page";
test("renders the wordmark", () => {
  render(<Home />);
  expect(screen.getByText("AUDIOMIND")).toBeInTheDocument();
});
```

- [ ] **Step 7: Run tests + dev server**

Run: `npm run test` → 1 passed. `npm run dev` → open localhost:3000, see the wordmark.

- [ ] **Step 8: Commit**

```bash
git add web/
git commit -m "feat(web): scaffold Next.js app, tokens, vitest, app shell"
```

---

### Task B2: Domain modules — genres, types, magma colormap

**Files:**
- Create: `web/lib/genres.ts`, `web/lib/types.ts`, `web/lib/magma.ts`
- Test: `web/lib/__tests__/magma.test.ts`

**Interfaces:**
- Produces:
  - `genres.ts`: `GENRES: string[]` (fixed order), `DISPLAY: Record<string,string>` (e.g. `hiphop → "HIP-HOP"`, others uppercased).
  - `types.ts`: `ClassifyResponse` (`genre`, `confidence`, `probabilities: {genre,prob}[]`, `spectrogram: {bands,frames,data:number[][]}`, `meta`).
  - `magma.ts`: `magma(t: number): [number, number, number]` — clamps `t` to `[0,1]`, interpolates the anchor table.

- [ ] **Step 1: Write failing tests `web/lib/__tests__/magma.test.ts`**

```ts
import { magma } from "../magma";
test("clamps out of range", () => {
  expect(magma(-1)).toEqual([0, 0, 4]);          // first anchor
  expect(magma(2)).toEqual([252, 253, 191]);     // last anchor
});
test("returns integer rgb in range", () => {
  const [r, g, b] = magma(0.5);
  for (const c of [r, g, b]) { expect(c).toBeGreaterThanOrEqual(0); expect(c).toBeLessThanOrEqual(255); expect(Number.isInteger(c)).toBe(true); }
});
```

- [ ] **Step 2: Run to verify failure** — `npx vitest run lib/__tests__/magma.test.ts` → FAIL (no module).

- [ ] **Step 3: Implement the three modules**

```ts
// web/lib/genres.ts
export const GENRES = ["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"];
export const DISPLAY: Record<string,string> = Object.fromEntries(
  GENRES.map(g => [g, g === "hiphop" ? "HIP-HOP" : g.toUpperCase()])
);
```

```ts
// web/lib/types.ts
export interface ClassifyResponse {
  genre: string;
  confidence: number;
  probabilities: { genre: string; prob: number }[];
  spectrogram: { bands: number; frames: number; data: number[][] };
  meta: { segments: number; duration_s: number; sample_rate: number };
}
```

```ts
// web/lib/magma.ts  (anchors copied verbatim from the prototype script)
const MAG: [number,number,number][] = [
  [0,0,4],[28,16,68],[79,18,123],[129,37,129],[181,54,122],
  [229,80,100],[251,135,97],[254,194,135],[252,253,191],
];
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
export function magma(t: number): [number, number, number] {
  if (t < 0) t = 0; if (t > 1) t = 1;
  const s = t * (MAG.length - 1), i = Math.floor(s), f = s - i;
  const a = MAG[i], b = MAG[Math.min(i + 1, MAG.length - 1)];
  return [lerp(a[0],b[0],f)|0, lerp(a[1],b[1],f)|0, lerp(a[2],b[2],f)|0];
}
```

- [ ] **Step 4: Run to verify pass** — `npx vitest run lib/__tests__/magma.test.ts` → 2 passed.

- [ ] **Step 5: Commit**

```bash
git add web/lib/
git commit -m "feat(web): genres, response types, magma colormap"
```

---

### Task B3: API client + error mapping

**Files:**
- Create: `web/lib/api.ts`
- Test: `web/lib/__tests__/api.test.ts`

**Interfaces:**
- Consumes: `ClassifyResponse` (B2).
- Produces:
  - `class ApiError extends Error` with `code: string`.
  - `classify(file: Blob, filename?: string): Promise<ClassifyResponse>` — POSTs multipart to `${NEXT_PUBLIC_API_BASE}/classify`; on non-2xx throws `ApiError` with the server `error` code and `message`; on network failure throws `ApiError("network", ...)`.

- [ ] **Step 1: Write failing tests `web/lib/__tests__/api.test.ts`**

```ts
import { classify, ApiError } from "../api";
import { afterEach, expect, test, vi } from "vitest";
afterEach(() => vi.restoreAllMocks());

test("returns parsed body on 200", async () => {
  const body = { genre: "jazz", confidence: 0.7, probabilities: [], spectrogram: {bands:128,frames:130,data:[]}, meta:{segments:2,duration_s:6,sample_rate:22050} };
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true, json: async () => body }));
  const r = await classify(new Blob(["x"]), "c.wav");
  expect(r.genre).toBe("jazz");
});

test("maps server error to ApiError with code", async () => {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, status: 422, json: async () => ({ error: "clip_too_short", message: "Clips need to be at least 3 seconds." }) }));
  await expect(classify(new Blob(["x"]))).rejects.toMatchObject({ code: "clip_too_short" });
});

test("maps network failure to ApiError('network')", async () => {
  vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("down")));
  await expect(classify(new Blob(["x"]))).rejects.toMatchObject({ code: "network" });
});
```

- [ ] **Step 2: Run to verify failure** — FAIL (no module).

- [ ] **Step 3: Implement `web/lib/api.ts`**

```ts
import type { ClassifyResponse } from "./types";
const BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export class ApiError extends Error {
  code: string;
  constructor(code: string, message: string) { super(message); this.code = code; }
}

export async function classify(file: Blob, filename = "clip.wav"): Promise<ClassifyResponse> {
  const form = new FormData();
  form.append("file", file, filename);
  let res: Response;
  try {
    res = await fetch(`${BASE}/classify`, { method: "POST", body: form });
  } catch {
    throw new ApiError("network", "Couldn't reach the model. Is the service running?");
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new ApiError(body.error ?? "server_error", body.message ?? "Something went wrong.");
  }
  return res.json();
}
```

- [ ] **Step 4: Run to verify pass** — 3 passed.

- [ ] **Step 5: Commit**

```bash
git add web/lib/api.ts web/lib/__tests__/api.test.ts
git commit -m "feat(web): typed API client with error mapping"
```

---

### Task B4: SpectrogramCanvas

**Files:**
- Create: `web/lib/spectro.ts` (pure), `web/components/SpectrogramCanvas.tsx`, `web/components/SpectrogramCanvas.module.css`
- Test: `web/lib/__tests__/spectro.test.ts`

**Interfaces:**
- Consumes: `magma` (B2).
- Produces:
  - `gridToImageData(data: number[][], cellW: number, cellH: number): {width:number; height:number; rgba: Uint8ClampedArray}` — maps a 0..255 grid through magma into an RGBA buffer (pure, testable without a DOM canvas).
  - `<SpectrogramCanvas data={number[][]} />` — draws the grid on a `<canvas>` via `gridToImageData`, `devicePixelRatio`-aware, resizes with its container. (Exact styling from prototype `.see-canvas-wrap`.)

- [ ] **Step 1: Write failing tests `web/lib/__tests__/spectro.test.ts`**

```ts
import { gridToImageData } from "../spectro";
test("produces rgba buffer sized by cells", () => {
  const grid = [[0, 255], [128, 64]];               // 2x2
  const out = gridToImageData(grid, 3, 3);           // 3px per cell → 6x6
  expect(out.width).toBe(6); expect(out.height).toBe(6);
  expect(out.rgba.length).toBe(6 * 6 * 4);
  expect(out.rgba[3]).toBe(255);                      // alpha of first pixel
});
```

- [ ] **Step 2: Run to verify failure** — FAIL.

- [ ] **Step 3: Implement `web/lib/spectro.ts`**

```ts
import { magma } from "./magma";
export function gridToImageData(data: number[][], cellW: number, cellH: number) {
  const rows = data.length, cols = data[0].length;
  const width = cols * cellW, height = rows * cellH;
  const rgba = new Uint8ClampedArray(width * height * 4);
  for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
    const [rr, gg, bb] = magma(data[r][c] / 255);
    for (let dy = 0; dy < cellH; dy++) for (let dx = 0; dx < cellW; dx++) {
      const x = c * cellW + dx, y = r * cellH + dy, i = (y * width + x) * 4;
      rgba[i] = rr; rgba[i+1] = gg; rgba[i+2] = bb; rgba[i+3] = 255;
    }
  }
  return { width, height, rgba };
}
```

- [ ] **Step 4: Implement `web/components/SpectrogramCanvas.tsx`**

```tsx
"use client";
import { useEffect, useRef } from "react";
import { gridToImageData } from "@/lib/spectro";
import styles from "./SpectrogramCanvas.module.css";

export function SpectrogramCanvas({ data }: { data: number[][] }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const cv = ref.current; if (!cv || !data.length) return;
    const rect = cv.getBoundingClientRect();
    const cellW = Math.max(1, Math.round(rect.width / data[0].length));
    const cellH = Math.max(1, Math.round(rect.height / data.length));
    const { width, height, rgba } = gridToImageData(data, cellW, cellH);
    cv.width = width; cv.height = height;
    cv.getContext("2d")!.putImageData(new ImageData(rgba, width, height), 0, 0);
  }, [data]);
  return <canvas ref={ref} className={styles.canvas} />;
}
```

`SpectrogramCanvas.module.css`: `.canvas{ width:100%; height:100%; display:block; }` (wrapper styling lives where the component is used, per prototype).

- [ ] **Step 5: Run to verify pass** — `npx vitest run lib/__tests__/spectro.test.ts` → 1 passed.

- [ ] **Step 6: Commit**

```bash
git add web/lib/spectro.ts web/components/SpectrogramCanvas.*
git commit -m "feat(web): SpectrogramCanvas + pure grid-to-rgba mapping"
```

---

### Task B5: ConfidenceBars

**Files:**
- Create: `web/lib/rank.ts` (pure), `web/components/ConfidenceBars.tsx`, `web/components/ConfidenceBars.module.css`
- Test: `web/lib/__tests__/rank.test.ts`

**Interfaces:**
- Consumes: `DISPLAY` (B2), `ClassifyResponse["probabilities"]`.
- Produces:
  - `rankProbabilities(probs: {genre:string;prob:number}[]): {genre:string;prob:number;win:boolean}[]` — sorted desc, `win=true` on the max only.
  - `<ConfidenceBars probabilities={...} />` — renders one row per genre (label / animated track / value), winner styled with the signal gradient. Bars grow from 0 with staggered delay; reduced-motion shows final width. Port row CSS from prototype `.row`.

- [ ] **Step 1: Write failing tests `web/lib/__tests__/rank.test.ts`**

```ts
import { rankProbabilities } from "../rank";
test("sorts desc and flags a single winner", () => {
  const out = rankProbabilities([
    { genre: "pop", prob: 0.1 }, { genre: "rock", prob: 0.7 }, { genre: "jazz", prob: 0.2 },
  ]);
  expect(out.map(o => o.genre)).toEqual(["rock", "jazz", "pop"]);
  expect(out.filter(o => o.win)).toHaveLength(1);
  expect(out[0].win).toBe(true);
});
```

- [ ] **Step 2: Run to verify failure** — FAIL.

- [ ] **Step 3: Implement `web/lib/rank.ts`**

```ts
export function rankProbabilities(probs: { genre: string; prob: number }[]) {
  const sorted = [...probs].sort((a, b) => b.prob - a.prob);
  const max = sorted.length ? sorted[0].prob : 0;
  return sorted.map((p, i) => ({ ...p, win: i === 0 && p.prob === max }));
}
```

- [ ] **Step 4: Implement `web/components/ConfidenceBars.tsx`** — map `rankProbabilities` to rows using `DISPLAY[genre]`, `(prob*100).toFixed(1)`, `font-variant-numeric: tabular-nums`, staggered `transition-delay`, `.win` gradient. Full row markup mirrors prototype `.row`.

- [ ] **Step 5: Run to verify pass** — 1 passed.

- [ ] **Step 6: Commit**

```bash
git add web/lib/rank.ts web/components/ConfidenceBars.*
git commit -m "feat(web): ConfidenceBars with ranked probabilities"
```

---

### Task B6: SignalChain + Dropzone

**Files:**
- Create: `web/components/SignalChain.tsx` (+ css), `web/components/Dropzone.tsx` (+ css)
- Test: `web/components/__tests__/Dropzone.test.tsx`

**Interfaces:**
- Produces:
  - `<SignalChain active="input"|"analyze"|"result" />` — the pipeline strip, highlights the active stage.
  - `<Dropzone onFile={(file: File) => void} onRecord={() => void} disabled?: boolean />` — Upload button (opens hidden `<input type=file accept="audio/*">`), Record button (calls `onRecord`), and drag/drop that calls `onFile`. Idle magma equalizer canvas is optional polish (port from prototype `idleViz`).

- [ ] **Step 1: Write failing test `web/components/__tests__/Dropzone.test.tsx`**

```tsx
import { render, screen, fireEvent } from "@testing-library/react";
import { Dropzone } from "../Dropzone";
test("emits the chosen file", () => {
  const onFile = vi.fn();
  render(<Dropzone onFile={onFile} onRecord={() => {}} />);
  const input = screen.getByTestId("file-input") as HTMLInputElement;
  const file = new File(["x"], "song.wav", { type: "audio/wav" });
  fireEvent.change(input, { target: { files: [file] } });
  expect(onFile).toHaveBeenCalledWith(file);
});
test("calls onRecord when Record pressed", () => {
  const onRecord = vi.fn();
  render(<Dropzone onFile={() => {}} onRecord={onRecord} />);
  fireEvent.click(screen.getByRole("button", { name: /record/i }));
  expect(onRecord).toHaveBeenCalled();
});
```

- [ ] **Step 2: Run to verify failure** — FAIL.

- [ ] **Step 3: Implement `Dropzone.tsx`**

```tsx
"use client";
import { useRef } from "react";
import styles from "./Dropzone.module.css";

export function Dropzone({ onFile, onRecord, disabled }: {
  onFile: (f: File) => void; onRecord: () => void; disabled?: boolean;
}) {
  const input = useRef<HTMLInputElement>(null);
  return (
    <div
      className={styles.dropzone}
      onDragOver={(e) => e.preventDefault()}
      onDrop={(e) => { e.preventDefault(); const f = e.dataTransfer.files?.[0]; if (f) onFile(f); }}
    >
      <h1 className={styles.title}>Drop a track to classify</h1>
      <p className={styles.sub}>Feed AudioMind a clip and it returns the genre from a 3-second window.</p>
      <div className={styles.actions}>
        <button className="btn btn-primary" disabled={disabled}
          onClick={() => input.current?.click()}>Upload audio</button>
        <span className="or">or</span>
        <button className="btn btn-ghost" disabled={disabled} onClick={onRecord}>Record 3s</button>
        <input ref={input} data-testid="file-input" type="file" accept="audio/*" hidden
          onChange={(e) => { const f = e.target.files?.[0]; if (f) onFile(f); e.target.value = ""; }} />
      </div>
      <div className={styles.foot}>wav · mp3 · flac — resampled to 22.05 kHz mono</div>
    </div>
  );
}
```

`SignalChain.tsx`: render the four stages (`waveform → mel spectrogram → cnn → genre`) with `.on` on the stage matching `active`; port `.chain`/`.stage` CSS from prototype.

- [ ] **Step 4: Run to verify pass** — 2 passed.

- [ ] **Step 5: Commit**

```bash
git add web/components/SignalChain.* web/components/Dropzone.* web/components/__tests__/Dropzone.test.tsx
git commit -m "feat(web): SignalChain and Dropzone (upload + drag/drop + record)"
```

---

### Task B7: useRecorder hook

**Files:**
- Create: `web/hooks/useRecorder.ts`
- Test: `web/hooks/__tests__/useRecorder.test.ts`

**Interfaces:**
- Produces: `useRecorder(): { recording: boolean; countdown: number; start: () => void }` and calls the `onDone(blob: Blob)` passed to it. Captures ~3s via `MediaRecorder`, counts 3→2→1, then yields a `Blob`. Permission errors surface via an `onError(code: "mic_denied")` callback.

- [ ] **Step 1: Write failing test `web/hooks/__tests__/useRecorder.test.ts`** (mock MediaRecorder + getUserMedia + fake timers)

```ts
import { renderHook, act } from "@testing-library/react";
import { useRecorder } from "../useRecorder";
import { beforeEach, expect, test, vi } from "vitest";

beforeEach(() => {
  vi.useFakeTimers();
  const chunks = [new Blob(["a"])];
  class FakeRec { ondataavailable: any; onstop: any; state = "inactive";
    start() { this.state = "recording"; this.ondataavailable?.({ data: chunks[0] }); }
    stop() { this.state = "inactive"; this.onstop?.(); } }
  vi.stubGlobal("MediaRecorder", FakeRec as any);
  vi.stubGlobal("navigator", { mediaDevices: { getUserMedia: vi.fn().mockResolvedValue({ getTracks: () => [{ stop() {} }] }) } });
});

test("counts down and returns a blob", async () => {
  const onDone = vi.fn();
  const { result } = renderHook(() => useRecorder({ onDone, onError: () => {} }));
  await act(async () => { result.current.start(); });
  await act(async () => { vi.advanceTimersByTime(3000); });
  expect(onDone).toHaveBeenCalledWith(expect.any(Blob));
});
```

- [ ] **Step 2: Run to verify failure** — FAIL.

- [ ] **Step 3: Implement `web/hooks/useRecorder.ts`** — `getUserMedia({audio:true})` → `MediaRecorder`, push chunks, `setInterval` countdown 3→0, on 0 `stop()`, assemble `new Blob(chunks, {type})`, call `onDone`; stop tracks; `catch` → `onError("mic_denied")`. Honor reduced-motion by skipping the visible countdown (still records ~3s).

- [ ] **Step 4: Run to verify pass** — 1 passed.

- [ ] **Step 5: Commit**

```bash
git add web/hooks/
git commit -m "feat(web): useRecorder mic-capture hook with countdown"
```

---

### Task B8: AnalyzingView (magma build animation)

**Files:**
- Create: `web/components/AnalyzingView.tsx` (+ css)
- Test: `web/components/__tests__/AnalyzingView.test.tsx`

**Interfaces:**
- Consumes: `SpectrogramCanvas`/`magma`.
- Produces: `<AnalyzingView sourceLabel={string} />` — shows the mono readout (`extracting mel spectrogram`, `frame NNN / 130 · 128 bands · 22.05 kHz`) and a progress meter animating while the request is in flight. The build animation is illustrative (client-side synthetic columns), not a server frame feed. Reduced-motion renders a static filled state.

- [ ] **Step 1: Write failing test** — render `<AnalyzingView sourceLabel="song.wav" />`, assert the text `extracting mel spectrogram` and `song.wav` appear.
- [ ] **Step 2: Run to verify failure** — FAIL.
- [ ] **Step 3: Implement** — canvas synthetic magma build via `requestAnimationFrame` (port `genGrid`/`drawSpectro` from the prototype, or a simplified column sweep); mono readout + `.meter` from prototype. Guard `requestAnimationFrame` behind `!prefers-reduced-motion`.
- [ ] **Step 4: Run to verify pass** — 1 passed.
- [ ] **Step 5: Commit**

```bash
git add web/components/AnalyzingView.*
git commit -m "feat(web): AnalyzingView with magma build animation"
```

---

### Task B9: PredictionView

**Files:**
- Create: `web/components/PredictionView.tsx` (+ css)
- Test: `web/components/__tests__/PredictionView.test.tsx`

**Interfaces:**
- Consumes: `ClassifyResponse` (B2), `SpectrogramCanvas` (B4), `ConfidenceBars` (B5), `DISPLAY` (B2).
- Produces: `<PredictionView result={ClassifyResponse} onReset={() => void} />` — the verdict (`DISPLAY[genre]` in the signal gradient), `(confidence*100).toFixed(1)%`, the "what the model sees" `SpectrogramCanvas` fed `result.spectrogram.data`, `ConfidenceBars`, and an "Analyze another" button.

- [ ] **Step 1: Write failing test** — render with a fixture `ClassifyResponse` (`genre:"hiphop", confidence:0.87`), assert `HIP-HOP` and `87.0%` render, and clicking "Analyze another" calls `onReset`.
- [ ] **Step 2: Run to verify failure** — FAIL.
- [ ] **Step 3: Implement** — compose the three sub-components; port `.result`/`.verdict`/`.genre`/`.see` CSS from prototype (two-column grid, stacks under 720px).
- [ ] **Step 4: Run to verify pass** — passed.
- [ ] **Step 5: Commit**

```bash
git add web/components/PredictionView.*
git commit -m "feat(web): PredictionView (verdict + spectrogram + confidence)"
```

---

### Task B10: AnalyzerConsole state machine + wire-up

**Files:**
- Create: `web/lib/machine.ts` (pure reducer), `web/components/AnalyzerConsole.tsx`
- Modify: `web/app/page.tsx` (render `<AnalyzerConsole />`)
- Test: `web/lib/__tests__/machine.test.ts`

**Interfaces:**
- Consumes: everything above; `classify`/`ApiError` (B3).
- Produces:
  - `type State = {status:"idle"|"analyzing"|"result"|"error"; result?: ClassifyResponse; error?: string; source?: string}`.
  - `reducer(state, action)` with actions `START(source)`, `SUCCESS(result)`, `FAIL(message)`, `RESET`.
  - `<AnalyzerConsole />` — owns the reducer, renders `SignalChain` + the active panel, and on file/record calls `classify(...)`, dispatching `SUCCESS`/`FAIL`.

- [ ] **Step 1: Write failing tests `web/lib/__tests__/machine.test.ts`**

```ts
import { reducer } from "../machine";
const idle = { status: "idle" } as const;
test("START → analyzing carries source", () => {
  expect(reducer(idle, { type: "START", source: "a.wav" })).toMatchObject({ status: "analyzing", source: "a.wav" });
});
test("SUCCESS → result", () => {
  const r: any = { genre: "rock" };
  expect(reducer({ status: "analyzing" }, { type: "SUCCESS", result: r })).toMatchObject({ status: "result", result: r });
});
test("FAIL → error carries message", () => {
  expect(reducer({ status: "analyzing" }, { type: "FAIL", message: "Clips need to be at least 3 seconds." })).toMatchObject({ status: "error", error: "Clips need to be at least 3 seconds." });
});
test("RESET → idle", () => {
  expect(reducer({ status: "result" }, { type: "RESET" })).toEqual({ status: "idle" });
});
```

- [ ] **Step 2: Run to verify failure** — FAIL.

- [ ] **Step 3: Implement `web/lib/machine.ts`**

```ts
import type { ClassifyResponse } from "./types";
export type State =
  | { status: "idle" }
  | { status: "analyzing"; source: string }
  | { status: "result"; result: ClassifyResponse }
  | { status: "error"; error: string };
export type Action =
  | { type: "START"; source: string }
  | { type: "SUCCESS"; result: ClassifyResponse }
  | { type: "FAIL"; message: string }
  | { type: "RESET" };
export function reducer(_state: State, action: Action): State {
  switch (action.type) {
    case "START": return { status: "analyzing", source: action.source };
    case "SUCCESS": return { status: "result", result: action.result };
    case "FAIL": return { status: "error", error: action.message };
    case "RESET": return { status: "idle" };
  }
}
```

- [ ] **Step 4: Implement `web/components/AnalyzerConsole.tsx`** — `useReducer(reducer, {status:"idle"})`; `handleFile(f)` dispatches `START(f.name)` then `classify(f).then(r => SUCCESS).catch(e => FAIL(e.message))`; `handleRecord` uses `useRecorder`; render `SignalChain active={statusToStage}` plus `Dropzone`/`AnalyzingView`/`PredictionView`/error panel by `status`. Port the console shell + footer ticker CSS from prototype.

- [ ] **Step 5: Update `web/app/page.tsx`**

```tsx
import { AnalyzerConsole } from "@/components/AnalyzerConsole";
export default function Home() { return <AnalyzerConsole />; }
```

- [ ] **Step 6: Run to verify pass** — `npm run test` → all green.

- [ ] **Step 7: Commit**

```bash
git add web/lib/machine.ts web/components/AnalyzerConsole.tsx web/app/page.tsx web/lib/__tests__/machine.test.ts
git commit -m "feat(web): AnalyzerConsole state machine wiring the full flow"
```

---

### Task B11: End-to-end verification + README

**Files:**
- Create: `web/README.md`
- Modify: root `README.md` (or create) — how to run both services together.

- [ ] **Step 1: Start both services**

```bash
# terminal 1
cd serving && . .venv/bin/activate && uvicorn serving.app:app --reload --port 8000
# terminal 2
cd web && npm run dev
```

- [ ] **Step 2: Manual end-to-end check** — open localhost:3000; upload a real ≥3s clip (or a GTZAN `.wav`); confirm: analyzing animation runs, a genre + confidence render, the "what the model sees" spectrogram draws, all 10 bars animate. Then test Record 3s (grant mic). Then test a <3s clip → the error panel shows "Clips need to be at least 3 seconds." Then stop the API and upload → "Couldn't reach the model" error.

- [ ] **Step 3: Confirm reduced-motion** — set OS reduce-motion; verify no build sweep / bar animation, final states render.

- [ ] **Step 4: Write both READMEs** — prerequisites, `norm_stats.json` reminder, run commands, the `NEXT_PUBLIC_API_BASE` env var.

- [ ] **Step 5: Commit**

```bash
git add web/README.md README.md
git commit -m "docs: run instructions for serving + web"
```

---

## Self-Review

**Spec coverage:**
- Two-service architecture → Phase A (serving) + Phase B (web). ✓
- Preprocessing parity §3.1 → A2 (segments), A3 (normalize/predict), Global Constraints (verbatim constants). ✓
- `mean/std` from notebook → A1 step 3 (export) + A3 (`load_norm_stats`). ✓
- `POST /classify` contract §3.2 → A4 (response body) + A5 (route). ✓
- Representative spectrogram (loudest, uint8) → A4. ✓
- `GET /health` §3.3 → A5. ✓
- Error cases §3.4 (too-short 422, unreadable 422, no-file → A5 tests; note: FastAPI returns 422 for a missing `file` field, matching the test). ✓
- Frontend state machine §4.1 → B10. ✓
- Component breakdown §4.2 → B2–B9 (each unit its own task). ✓
- Mic recording §4.3 → B7. ✓
- Real spectrogram rendering §4.4 → B4 + B9. ✓
- Design tokens §4.5 → B1 (globals) + prototype-referenced CSS in each UI task. ✓
- Testing strategy §6 → parity/preprocess (A2–A5), component/logic (B2–B10), interaction (B6, B11). ✓
- Copy §8 → surfaced in B6/B9/B10 markup and A5 error messages. ✓

**Note on the parity test (§6):** exact numeric parity against the notebook's `classify_clip()` requires a real GTZAN clip + the true `norm_stats.json`; A2–A5 prove the pipeline reproduces the documented transform and produces well-formed output. Add a fixture-based numeric snapshot in A5 once a real clip + real stats are available (tracked as a follow-up, not a blocker for the plumbing).

**Placeholder scan:** UI-only CSS is delegated to `docs/prototype/audiomind-prototype.html` (a committed, concrete source), not left as "TBD". Logic modules carry full code. No `TODO`/`implement later`.

**Type consistency:** `classify(file, filename)` (B3) used by B10; `ClassifyResponse` shape consistent B2→B3→B9→B10; `rankProbabilities`/`gridToImageData`/`magma`/`reducer` signatures match their consumers; serving `Classifier(model_path, mean, std)` consistent A3→A4→A5.
