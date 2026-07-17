# AudioMind

AudioMind classifies a music clip into one of 10 genres (blues, classical,
country, disco, hiphop, jazz, metal, pop, reggae, rock) from its mel
spectrogram, using a small CNN trained on GTZAN in Keras/TensorFlow. The
project has two parts:

- **The model** — trained in a Colab notebook (`index.ipynb`), exported as
  `best_model.keras` (checked into the repo root).
- **The app** — a two-service web app that serves the model and lets you try
  it on your own audio:
  - `serving/` — a FastAPI service that loads `best_model.keras` and exposes
    `GET /health` and `POST /classify` over HTTP.
  - `web/` — a Next.js frontend where you drag in a clip (or record 3s from
    the mic) and see the predicted genre, a confidence bar per genre, and the
    spectrogram the model actually saw.

For the model design/roadmap, see [`docs/AudioMind-design.md`](docs/AudioMind-design.md).
For the frontend spec and implementation plan, see
[`docs/superpowers/specs/`](docs/superpowers/specs/) and
[`docs/superpowers/plans/`](docs/superpowers/plans/). For an early static
mockup of the UI, see [`docs/prototype/audiomind-prototype.html`](docs/prototype/audiomind-prototype.html).

## Quickstart — run both services

**Terminal 1 — serving API** (Python 3.12; see
[`serving/README.md`](serving/README.md) for full setup):

```bash
cd serving
python3.12 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cd ..
serving/.venv/bin/python -m uvicorn serving.app:app --reload --port 8000   # from repo root; venv path is explicit, no activation needed
```

**Terminal 2 — web frontend** (Node.js 20+; see
[`web/README.md`](web/README.md) for full setup):

```bash
cd web
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). The frontend talks to
the serving API at `http://localhost:8000` by default
(`NEXT_PUBLIC_API_BASE` in `web/lib/api.ts`); both defaults line up so no
extra configuration is needed for local development.

## `norm_stats.json` — real training stats

`serving/norm_stats.json` holds the real training-set mean/std the model was
fit with (`mean = -40.51021194458008`, `std = 15.245159149169922`), taken
from `X_train.mean()` / `X_train.std()` in the training notebook
(`index.ipynb`, cell 27; originally computed in cell 14). The full pipeline
(upload → decode → segment → mel-spectrogram → normalize → predict →
render) runs end-to-end with these values, though real-world end-to-end
prediction accuracy hasn't yet been visually validated against a live model
in a browser.

If the model is ever retrained, re-export `X_train.mean()` / `X_train.std()`
from the notebook and overwrite `serving/norm_stats.json` — see
[`serving/README.md`](serving/README.md#norm_statsjson) for the exact steps.

## Tests

```bash
# serving (from repo root, with the serving venv active or referenced directly)
serving/.venv/bin/python -m pytest serving/tests/ -v

# web
cd web && npm run test
```

## Repo layout

```
best_model.keras     trained model (git-tracked)
index.ipynb          training notebook (Colab)
serving/             FastAPI serving layer — see serving/README.md
web/                 Next.js frontend — see web/README.md
docs/                design doc, specs/plans, UI prototype
```
