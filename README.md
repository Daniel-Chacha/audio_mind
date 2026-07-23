# AudioMind

AudioMind classifies a music clip into one of 10 genres (blues, classical,
country, disco, hiphop, jazz, metal, pop, reggae, rock) from its mel
spectrogram, using a small CNN trained on GTZAN in Keras/TensorFlow.

**The model runs entirely in your browser.** There is no backend: audio never
leaves your machine, there's no server to deploy or pay for, and inference is
instant.

- **The model** — trained in a Colab notebook (`index.ipynb`), exported as
  `best_model.keras`, then converted to a TensorFlow.js GraphModel that ships in
  `web/public/model/` (~0.42 MB).
- **The app** — `web/`, a Next.js frontend where you drop in a clip (or record
  3 s from the mic, or click one of 10 bundled sample clips) and see the
  predicted genre, a confidence bar per genre, and the spectrogram the model
  actually saw.

For the model design/roadmap, see [`docs/AudioMind-design.md`](docs/AudioMind-design.md).
For the frontend spec and implementation plan, see
[`docs/superpowers/specs/`](docs/superpowers/specs/) and
[`docs/superpowers/plans/`](docs/superpowers/plans/) — note those describe the
original client/server design, before inference moved into the browser. For an
early static mockup of the UI, see
[`docs/prototype/audiomind-prototype.html`](docs/prototype/audiomind-prototype.html).

## Quickstart

```bash
cd web
npm install
npm run dev     # http://localhost:3000
```

That's the whole thing — one process, no API to start.

## How the browser reproduces the training pipeline

The model was trained on log-mel spectrograms built by librosa, so the browser
has to produce the *same* input or predictions drift:

```text
audio file ──decodeAudioData──▶ mono 22.05 kHz
           ──▶ non-overlapping 3 s segments (66,150 samples)
           ──▶ STFT (n_fft 2048, hop 512, librosa's Hann window, center/zero-pad)
           ──▶ |X|² ──▶ × librosa's Slaney mel filterbank (128 bands)
           ──▶ power_to_db(ref=max, top_db=80) ──▶ (x − mean) / std
           ──▶ tfjs GraphModel ──▶ average softmax over segments
```

The mel filterbank and analysis window are **exported verbatim from librosa**
([`scripts/export_dsp_assets.py`](scripts/export_dsp_assets.py)) rather than
re-derived in JS, which removes the biggest source of drift.

This is verified, not assumed — two tests in `web/lib/__tests__/`:

- `dsp.parity.test.ts` — the JS log-mel matches librosa's reference to **1.7e-4 dB**.
- `classifier.parity.test.ts` — the full browser pipeline reproduces the original
  **Keras model's probabilities** on a real clip (same top genre, diff < 2e-3).

## Tests

```bash
cd web && npm run test
```

## Deploying

Static deploy to Vercel (Root Directory = `web`) — see [`DEPLOY.md`](DEPLOY.md).

## Repo layout

```text
best_model.keras     trained Keras model (source for the tfjs conversion)
index.ipynb          training notebook (Colab)
scripts/             DSP asset export + parity fixtures
web/                 Next.js app (UI + in-browser inference) — see web/README.md
docs/                design doc, specs/plans, UI prototype
```
