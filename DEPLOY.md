# Deploying AudioMind

AudioMind is a **single static Next.js app**. The model runs in the browser via
TensorFlow.js, so there is no API to host — nothing to pay for, wake up, or keep
alive, and no CORS to configure.

## Deploy to Vercel

1. Import the repo at [vercel.com](https://vercel.com) → **Add New → Project**.
2. **Root Directory** → `web` (this is a monorepo; without this Vercel builds the
   repo root, fails to detect Next.js, and errors with *"No Output Directory
   named 'public' found"*).
3. Framework preset auto-detects **Next.js**. No environment variables, no
   build-command overrides.
4. Deploy.

That's it. Vercel serves it over HTTPS, which the mic recorder requires.

Any static host works equally well (Netlify, Cloudflare Pages, GitHub Pages with
a static export) — the app is just files.

## What ships to the browser

| Asset | Size | What it is |
| --- | --- | --- |
| `web/public/model/model.json` + `group1-shard1of1.bin` | ~0.42 MB | the CNN, converted to a tfjs GraphModel |
| `web/public/model/mel_filters.bin` | ~0.5 MB | librosa's Slaney mel filterbank, exported verbatim |
| `web/public/model/hann_window.bin` | 8 KB | librosa's analysis window |
| `web/public/model/dsp.json` | <1 KB | STFT/mel parameters + normalization stats |
| `web/public/samples/*.wav` | ~2.6 MB | the 10 one-click demo clips |

All of it is static and cache-friendly, and it's fetched once.

## Verify after deploying

- Open the site, click a sample chip → a genre and confidence bars appear.
- Upload your own clip (wav/mp3) → same.
- Record 3 s (grant mic access) → same. If the mic button does nothing, the site
  isn't on HTTPS.
- A clip shorter than 3 s → "Clips need to be at least 3 seconds."

## Regenerating the browser assets

Only needed if the model is retrained or the DSP constants change:

```bash
# 1. Convert best_model.keras -> tfjs GraphModel (see the conversion workspace)
#    then copy dist/audiomind/* into web/public/model/
# 2. Re-export the mel filterbank / window / stats (needs librosa + keras):
python scripts/export_dsp_assets.py
# 3. Re-run the parity tests, which compare the browser pipeline against librosa
#    and the original Keras model:
cd web && npm run test
```

Do not hand-edit anything in `web/public/model/` — it is generated.
