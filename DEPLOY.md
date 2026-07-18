# Deploying AudioMind

Two independently-deployed pieces:

- **web/** → **Vercel** (Next.js frontend; free HTTPS, which the mic recorder needs).
- **serving/** → a **container host** (Render / Fly.io / Railway / a VM) running the `Dockerfile` at the repo root. TensorFlow + the model live in memory, so this needs a long-running host — not classic serverless.

They talk over HTTP and must agree on two things:

| Setting | Where | Value |
| --- | --- | --- |
| `NEXT_PUBLIC_API_BASE` | Vercel env (build-time) | the API's public URL, e.g. `https://audiomind-api.onrender.com` |
| `ALLOWED_ORIGINS` | API host env | the web's public URL, e.g. `https://audiomind.vercel.app` |

If these don't match, the browser gets CORS-blocked.

## Recommended order (avoids the chicken-and-egg)

1. **Deploy the API first** so you have its URL. Set `ALLOWED_ORIGINS` to a placeholder for now (you'll correct it in step 3).
2. **Deploy the web app** to Vercel with `NEXT_PUBLIC_API_BASE` = the API URL from step 1. Note the Vercel URL it gives you.
3. **Set `ALLOWED_ORIGINS`** on the API to the Vercel URL from step 2 and redeploy/restart the API.

## 1 & 3 — The API (container host)

The repo-root `Dockerfile` builds the API image (installs `ffmpeg` for mp3 decoding, `libsndfile1`, and uses `tensorflow-cpu` to keep the image lean). Build **from the repo root** (it needs `best_model.keras`). The image is ~2.3 GB (TensorFlow + librosa + ffmpeg), so the first build/deploy takes a few minutes and needs a host that allows multi-GB images (Render/Fly/Railway all do). The container itself loads the model and answers `/health` in ~10 s.

**Render** (example): New → Web Service → connect the repo → Runtime **Docker** → it uses the root `Dockerfile`. Set env vars:

- `ALLOWED_ORIGINS` = your Vercel URL (comma-separate multiple)
- `MAX_UPLOAD_MB` = `15` (optional; default is 15)
- `PORT` is injected by the host — the container already binds `0.0.0.0:$PORT`.

Health check path: **`/health`** (returns `{"status":"ok","model_loaded":true,...}`). Give it a generous start-up window — loading TensorFlow + the model takes a bit.

**Fly.io**: `fly launch` (detects the Dockerfile) → set `internal_port = 8000` in `fly.toml`, then `fly secrets set ALLOWED_ORIGINS=https://audiomind.vercel.app`.

**Railway**: New Project → Deploy from repo → it builds the Dockerfile → add the same env vars.

**Any host, manually:**
```bash
docker build -t audiomind-api .
docker run -p 8000:8000 -e ALLOWED_ORIGINS=https://audiomind.vercel.app audiomind-api
```

## 2 — The web app (Vercel)

Because this is a monorepo, point Vercel at the `web/` subdirectory:

1. Import the repo in Vercel.
2. **Root Directory** → `web`.
3. Framework preset auto-detects **Next.js** (build `next build`, no overrides needed).
4. **Environment Variables** → add `NEXT_PUBLIC_API_BASE` = your API URL. (It's inlined at build time — changing it later requires a redeploy.)
5. Deploy. Vercel serves it over HTTPS, so **Record 3s** (which needs a secure origin) works.

## Local prod-parity

Run the API container exactly as it'll run in prod, with the web dev server against it:
```bash
docker compose up --build      # API → http://localhost:8000
cd web && npm run dev          # web → http://localhost:3000
```

## Verify after deploy

- `curl https://<api-host>/health` → `{"status":"ok","model_loaded":true,...}`
- Open the Vercel URL, upload a ≥3 s clip → a genre + confidence bars render.
- Record 3 s (grant mic) → classifies. If the mic button does nothing, confirm the site is HTTPS.
- Try a <3 s clip → "Clips need to be at least 3 seconds."; stop the API → "Couldn't reach the model."

## Not included (consider before a public launch)

- **No auth / rate limiting** — fine for a private demo; add a gateway or key if it's public.
- **Real-model / real-browser validation** — verified via tests + an HTTP smoke, not yet a full browser run on real music.
- Uploads are capped at `MAX_UPLOAD_MB` (default 15 MB) and read into memory; raise/lower via the env var.
