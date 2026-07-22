# Deploy the API free on Hugging Face Spaces

Render's free tier is 512 MB RAM — too small for TensorFlow. A free **Hugging
Face Docker Space** gives **16 GB RAM**, which fits comfortably. The web app
stays on Vercel; only the API moves here.

> Trade-off: free Spaces sleep after ~48 h idle and cold-start on the next
> visit. The Space takes ~30 s–2 min to become *ready* — it loads and warms up
> the model at startup — after which every classification is sub-second (the
> app pays the model's one-time compile at startup, not on your first request).
> Fine for a demo.

## 1. Create the Space

1. Go to https://huggingface.co/new-space (sign in / make a free account).
2. **Owner/name:** e.g. `audiomind-api`. **SDK:** **Docker** (blank template).
   **Visibility:** **Public** (so the browser can call it without a token).
3. Create it. It starts empty — you'll push to it next.

## 2. Authenticate git for Hugging Face (once)

```bash
pip install -U "huggingface_hub[cli]"
hf auth login          # paste a token that has WRITE access
```
This installs a git credential helper so `git push` to the Space works.

## 3. Push the API

From the repo root:

```bash
deploy/hf-space/deploy-space.sh <your-hf-username>/audiomind-api
```

The script clones the Space, copies in the Dockerfile, `serving/`, and
`best_model.keras`, commits, and pushes. The Space then **builds the image
automatically** (watch the "Building" logs on the Space page — a few minutes).

When it flips to **Running**, your API URL is:

```
https://<hf-username>-audiomind-api.hf.space
```

(lowercase, with `/` turned into `-`). Test it:

```bash
curl https://<hf-username>-audiomind-api.hf.space/health
# → {"status":"ok","model_loaded":true,"genres":[...]}
```

## 4. Wire CORS + the frontend

1. **Space → Settings → Variables and secrets → New variable:**
   `ALLOWED_ORIGINS` = your Vercel URL, e.g. `https://audio-mind-zeta.vercel.app`
   (no trailing slash). The Space restarts and now allows that origin.
2. **Vercel → your project → Settings → Environment Variables:** set
   `NEXT_PUBLIC_API_BASE` = `https://<hf-username>-audiomind-api.hf.space`, then
   **redeploy** the web app (that value is baked in at build time).

## 5. Verify end-to-end

Open your Vercel URL, click a sample chip or upload a clip → you should get a
prediction. If the very first request after idle fails, the Space was asleep —
retry once it's awake.

## Updating later

Re-run `deploy/hf-space/deploy-space.sh <user>/audiomind-api` after any change
to `serving/` or the model; it pushes a new commit and the Space rebuilds.
