import io
import os
import librosa
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from serving.constants import MODEL_PATH, STATS_PATH, SR, GENRES
from serving.inference import Classifier, load_norm_stats
from serving.preprocess import ClipTooShortError

# Allowed browser origins for CORS. Comma-separate for multiple; set this to
# the deployed web origin in production (e.g. ALLOWED_ORIGINS=https://audiomind.vercel.app).
# Defaults to the local Next.js dev server.
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    if o.strip()
]

# Reject uploads larger than this before decoding (a huge file could exhaust memory).
# A 30 s WAV at 22.05 kHz is ~2.6 MB; the default 15 MB leaves generous headroom.
MAX_UPLOAD_MB = float(os.environ.get("MAX_UPLOAD_MB", "15"))
MAX_UPLOAD_BYTES = int(MAX_UPLOAD_MB * 1024 * 1024)

app = FastAPI(title="AudioMind serving")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
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
async def classify(file: UploadFile | None = File(None)):
    if file is None:
        return _err(400, "no_file", "Attach an audio file.")
    data = await file.read()
    if not data:
        return _err(400, "no_file", "Attach an audio file.")
    if len(data) > MAX_UPLOAD_BYTES:
        return _err(
            413,
            "file_too_large",
            f"That file is too large — keep it under {MAX_UPLOAD_MB:g} MB.",
        )
    try:
        y, _ = librosa.load(io.BytesIO(data), sr=SR, mono=True)
    except Exception:
        return _err(422, "unreadable_audio", "Couldn't read that file — try a WAV or MP3.")
    try:
        return clf.classify(y)
    except ClipTooShortError:
        return _err(422, "clip_too_short", "Clips need to be at least 3 seconds.")
