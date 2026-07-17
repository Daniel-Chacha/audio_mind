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
