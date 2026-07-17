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

def test_classify_no_file_is_400():
    # Absent `file` field entirely.
    r = client.post("/classify")
    assert r.status_code == 400
    assert r.json()["error"] == "no_file"
    # Explicitly-empty file body.
    r = client.post("/classify", files={"file": ("empty.wav", b"", "audio/wav")})
    assert r.status_code == 400
    assert r.json()["error"] == "no_file"
