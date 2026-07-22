import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from serving.preprocess import waveform_to_segments
from serving.constants import GENRES, SR, SAMPLES_PER_SEGMENT

def load_norm_stats(path) -> tuple[float, float]:
    d = json.loads(Path(path).read_text())
    return float(d["mean"]), float(d["std"])

def normalize(X: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (X - mean) / std

def representative_spectrogram(segments: np.ndarray) -> list[list[int]]:
    idx = int(segments.mean(axis=(1, 2, 3)).argmax())   # loudest segment
    grid = segments[idx, :, :, 0]                        # (128,130) dB
    lo, hi = float(grid.min()), float(grid.max())
    scaled = (grid - lo) / (hi - lo + 1e-9)              # 0..1
    return (scaled * 255).astype(np.uint8).astype(int).tolist()

class Classifier:
    def __init__(self, model_path, mean: float, std: float):
        self.model = tf.keras.models.load_model(model_path)
        self.mean = mean
        self.std = std
        self.ready = True

    def segments(self, y: np.ndarray) -> np.ndarray:
        return waveform_to_segments(y)                 # (n_seg,128,130,1) unnormalized

    def _mean_probs(self, X: np.ndarray) -> np.ndarray:
        # Call the model eagerly — model(X) — rather than model.predict(X). For the
        # small per-clip batch this is faster and, crucially, avoids predict()'s
        # one-time graph/XLA compile (a ~90s cold-start spike on the first request).
        # The output (softmax) is identical.
        return self.model(X, training=False).numpy().mean(axis=0)   # avg over segments

    def predict(self, y: np.ndarray) -> np.ndarray:
        X = normalize(self.segments(y), self.mean, self.std)
        return self._mean_probs(X).astype(np.float64)

    def classify(self, y: np.ndarray) -> dict:
        segs = self.segments(y)
        X = normalize(segs, self.mean, self.std)
        probs = self._mean_probs(X).astype(float)
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
