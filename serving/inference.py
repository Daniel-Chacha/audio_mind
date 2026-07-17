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
