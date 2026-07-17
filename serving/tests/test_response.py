import numpy as np
import pytest
from serving.inference import representative_spectrogram, Classifier
from serving.constants import GENRES, MODEL_PATH, SR

def _tone(seconds, freq=220.0):
    t = np.linspace(0, seconds, int(SR * seconds), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)

def test_representative_spectrogram_shape_and_range():
    segs = np.random.rand(3, 128, 130, 1).astype(np.float32)
    grid = representative_spectrogram(segs)
    assert len(grid) == 128 and len(grid[0]) == 130
    flat = [v for row in grid for v in row]
    assert min(flat) >= 0 and max(flat) <= 255
    assert all(isinstance(v, int) for v in flat[:5])

def test_classify_result_schema():
    clf = Classifier(MODEL_PATH, mean=0.0, std=1.0)
    r = clf.classify(_tone(6.0))
    assert r["genre"] in GENRES
    assert 0.0 <= r["confidence"] <= 1.0
    assert len(r["probabilities"]) == len(GENRES)
    assert {p["genre"] for p in r["probabilities"]} == set(GENRES)
    assert r["spectrogram"]["bands"] == 128 and r["spectrogram"]["frames"] == 130
    assert r["meta"]["segments"] == 2
