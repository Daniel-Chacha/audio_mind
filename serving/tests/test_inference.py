import json
import numpy as np
import pytest
from serving.inference import load_norm_stats, normalize, Classifier
from serving.constants import GENRES, MODEL_PATH, SR

def _tone(seconds, freq=220.0):
    t = np.linspace(0, seconds, int(SR * seconds), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)

def test_load_norm_stats(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"mean": 3.0, "std": 2.0}))
    assert load_norm_stats(p) == (3.0, 2.0)

def test_normalize_applies_formula():
    X = np.array([[5.0]], dtype=np.float32)
    out = normalize(X, mean=1.0, std=2.0)
    assert out[0, 0] == pytest.approx(2.0)

def test_predict_returns_probability_vector():
    clf = Classifier(MODEL_PATH, mean=0.0, std=1.0)
    probs = clf.predict(_tone(6.0))
    assert probs.shape == (len(GENRES),)
    assert probs.sum() == pytest.approx(1.0, abs=1e-4)
    assert (probs >= 0).all()
