import numpy as np
import pytest
from serving.preprocess import waveform_to_segments, ClipTooShortError
from serving.constants import SR, SAMPLES_PER_SEGMENT

def _tone(seconds, freq=220.0):
    t = np.linspace(0, seconds, int(SR * seconds), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)

def test_shape_for_two_segments():
    X = waveform_to_segments(_tone(6.0))          # exactly 2 segments
    assert X.shape == (2, 128, 130, 1)
    assert X.dtype == np.float32

def test_partial_tail_is_dropped():
    X = waveform_to_segments(_tone(7.5))          # 2 full segments + 1.5s tail
    assert X.shape[0] == 2

def test_too_short_raises():
    with pytest.raises(ClipTooShortError):
        waveform_to_segments(_tone(1.0))

def test_values_are_finite_db():
    X = waveform_to_segments(_tone(3.0))
    assert np.isfinite(X).all()
    assert X.max() <= 0.0 + 1e-3                   # power_to_db(ref=max) → dB <= 0
