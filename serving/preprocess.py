import numpy as np
import librosa
from serving.constants import (
    SR, N_FFT, HOP_LENGTH, N_MELS, SAMPLES_PER_SEGMENT,
)

class ClipTooShortError(ValueError):
    """Raised when a clip has no full 3-second segment."""

def waveform_to_segments(y: np.ndarray) -> np.ndarray:
    n_seg = len(y) // SAMPLES_PER_SEGMENT
    if n_seg == 0:
        raise ClipTooShortError("clip shorter than one 3s segment")
    segs = []
    for i in range(n_seg):
        seg = y[i * SAMPLES_PER_SEGMENT:(i + 1) * SAMPLES_PER_SEGMENT]
        mel = librosa.feature.melspectrogram(
            y=seg, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)   # (128, 130)
        segs.append(mel_db.astype(np.float32))
    X = np.stack(segs)[..., np.newaxis]                 # (n_seg, 128, 130, 1)
    return X
