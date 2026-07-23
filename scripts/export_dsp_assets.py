"""Export the DSP assets the browser needs to reproduce librosa's mel pipeline.

The browser must build the exact same log-mel spectrogram the model trained on.
Rather than re-deriving librosa's Slaney mel filterbank and window in JS (easy to
get subtly wrong), we export them here and ship them as static assets; the JS
side just does STFT -> |.|^2 -> matmul(filterbank) -> power_to_db -> normalize.

Also writes a parity fixture (raw samples + the reference mel) so the JS
pipeline can be checked numerically against librosa.

Run with an env that has librosa/numpy:
    python scripts/export_dsp_assets.py
"""

import json
from pathlib import Path

import numpy as np
import librosa

# Constants from the training notebook (index.ipynb cell 3) — must not drift.
SR = 22050
DURATION = 3
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
SAMPLES_PER_SEGMENT = SR * DURATION          # 66150
FRAMES = 1 + SAMPLES_PER_SEGMENT // HOP_LENGTH  # 130, with center=True
GENRES = ["blues", "classical", "country", "disco", "hiphop",
          "jazz", "metal", "pop", "reggae", "rock"]

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "web" / "public" / "model"
FIXTURE = ROOT / "scripts" / "parity_fixture"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIXTURE.mkdir(parents=True, exist_ok=True)

    stats = json.loads((ROOT / "scripts" / "norm_stats.json").read_text())
    mean, std = float(stats["mean"]), float(stats["std"])

    # librosa's Slaney mel filterbank: (n_mels, 1 + n_fft/2) = (128, 1025)
    mel_fb = librosa.filters.mel(sr=SR, n_fft=N_FFT, n_mels=N_MELS).astype(np.float32)
    (OUT / "mel_filters.bin").write_bytes(mel_fb.tobytes())

    # The exact analysis window librosa uses (periodic Hann, length n_fft).
    window = librosa.filters.get_window("hann", N_FFT, fftbins=True).astype(np.float32)
    (OUT / "hann_window.bin").write_bytes(window.tobytes())

    # librosa's stft default pad mode matters for center=True; record what this
    # version actually uses so the JS side can mirror it.
    import inspect
    pad_mode = inspect.signature(librosa.stft).parameters["pad_mode"].default

    meta = {
        "sr": SR,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "n_mels": N_MELS,
        "frames": FRAMES,
        "samples_per_segment": SAMPLES_PER_SEGMENT,
        "mel_filters_shape": list(mel_fb.shape),
        "window_length": int(window.shape[0]),
        "center": True,
        "pad_mode": str(pad_mode),
        "power": 2.0,
        "top_db": 80.0,
        "amin": 1e-10,
        "norm_mean": mean,
        "norm_std": std,
        "genres": GENRES,
        "librosa_version": librosa.__version__,
    }
    (OUT / "dsp.json").write_text(json.dumps(meta, indent=2))

    # Parity fixture: one real segment + librosa's reference log-mel for it.
    clip = ROOT / "web" / "public" / "samples" / "metal.wav"
    y, _ = librosa.load(clip, sr=SR, mono=True)
    seg = y[:SAMPLES_PER_SEGMENT].astype(np.float32)
    mel = librosa.feature.melspectrogram(
        y=seg, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)

    # Reference probabilities straight from the Keras model for this exact
    # segment, so the browser pipeline can be verified end-to-end (spectrogram
    # AND model), not just the spectrogram.
    import keras
    model = keras.saving.load_model(ROOT / "best_model.keras", compile=False)
    x = ((mel_db - mean) / std)[np.newaxis, ..., np.newaxis]
    probs = model(x, training=False).numpy()[0].astype(float)

    (FIXTURE / "segment.bin").write_bytes(seg.tobytes())
    (FIXTURE / "mel_db.bin").write_bytes(mel_db.tobytes())
    (FIXTURE / "meta.json").write_text(json.dumps({
        "clip": clip.name,
        "segment_samples": int(seg.shape[0]),
        "mel_db_shape": list(mel_db.shape),
        "mel_db_min": float(mel_db.min()),
        "mel_db_max": float(mel_db.max()),
        "keras_probs": probs.tolist(),
        "keras_top": GENRES[int(probs.argmax())],
    }, indent=2))
    print(f"keras reference: {GENRES[int(probs.argmax())]} @ {probs.max()*100:.2f}%")

    print(f"mel filterbank {mel_fb.shape} -> {OUT/'mel_filters.bin'}")
    print(f"hann window    {window.shape} -> {OUT/'hann_window.bin'}")
    print(f"meta (librosa {librosa.__version__}, pad_mode={pad_mode}) -> {OUT/'dsp.json'}")
    print(f"parity fixture {mel_db.shape} (min {mel_db.min():.2f}, max {mel_db.max():.2f}) -> {FIXTURE}")


if __name__ == "__main__":
    main()
