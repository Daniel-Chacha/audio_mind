from pathlib import Path

SR = 22050
DURATION = 3
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
SAMPLES_PER_SEGMENT = SR * DURATION  # 66150

GENRES = ["blues", "classical", "country", "disco", "hiphop",
          "jazz", "metal", "pop", "reggae", "rock"]

_ROOT = Path(__file__).resolve().parent
MODEL_PATH = _ROOT.parent / "best_model.keras"
STATS_PATH = _ROOT / "norm_stats.json"
