from pathlib import Path
import tensorflow as tf
from serving.constants import MODEL_PATH, GENRES

def test_model_file_exists():
    assert Path(MODEL_PATH).exists(), f"missing model at {MODEL_PATH}"

def test_model_loads_with_expected_io():
    model = tf.keras.models.load_model(MODEL_PATH)
    assert model.input_shape == (None, 128, 130, 1)
    assert model.output_shape == (None, len(GENRES))  # 10
