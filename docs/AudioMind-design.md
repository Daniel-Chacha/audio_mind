# AudioMind — Music Genre Classification from Spectrograms

**Design doc / roadmap** · 2026-07-11
A beginner-friendly deep learning project: predict a music clip's genre from its
(mel) spectrogram using a CNN in Keras/TensorFlow.

---

## 1. Goal & scope

Build an end-to-end pipeline that takes a raw audio clip, converts it into a
mel spectrogram, and classifies it into one of 10 genres with a small CNN.

- **Learning objective:** understand audio preprocessing & feature extraction
  (the genuinely new part), then reuse familiar image-classification skills for
  the model.
- **Framework:** Keras / TensorFlow (model + training), librosa (audio features).
- **Environment:** Google Colab (free GPU).
- **Success:** a trained model reaching ~65–80% test accuracy, plus a confusion
  matrix showing which genres get confused.

## Environment: VS Code connected to a Colab kernel

The notebook runs in **VS Code using a Colab runtime as the remote kernel**. The
kernel (GPU, filesystem, ffmpeg) lives on Colab, but the frontend is VS Code —
which changes two things.

**Visualizations** — add this as the notebook's first cell so plots render sharply:

```python
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12, 4)
```

- Keep every plot ending in `plt.show()`.
- Wrap audio in `display(Audio(...))`, not a bare last-line `Audio(...)`.
- Do NOT use `%matplotlib widget` / `notebook` — they render blank over the bridge.

**`google.colab` widgets don't work** — `files.upload()` and `drive.mount()` need
the Colab web frontend, which isn't present in VS Code. Replace them:

- **Kaggle auth:** set `os.environ['KAGGLE_USERNAME']` / `['KAGGLE_KEY']` instead of
  uploading `kaggle.json`.
- **Own-clip inference:** pass a file path or `wget` a URL instead of `files.upload()`.
- **Persistence:** `np.savez_compressed(...)` / `model.save(...)` to Colab disk, then
  retrieve via VS Code Remote-SSH download or an upload link — not by mounting Drive.

## 2. The core idea

**A spectrogram is a 2D image, so genre classification is image classification.**
The CNN, training loop, loss/optimizer, and overfitting checks are all familiar.
The new work is the front of the pipeline: turning a `.wav` into that image.

```
raw audio (.wav)  →  [preprocessing]  →  mel spectrogram (2D image)  →  CNN  →  genre
   1D signal            the new part          time × frequency          familiar
```

## 3. Audio concepts to understand

- **Waveform** — raw audio is a 1D array of amplitude samples over time.
- **Sample rate (`sr`)** — samples per second. GTZAN = `22050 Hz`
  (a 30 s clip ≈ 661,500 samples).
- **STFT** — slide a small window along the signal; for each window run an FFT to
  measure how much of each frequency is present. Stack results → a 2D map of
  frequency (y) over time (x) = the spectrogram. Knobs: `n_fft`, `hop_length`.
- **Mel spectrogram** — the spectrogram with its frequency axis warped to the
  perceptual **mel scale**, then converted to **decibels (log scale)**. This
  log-mel spectrogram is the standard input for music DL. Knob: `n_mels`.
- **MFCC** — a compressed summary (~13–40 coefficients) of the mel spectrogram.
  Classic for speech/traditional ML; good to understand, but the full mel
  spectrogram usually works better for a CNN. Not our main input.

## 4. Dataset — GTZAN

- 1,000 clips · 30 s each · `22050 Hz` mono
- 10 genres × 100 clips: blues, classical, country, disco, hiphop, jazz, metal,
  pop, reggae, rock (balanced)
- Chance accuracy = 10%; a solid CNN typically reaches ~65–80%.
- **Caveats:** known label noise / duplicates (fine for learning); the file
  `jazz.00054.wav` is corrupted — skip it.

## 5. Feature configuration

These numbers determine the CNN's input size.

| Setting          | Value      | Why                                    |
| ---------------- | ---------- | -------------------------------------- |
| Sample rate      | `22050 Hz` | GTZAN's native rate                    |
| Segment length   | `3 s`      | Turns 1,000 clips into ~10,000 samples |
| `n_fft`          | `2048`     | STFT window size (standard)            |
| `hop_length`     | `512`      | Step between windows                   |
| `n_mels`         | `128`      | Mel bands (the image height)           |
| **Input shape**  | **`(128, ~130, 1)`** | 128 mel bands × ~130 frames × 1 channel |

## 6. Data flow

```
GTZAN → split SONGS into train/val/test (70/15/15)   ← split happens FIRST
      → cut each song into 3 s segments
      → librosa: load → mel spectrogram → power_to_db (log scale)
      → normalize (global mean/std computed from TRAIN only)
      → X: (N, 128, 130, 1),  y: integer 0–9
```

## 7. Model (Keras)

A small CNN, ~3 conv blocks:

```
Input(128,130,1)
  → [Conv2D → BatchNorm → MaxPooling2D] × 3
  → GlobalAveragePooling2D
  → Dense(64, relu) → Dropout(0.3)
  → Dense(10, softmax)
```

**Compile:** `optimizer='adam'`, `loss='sparse_categorical_crossentropy'`,
`metrics=['accuracy']`.
**Train:** `model.fit(...)` with `EarlyStopping(patience=10)` on val loss and
`ModelCheckpoint` to save the best model.

## 8. Evaluation

- Test accuracy.
- **Confusion matrix** — the interesting part: which genres get mixed up
  (rock↔metal, disco↔pop are common).

## 9. Two beginner gotchas (important)

1. **Split by SONG, not by SEGMENT.** Assign whole songs to train/val/test
   *before* cutting them into 3 s segments. Otherwise segments from the same song
   leak across splits, the model memorizes songs, and accuracy looks great but is
   a lie.
2. **Normalize spectrograms** (subtract mean, divide by std) before the CNN,
   using statistics computed from the training set only.

## 10. Notebook structure (one Colab notebook)

```
1. Setup & download GTZAN
2. Explore: listen + plot spectrograms
3. Preprocessing / feature extraction   ← the learning core
4. Build train/val/test arrays (song-level split!)
5. Define CNN
6. Train
7. Evaluate + confusion matrix
8. (Stretch) predict on your own clip
```

## 11. Phase 2 (stretch, later)

- **MFCC + a smaller network** — a lighter, more "classical" feature experiment
  to compare against mel spectrograms.
- **Transfer learning** — fine-tune a pretrained image CNN (e.g. ResNet /
  EfficientNet) on spectrogram images for higher accuracy with less data. More of
  a black box; best attempted after the basic version works.

---

## Build approach

The author is building this hands-on by copying code stage by stage (learning by
doing), not having it auto-implemented. Work through the notebook sections in
order; each stage's code is provided and explained in chat before moving on.
