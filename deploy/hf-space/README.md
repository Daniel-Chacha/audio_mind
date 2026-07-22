---
title: AudioMind API
emoji: 🎛️
colorFrom: purple
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
---

# AudioMind API

FastAPI service that classifies a music clip into one of 10 GTZAN genres
(blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock)
with a small Keras/TensorFlow CNN.

- `GET /health` — liveness + the genre list.
- `POST /classify` — multipart `file` (wav/mp3/flac) → predicted genre,
  per-genre confidence, and the mel spectrogram the model saw.

This Space is the backend for the AudioMind web app. Set the `ALLOWED_ORIGINS`
Space variable to your frontend's origin so the browser can call it.
