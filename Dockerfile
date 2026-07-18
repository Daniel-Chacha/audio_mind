# AudioMind serving API (FastAPI + TensorFlow) — container image for a
# long-running host (Render / Fly / Railway / a VM). The web frontend is
# deployed separately (Vercel); it is not part of this image.
#
# Build from the repo root (the image needs best_model.keras at the root):
#   docker build -t audiomind-api .
#   docker run -p 8000:8000 -e ALLOWED_ORIGINS=https://your-web.vercel.app audiomind-api
FROM python:3.12-slim

# System deps: ffmpeg (decode mp3/other formats via audioread), libsndfile1
# (soundfile runtime), libgomp1 (OpenMP runtime TensorFlow links against).
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install Python deps first (cached layer). Swap tensorflow -> tensorflow-cpu:
# identical inference, but a much smaller image (no bundled CUDA) for CPU hosts.
COPY serving/requirements.txt ./serving/requirements.txt
RUN sed 's/^tensorflow==/tensorflow-cpu==/' serving/requirements.txt > /tmp/req.txt \
    && pip install --timeout 120 --retries 5 --prefer-binary -r /tmp/req.txt

# App code + model + normalization stats. Keep the repo-relative layout so
# serving/constants.py resolves MODEL_PATH to /app/best_model.keras.
COPY serving ./serving
COPY best_model.keras ./best_model.keras

# Run as a non-root user.
RUN useradd --create-home --uid 10001 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

# Container hosts inject $PORT; default to 8000 locally. No --reload in prod.
CMD ["sh", "-c", "uvicorn serving.app:app --host 0.0.0.0 --port ${PORT:-8000}"]

# Liveness: the process is healthy once the model is loaded and /health answers.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen(f\"http://127.0.0.1:{os.environ.get('PORT','8000')}/health\").read()" || exit 1
