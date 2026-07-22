# AudioMind API as a Hugging Face Docker Space (free CPU tier: 2 vCPU, 16 GB RAM).
#
# The Space's build context is the Space repo root, which must contain
# `serving/` and `best_model.keras` (the deploy-space.sh script copies them in).
# HF routes traffic to port 7860 by default, so we listen there — no $PORT needed.
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# HF Docker Spaces run the container as root, but librosa/numba/matplotlib and
# TF try to write caches under $HOME; point them all at writable /tmp locations.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/tmp/hf \
    XDG_CACHE_HOME=/tmp/cache \
    NUMBA_CACHE_DIR=/tmp/numba \
    MPLCONFIGDIR=/tmp/mpl \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

# Install deps first (cached layer). tensorflow -> tensorflow-cpu keeps the image lean.
COPY serving/requirements.txt ./serving/requirements.txt
RUN sed 's/^tensorflow==/tensorflow-cpu==/' serving/requirements.txt > /tmp/req.txt \
    && pip install --timeout 120 --retries 5 --prefer-binary -r /tmp/req.txt

COPY serving ./serving
COPY best_model.keras ./best_model.keras

RUN mkdir -p /tmp/hf /tmp/cache /tmp/numba /tmp/mpl && chmod -R 777 /tmp/hf /tmp/cache /tmp/numba /tmp/mpl

EXPOSE 7860
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "7860"]
