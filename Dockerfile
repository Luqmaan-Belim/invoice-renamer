# Dockerfile
FROM python:3.11-slim

# System deps for Tesseract + pdf2image
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      tesseract-ocr poppler-utils \
 && rm -rf /var/lib/apt/lists/*

# Python deps (install once at build time)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Default workdir (Actions will mount your repo here anyway)
WORKDIR /work
