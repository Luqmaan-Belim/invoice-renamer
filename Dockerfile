# Solid base with Python preinstalled; good wheels available
FROM python:3.11-slim-bookworm

# System libs needed by tesseract + opencv headless runtime
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      tesseract-ocr \
      libgl1 \
      libglib2.0-0 \
      libsm6 \
      libxrender1 \
      libxext6 && \
    rm -rf /var/lib/apt/lists/*

# Speed up pip and prefer prebuilt wheels
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install Python deps first (for better Docker layer caching)
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --prefer-binary -r /app/requirements.txt

# Add your script
COPY process_invoices.py /app/process_invoices.py

CMD ["python", "/app/process_invoices.py"]
