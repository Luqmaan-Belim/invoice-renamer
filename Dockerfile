FROM debian:bookworm-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3 python3-pip tesseract-ocr libgl1 && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt

WORKDIR /app
COPY process_invoices.py /app/process_invoices.py

CMD ["python3", "/app/process_invoices.py"]
