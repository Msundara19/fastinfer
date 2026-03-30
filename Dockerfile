FROM python:3.11-slim

WORKDIR /app

# System dependencies required by PyTorch and Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer cached unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

RUN chmod +x entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]
