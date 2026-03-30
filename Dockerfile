FROM python:3.11-slim

WORKDIR /app

# System dependencies required by PyTorch and Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

# Install torch CPU-only first — own layer so it stays cached across rebuilds
# CPU-only is ~200MB vs ~800MB for the default CUDA build
RUN pip install --no-cache-dir \
    torch==2.6.0 \
    torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies (cached separately from torch)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

RUN chmod +x entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]
