# VR180 AI Pipeline — Docker image
# Build:  docker build -t vr180-pipeline .
# Run:    docker run --rm -v $(pwd)/video:/app/video vr180-pipeline --input /app/video/input.mp4 --output /app/video/output_vr180.mp4

FROM python:3.11-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps (copy requirements first for Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY pipeline/ pipeline/
COPY scripts/ scripts/
COPY pyproject.toml .

# Default entrypoint
ENTRYPOINT ["python", "scripts/run_pipeline.py"]