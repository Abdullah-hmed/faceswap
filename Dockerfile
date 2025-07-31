# Use NVIDIA base image with CUDA 11.8 and cuDNN 8
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python and system dependencies
RUN apt-get update && \
    apt-get install -y python3.11 python3.11-venv python3.11-dev python3-pip ffmpeg git libgl1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /app

# Copy only requirements first to cache pip install
COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy model files early for cache benefits
COPY models/ /app/models/

# Copy source code and everything else
COPY . .

# Expose for Flask
EXPOSE 3000

ENTRYPOINT ["python"]
CMD ["main.py"]