# Stage 1
FROM superlinear/python-gpu:3.11-cuda11.8 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Install all system dependencies, including build-essential and git
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    libgl1 \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory and copy requirements
WORKDIR /app
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2
FROM superlinear/python-gpu:3.11-cuda11.8 AS final

# Installing only runtime dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY models/ /app/models/
COPY . .

EXPOSE 3000

ENTRYPOINT ["python"]
CMD ["main.py"]