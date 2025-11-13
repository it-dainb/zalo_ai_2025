# ============================================================================
# Dockerfile for YOLOv8n-RefDet Cloud GPU Training
# CUDA 12.6 | PyTorch 2.9.0 | Python 3.10
# Optimized for: Vast.ai, RunPod, Lambda Labs, etc.
# ============================================================================

# ============================================================================
# Stage 1: Base Image with CUDA Runtime
# ============================================================================
FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu22.04 AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Python and build essentials
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3-setuptools \
    build-essential \
    # Computer Vision libraries
    libopencv-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # System utilities
    wget \
    curl \
    git \
    vim \
    htop \
    # For better performance
    libopenblas-dev \
    libomp-dev \
    # Cleanup
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip, setuptools, and wheel
RUN python3 -m pip install --upgrade pip setuptools wheel

# Create symbolic link for python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# ============================================================================
# Stage 2: Dependencies Builder (with build cache)
# ============================================================================
FROM base AS builder

# Set working directory for build
WORKDIR /build

# Copy only requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN python -m pip install --user --no-warn-script-location \
    # Install PyTorch with CUDA 12.6 support first (large package)
    torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu126 \
    && python -m pip install --user --no-warn-script-location -r requirements.txt

# ============================================================================
# Stage 3: Production Image (for cloud GPU training)
# ============================================================================
FROM base

# Create non-root user for security
RUN useradd -m -u 1000 -s /bin/bash trainer && \
    mkdir -p /workspace /workspace/datasets /workspace/checkpoints /workspace/logs && \
    chown -R trainer:trainer /workspace

# Copy installed packages from builder
COPY --from=builder --chown=trainer:trainer /root/.local /home/trainer/.local

# Set environment variables
ENV PATH=/home/trainer/.local/bin:$PATH \
    PYTHONPATH=/workspace:$PYTHONPATH \
    WANDB_DIR=/workspace/wandb \
    TORCH_HOME=/workspace/.cache/torch \
    HF_HOME=/workspace/.cache/huggingface

# Set working directory
WORKDIR /workspace

# Switch to non-root user
USER trainer

# Copy application code (exclude datasets/checkpoints via .dockerignore)
COPY --chown=trainer:trainer . /workspace/

# Create necessary directories
RUN mkdir -p /workspace/wandb /workspace/outputs \
    /workspace/.cache/torch /workspace/.cache/huggingface

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available()" || exit 1

# Default command (can be overridden)
CMD ["python", "train.py", "--help"]

# ============================================================================
# BUILD & DEPLOYMENT INSTRUCTIONS
# ============================================================================
#
# 1. BUILD IMAGE:
#    docker build -t <your-dockerhub-username>/yolov8n-refdet:latest .
#
# 2. PUSH TO DOCKER HUB:
#    docker login
#    docker push <your-dockerhub-username>/yolov8n-refdet:latest
#
# 3. RUN ON CLOUD GPU (Vast.ai, RunPod, Lambda Labs, etc.):
#    docker run --gpus all --shm-size=16g \
#      -e WANDB_API_KEY=<your-wandb-key> \
#      -v /path/to/datasets:/workspace/datasets:ro \
#      -v /path/to/checkpoints:/workspace/checkpoints \
#      <your-dockerhub-username>/yolov8n-refdet:latest \
#      python train.py --stage 2 --epochs 100 --n_way 2 --n_query 4 --use_wandb
#
# 4. INTERACTIVE MODE (for debugging):
#    docker run --gpus all --shm-size=16g -it \
#      -v /path/to/datasets:/workspace/datasets \
#      <your-dockerhub-username>/yolov8n-refdet:latest /bin/bash
#
# 5. WITH CUSTOM TRAINING SCRIPT:
#    docker run --gpus all --shm-size=16g \
#      -e WANDB_API_KEY=<your-wandb-key> \
#      -v /path/to/datasets:/workspace/datasets:ro \
#      -v /path/to/checkpoints:/workspace/checkpoints \
#      -v /path/to/train_custom.sh:/workspace/train_custom.sh \
#      <your-dockerhub-username>/yolov8n-refdet:latest \
#      bash train_custom.sh
#
# ============================================================================
# CLOUD GPU PLATFORM SPECIFICS
# ============================================================================
#
# VAST.AI:
#   - Use "Docker Image Name" field: <your-dockerhub-username>/yolov8n-refdet:latest
#   - Set "On-start script": python train.py --stage 2 --epochs 100 --use_wandb
#   - Add environment variable: WANDB_API_KEY=<key>
#   - Mount disk storage to /workspace/datasets and /workspace/checkpoints
#
# RUNPOD:
#   - Template > Docker Image: <your-dockerhub-username>/yolov8n-refdet:latest
#   - Container Disk: 50GB+ recommended
#   - Environment Variables: WANDB_API_KEY=<key>
#   - Volume mounts for datasets/checkpoints
#   - Start command: python train.py --stage 2 --epochs 100 --use_wandb
#
# LAMBDA LABS:
#   - Similar to standard docker run command
#   - Ensure --shm-size=16g for DataLoader with multiple workers
#
# ============================================================================
