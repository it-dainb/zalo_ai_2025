# Docker Deployment Guide for Cloud GPU Training

This guide covers building, pushing, and deploying the YOLOv8n-RefDet Docker image to cloud GPU platforms.

## Table of Contents
- [Quick Start](#quick-start)
- [Building and Pushing Image](#building-and-pushing-image)
- [Cloud GPU Platforms](#cloud-gpu-platforms)
- [Training Configuration](#training-configuration)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

### 1. Setup Environment Variables

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your values
nano .env

# Load environment variables
export $(cat .env | xargs)
```

### 2. Build and Push Image

```bash
# Using the helper script (recommended)
./build_and_push.sh

# Or manually
docker build -t ${DOCKER_USERNAME}/yolov8n-refdet:latest .
docker push ${DOCKER_USERNAME}/yolov8n-refdet:latest
```

### 3. Run on Cloud GPU

```bash
docker run --gpus all --shm-size=16g \
  -e WANDB_API_KEY=${WANDB_API_KEY} \
  -v /path/to/datasets:/workspace/datasets:ro \
  -v /path/to/checkpoints:/workspace/checkpoints \
  ${DOCKER_USERNAME}/yolov8n-refdet:latest \
  python train.py --stage 2 --epochs 100 --n_way 2 --n_query 4 --use_wandb
```

---

## Building and Pushing Image

### Prerequisites

- Docker installed and running
- Docker Hub account ([sign up](https://hub.docker.com/signup))
- 10GB+ free disk space
- Good internet connection (image ~2-3GB)

### Step 1: Login to Docker Hub

```bash
docker login
# Enter your Docker Hub username and password
```

### Step 2: Build the Image

```bash
# Set your Docker Hub username
export DOCKER_USERNAME=yourusername

# Build the image (takes 10-15 minutes)
docker build -t ${DOCKER_USERNAME}/yolov8n-refdet:latest .

# Check image size
docker images ${DOCKER_USERNAME}/yolov8n-refdet:latest
```

**Expected image size**: ~2-3GB

### Step 3: Push to Docker Hub

```bash
# Push the image (takes 10-20 minutes depending on upload speed)
docker push ${DOCKER_USERNAME}/yolov8n-refdet:latest
```

### Step 4: Verify Upload

Visit `https://hub.docker.com/r/${DOCKER_USERNAME}/yolov8n-refdet` to confirm your image is public.

---

## Cloud GPU Platforms

### Vast.ai

**Best for**: Budget-conscious training, spot instances

#### Setup Steps:

1. **Create Account**: [vast.ai](https://vast.ai)

2. **Configure Instance**:
   - Click "Create" > "New Instance"
   - **Docker Image**: `yourusername/yolov8n-refdet:latest`
   - **GPU**: RTX 3090, RTX 4090, or A6000 (24GB+ VRAM recommended)
   - **Disk Space**: 50GB minimum (100GB+ recommended)
   - **Template Type**: Docker

3. **Environment Variables**:
   ```bash
   WANDB_API_KEY=your_key_here
   WANDB_PROJECT=yolov8n-refdet
   ```

4. **Storage Setup**:
   - Upload your datasets to Vast.ai storage
   - Mount to `/workspace/datasets`
   - Create output directory and mount to `/workspace/checkpoints`

5. **On-Start Script**:
   ```bash
   python train.py --stage 2 --epochs 100 --n_way 2 --n_query 4 \
     --use_wandb --data_root /workspace/datasets/train/samples \
     --annotations /workspace/datasets/train/annotations/annotations.json \
     --checkpoint_dir /workspace/checkpoints
   ```

6. **SSH Access** (optional):
   ```bash
   ssh -p <port> root@<ip-address>
   docker exec -it $(docker ps -q) /bin/bash
   ```

---

### RunPod

**Best for**: Easy setup, reliable infrastructure

#### Setup Steps:

1. **Create Account**: [runpod.io](https://runpod.io)

2. **Create Pod**:
   - Go to "Pods" > "Deploy"
   - **GPU**: RTX 4090, A6000, or A100 (24GB+ VRAM)
   - **Template**: Custom Docker Image
   - **Docker Image**: `yourusername/yolov8n-refdet:latest`
   - **Container Disk**: 50GB+
   - **Volume**: Create or attach volume for datasets (100GB+)

3. **Environment Variables**:
   ```
   WANDB_API_KEY=your_key_here
   WANDB_PROJECT=yolov8n-refdet
   ```

4. **Volume Mounts**:
   - Dataset volume → `/workspace/datasets`
   - Create output directory in volume → `/workspace/checkpoints`

5. **Start Command**:
   ```bash
   python train.py --stage 2 --epochs 100 --n_way 2 --n_query 4 \
     --use_wandb --data_root /workspace/datasets/train/samples \
     --annotations /workspace/datasets/train/annotations/annotations.json \
     --checkpoint_dir /workspace/checkpoints
   ```

6. **Connect via Web Terminal**:
   - Click "Connect" > "Start Web Terminal"
   - Or SSH: `ssh -i ~/.ssh/id_ed25519 <username>@<pod-ip> -p <port>`

---

### Lambda Labs

**Best for**: High-performance training, A100 GPUs

#### Setup Steps:

1. **Create Account**: [lambdalabs.com](https://lambdalabs.com/service/gpu-cloud)

2. **Launch Instance**:
   - Select GPU (A100 recommended for fast training)
   - Choose Ubuntu with NVIDIA drivers

3. **SSH into Instance**:
   ```bash
   ssh ubuntu@<instance-ip>
   ```

4. **Run Docker Container**:
   ```bash
   # Pull your image
   docker pull yourusername/yolov8n-refdet:latest
   
   # Upload datasets (use scp or cloud storage)
   scp -r datasets/ ubuntu@<instance-ip>:/home/ubuntu/
   
   # Run training
   docker run --gpus all --shm-size=16g \
     -e WANDB_API_KEY=your_key_here \
     -v /home/ubuntu/datasets:/workspace/datasets:ro \
     -v /home/ubuntu/checkpoints:/workspace/checkpoints \
     yourusername/yolov8n-refdet:latest \
     python train.py --stage 2 --epochs 100 --n_way 2 --n_query 4 --use_wandb
   ```

---

### Google Cloud Platform (Vertex AI)

**Best for**: Enterprise deployment, integration with GCP services

#### Setup Steps:

1. **Push to GCR/Artifact Registry**:
   ```bash
   # Tag for GCR
   docker tag yourusername/yolov8n-refdet:latest \
     gcr.io/<project-id>/yolov8n-refdet:latest
   
   # Push to GCR
   docker push gcr.io/<project-id>/yolov8n-refdet:latest
   ```

2. **Create Training Job**:
   ```bash
   gcloud ai custom-jobs create \
     --region=us-central1 \
     --display-name=yolov8n-refdet-training \
     --worker-pool-spec=machine-type=n1-standard-8,\
       replica-count=1,\
       accelerator-type=NVIDIA_TESLA_V100,\
       accelerator-count=1,\
       container-image-uri=gcr.io/<project-id>/yolov8n-refdet:latest
   ```

---

## Training Configuration

### Essential Arguments

```bash
python train.py \
  --stage 2 \                      # Training stage (2=meta-learning)
  --epochs 100 \                   # Number of epochs
  --n_way 2 \                      # Classes per episode
  --n_query 4 \                    # Query samples per class
  --use_wandb \                    # Enable W&B logging
  --data_root /workspace/datasets/train/samples \
  --annotations /workspace/datasets/train/annotations/annotations.json \
  --checkpoint_dir /workspace/checkpoints
```

### Advanced Configuration

```bash
python train.py \
  --stage 2 \
  --epochs 100 \
  --n_way 2 \
  --n_query 4 \
  --n_episodes 100 \               # Episodes per epoch
  --batch_size 4 \                 # Batch size
  --lr 1e-4 \                      # Learning rate
  --weight_decay 0.05 \            # Weight decay
  --gradient_accumulation 1 \      # Gradient accumulation steps
  --use_triplet \                  # Enable triplet loss
  --triplet_ratio 0.3 \            # Ratio of triplet batches
  --mixed_precision \              # Use AMP (recommended)
  --num_workers 4 \                # DataLoader workers
  --use_wandb \
  --wandb_project yolov8n-refdet \
  --wandb_name stage2_100epochs \
  --checkpoint_dir /workspace/checkpoints
```

### Resume Training

```bash
python train.py \
  --stage 2 \
  --epochs 100 \
  --resume /workspace/checkpoints/checkpoint_epoch_50.pt \
  --use_wandb \
  --checkpoint_dir /workspace/checkpoints
```

---

## Dataset Upload Methods

### Method 1: Cloud Storage (Recommended)

**Upload to cloud storage, then download in container**:

```bash
# 1. Upload to Google Cloud Storage
gsutil -m cp -r datasets/ gs://your-bucket/datasets/

# 2. In container, download datasets
apt-get update && apt-get install -y gsutil
gsutil -m cp -r gs://your-bucket/datasets/ /workspace/datasets/
```

**Or use AWS S3**:
```bash
aws s3 sync datasets/ s3://your-bucket/datasets/
aws s3 sync s3://your-bucket/datasets/ /workspace/datasets/
```

### Method 2: Direct Upload (Smaller Datasets)

**Via SCP**:
```bash
scp -r datasets/ user@gpu-instance:/workspace/datasets/
```

### Method 3: Platform Storage

- **Vast.ai**: Use their storage system
- **RunPod**: Network volumes
- **Lambda Labs**: Direct instance storage

---

## Resource Requirements

### Minimum Requirements
- **GPU**: 16GB VRAM (RTX 4000, T4)
- **RAM**: 16GB system RAM
- **Disk**: 50GB
- **Shared Memory**: 8GB (`--shm-size=8g`)

### Recommended Requirements
- **GPU**: 24GB VRAM (RTX 3090, RTX 4090, A6000)
- **RAM**: 32GB+ system RAM
- **Disk**: 100GB
- **Shared Memory**: 16GB (`--shm-size=16g`)

### For Large-Scale Training
- **GPU**: 40-80GB VRAM (A100, H100)
- **RAM**: 64GB+ system RAM
- **Disk**: 200GB+
- **Shared Memory**: 32GB (`--shm-size=32g`)

---

## Monitoring Training

### Weights & Biases (Recommended)

1. **Setup W&B**:
   ```bash
   # Create account at wandb.ai
   # Get API key from https://wandb.ai/authorize
   ```

2. **Set API Key**:
   ```bash
   -e WANDB_API_KEY=your_key_here
   ```

3. **Monitor Training**:
   - Visit: `https://wandb.ai/<entity>/<project>`
   - Real-time metrics, loss curves, system stats
   - Model checkpoints and artifacts

### Container Logs

```bash
# Follow logs in real-time
docker logs -f <container-id>

# Or on Vast.ai/RunPod via web terminal
docker exec -it <container-id> /bin/bash
tail -f /workspace/logs/training.log
```

### TensorBoard (Alternative)

```bash
# Expose TensorBoard port
docker run --gpus all --shm-size=16g -p 6006:6006 \
  -v /path/to/checkpoints:/workspace/checkpoints \
  yourusername/yolov8n-refdet:latest \
  tensorboard --logdir=/workspace/checkpoints --host=0.0.0.0
```

---

## Troubleshooting

### Issue: CUDA Out of Memory

**Solutions**:
1. Reduce batch size: `--batch_size 2`
2. Reduce num_workers: `--num_workers 2`
3. Enable gradient accumulation: `--gradient_accumulation 2`
4. Use mixed precision: `--mixed_precision`
5. Use GPU with more VRAM

### Issue: Shared Memory Error

**Error**: `RuntimeError: DataLoader worker exited unexpectedly`

**Solution**: Increase shared memory:
```bash
docker run --shm-size=16g ...  # or higher
```

### Issue: Cannot Access GPU

**Solutions**:
1. Verify GPU is available: `docker run --gpus all ... nvidia-smi`
2. Check Docker GPU support: `docker run --rm --gpus all nvidia/cuda:12.6.0-base nvidia-smi`
3. Install NVIDIA Container Toolkit on host

### Issue: Slow Data Loading

**Solutions**:
1. Increase num_workers: `--num_workers 8`
2. Use SSD storage for datasets
3. Increase shared memory: `--shm-size=32g`
4. Enable dataset caching in code

### Issue: Training Hangs

**Common Causes**:
1. Dataset path incorrect
2. Insufficient shared memory
3. Network connectivity (downloading models)

**Debug Steps**:
```bash
# Enter container
docker exec -it <container-id> /bin/bash

# Verify datasets
ls /workspace/datasets/train/samples
cat /workspace/datasets/train/annotations/annotations.json | head

# Test CUDA
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

# Run with verbose logging
python train.py --stage 2 --epochs 1 --n_episodes 1
```

---

## Cost Optimization

### 1. Use Spot/Interruptible Instances
- **Vast.ai**: Interruptible instances (50-70% cheaper)
- **GCP**: Preemptible VMs
- **AWS**: Spot instances

### 2. Checkpoint Frequently
```bash
--save_interval 5  # Save every 5 epochs
```

### 3. Use Smaller GPUs for Testing
- Test with RTX 3060 (12GB) or T4 (16GB)
- Scale to A100 for production

### 4. Monitor Costs
- Set spending limits on cloud platforms
- Use W&B to track GPU utilization
- Stop instances when not training

---

## Security Best Practices

### 1. Use Environment Variables for Secrets
```bash
# Never hardcode API keys
-e WANDB_API_KEY=${WANDB_API_KEY}
```

### 2. Use Read-Only Mounts for Datasets
```bash
-v /path/to/datasets:/workspace/datasets:ro
```

### 3. Limit Container Capabilities
```bash
--security-opt=no-new-privileges:true
```

### 4. Use Private Docker Registry
```bash
# For sensitive projects
docker tag yourusername/yolov8n-refdet:latest \
  registry.company.com/yolov8n-refdet:latest
docker push registry.company.com/yolov8n-refdet:latest
```

---

## Example Training Runs

### Stage 2: Meta-Learning (100 epochs)
```bash
docker run --gpus all --shm-size=16g \
  -e WANDB_API_KEY=${WANDB_API_KEY} \
  -v $(pwd)/datasets:/workspace/datasets:ro \
  -v $(pwd)/checkpoints:/workspace/checkpoints \
  yourusername/yolov8n-refdet:latest \
  python train.py --stage 2 --epochs 100 --n_way 2 --n_query 4 \
  --use_wandb --mixed_precision --gradient_accumulation 1
```

**Expected Time**: 8-12 hours on RTX 4090

### Stage 2 with Triplet Loss
```bash
docker run --gpus all --shm-size=16g \
  -e WANDB_API_KEY=${WANDB_API_KEY} \
  -v $(pwd)/datasets:/workspace/datasets:ro \
  -v $(pwd)/checkpoints:/workspace/checkpoints \
  yourusername/yolov8n-refdet:latest \
  python train.py --stage 2 --epochs 100 --n_way 2 --n_query 4 \
  --use_triplet --triplet_ratio 0.3 --use_wandb
```

**Expected Time**: 10-14 hours on RTX 4090

### Stage 3: Fine-tuning (50 epochs)
```bash
docker run --gpus all --shm-size=16g \
  -e WANDB_API_KEY=${WANDB_API_KEY} \
  -v $(pwd)/datasets:/workspace/datasets:ro \
  -v $(pwd)/checkpoints:/workspace/checkpoints \
  yourusername/yolov8n-refdet:latest \
  python train.py --stage 3 --epochs 50 --n_way 2 --n_query 4 \
  --resume /workspace/checkpoints/stage2_best_model.pt \
  --use_wandb
```

**Expected Time**: 4-6 hours on RTX 4090

---

## Next Steps

After successful training:

1. **Evaluate Model**:
   ```bash
   docker run --gpus all \
     -v $(pwd)/datasets:/workspace/datasets:ro \
     -v $(pwd)/checkpoints:/workspace/checkpoints:ro \
     -v $(pwd)/outputs:/workspace/outputs \
     yourusername/yolov8n-refdet:latest \
     python evaluate.py --checkpoint /workspace/checkpoints/best_model.pt
   ```

2. **Download Checkpoints**:
   ```bash
   # From cloud instance
   scp user@instance:/workspace/checkpoints/best_model.pt ./
   
   # Or use cloud storage
   gsutil cp /workspace/checkpoints/best_model.pt gs://your-bucket/
   ```

3. **Deploy for Inference**:
   - See `INFERENCE_GUIDE.md` (if available)
   - Convert to ONNX/TensorRT for optimization
   - Deploy on edge devices or API servers

---

## Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/yolov8n-refdet/issues)
- **Documentation**: See `README.md`, `TRAINING_GUIDE.md`
- **Wandb Dashboard**: Track experiments at wandb.ai

---

**Happy Training! 🚀**
