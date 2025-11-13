# Quick Reference: Cloud GPU Platforms for YOLOv8n-RefDet

## Platform Comparison

| Platform | Best For | GPU Options | Price Range | Ease of Use |
|----------|----------|-------------|-------------|-------------|
| **Vast.ai** | Budget training | RTX 3090, 4090, A6000 | $0.20-$0.80/hr | ⭐⭐⭐⭐ |
| **RunPod** | Reliability | RTX 4090, A6000, A100 | $0.40-$2.00/hr | ⭐⭐⭐⭐⭐ |
| **Lambda Labs** | High performance | A100, H100 | $1.10-$2.50/hr | ⭐⭐⭐⭐ |
| **GCP Vertex AI** | Enterprise | T4, V100, A100 | $0.95-$3.67/hr | ⭐⭐⭐ |
| **AWS SageMaker** | Enterprise | V100, A100, H100 | $1.26-$5.00/hr | ⭐⭐⭐ |

---

## Vast.ai Quick Start

### 1. Setup (5 minutes)
```bash
# 1. Create account at vast.ai
# 2. Add payment method
# 3. Click "Create" > "New Instance"
```

### 2. Configuration
```
Docker Image: yourusername/yolov8n-refdet:latest
GPU: RTX 4090 (recommended)
Disk: 100GB
Environment Variables:
  WANDB_API_KEY=your_key
```

### 3. On-Start Script
```bash
python train.py --stage 2 --epochs 100 --n_way 2 --n_query 4 \
  --use_wandb --data_root /workspace/datasets/train/samples \
  --annotations /workspace/datasets/train/annotations/annotations.json \
  --checkpoint_dir /workspace/checkpoints
```

### 4. Upload Datasets
```bash
# Option 1: Via SSH
vast-cli scp -r datasets/ <instance-id>:/workspace/datasets/

# Option 2: Via web interface
# Use Vast.ai's storage browser
```

### 5. Monitor
- Connect via SSH: `vast-cli ssh <instance-id>`
- View logs: `docker logs -f $(docker ps -q)`
- Check W&B: `https://wandb.ai/your-username/yolov8n-refdet`

**Cost**: ~$0.50-$0.80/hr for RTX 4090

---

## RunPod Quick Start

### 1. Setup (3 minutes)
```bash
# 1. Create account at runpod.io
# 2. Go to "Pods" > "Deploy"
```

### 2. Configuration
```
Template: Custom Docker Image
Docker Image: yourusername/yolov8n-refdet:latest
GPU: RTX 4090
Container Disk: 50GB
Volume: 100GB (for datasets)

Environment Variables:
  WANDB_API_KEY=your_key
```

### 3. Upload Datasets
```bash
# Option 1: Via RunPod web terminal
# Click "Connect" > "Start Web Terminal"
# Then upload via their file manager

# Option 2: Via SSH
ssh -i ~/.ssh/key <username>@<pod-ip> -p <port>
scp -r datasets/ <username>@<pod-ip>:/workspace/datasets/
```

### 4. Start Training
```bash
# Via web terminal or SSH
docker run --gpus all --shm-size=16g \
  -e WANDB_API_KEY=${WANDB_API_KEY} \
  -v /workspace/datasets:/workspace/datasets:ro \
  -v /workspace/checkpoints:/workspace/checkpoints \
  yourusername/yolov8n-refdet:latest \
  python train.py --stage 2 --epochs 100 --n_way 2 --n_query 4 --use_wandb
```

**Cost**: ~$0.70-$1.00/hr for RTX 4090

---

## Lambda Labs Quick Start

### 1. Setup (5 minutes)
```bash
# 1. Create account at lambdalabs.com/service/gpu-cloud
# 2. Add SSH key
# 3. Launch instance with A100
```

### 2. Connect & Setup
```bash
# SSH into instance
ssh ubuntu@<instance-ip>

# Install Docker (if not installed)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 3. Upload Datasets
```bash
# From local machine
scp -r datasets/ ubuntu@<instance-ip>:/home/ubuntu/datasets/
```

### 4. Run Training
```bash
# Pull image
docker pull yourusername/yolov8n-refdet:latest

# Start training
docker run --gpus all --shm-size=16g \
  -e WANDB_API_KEY=your_key \
  -v /home/ubuntu/datasets:/workspace/datasets:ro \
  -v /home/ubuntu/checkpoints:/workspace/checkpoints \
  yourusername/yolov8n-refdet:latest \
  python train.py --stage 2 --epochs 100 --n_way 2 --n_query 4 --use_wandb
```

**Cost**: ~$1.10-$1.50/hr for A100 (40GB)

---

## Common Commands

### Check GPU Status
```bash
nvidia-smi

# In container
docker run --rm --gpus all yourusername/yolov8n-refdet:latest nvidia-smi
```

### Monitor Training
```bash
# View logs
docker logs -f <container-id>

# Enter container
docker exec -it <container-id> /bin/bash

# Check GPU usage
watch -n 1 nvidia-smi
```

### Download Checkpoints
```bash
# Copy from container
docker cp <container-id>:/workspace/checkpoints/best_model.pt ./

# Or via SCP
scp user@gpu-instance:/workspace/checkpoints/best_model.pt ./
```

### Stop Training
```bash
# Graceful stop
docker stop <container-id>

# Force stop
docker kill <container-id>
```

---

## Troubleshooting

### "RuntimeError: CUDA out of memory"
```bash
# Reduce batch size
--batch_size 2

# Or use gradient accumulation
--gradient_accumulation 2
```

### "Cannot allocate memory in static TLS block"
```bash
# Increase shared memory
docker run --shm-size=16g ...
```

### "Dataset not found"
```bash
# Check mount paths
docker exec -it <container-id> ls /workspace/datasets/

# Verify paths in command
--data_root /workspace/datasets/train/samples
--annotations /workspace/datasets/train/annotations/annotations.json
```

---

## Cost Estimation

### 100 Epochs Training

| GPU | Time | Cost/hr | Total Cost |
|-----|------|---------|------------|
| RTX 3090 | ~14h | $0.50 | ~$7 |
| RTX 4090 | ~10h | $0.70 | ~$7 |
| A6000 | ~10h | $0.80 | ~$8 |
| A100 (40GB) | ~6h | $1.50 | ~$9 |
| A100 (80GB) | ~5h | $2.50 | ~$12.50 |

*Approximate costs, may vary by platform and availability*

---

## Support

- **Vast.ai**: [Discord](https://discord.gg/vast-ai)
- **RunPod**: [Discord](https://discord.gg/runpod)
- **Lambda Labs**: support@lambdalabs.com

---

**Happy Training! 🚀**
