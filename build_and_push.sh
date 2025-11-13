#!/bin/bash
# ============================================================================
# Build and Push Docker Image to Docker Hub
# ============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
DOCKER_USERNAME="${DOCKER_USERNAME:-yourusername}"
IMAGE_NAME="yolov8n-refdet"
VERSION="${VERSION:-latest}"
FULL_IMAGE="${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION}"

echo -e "${GREEN}============================================================================${NC}"
echo -e "${GREEN}Building and Pushing YOLOv8n-RefDet Docker Image${NC}"
echo -e "${GREEN}============================================================================${NC}"
echo ""
echo -e "Image: ${YELLOW}${FULL_IMAGE}${NC}"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    exit 1
fi

# Check if logged in to Docker Hub
echo -e "${YELLOW}Step 1: Checking Docker Hub login...${NC}"
if ! docker info | grep -q "Username"; then
    echo -e "${YELLOW}Not logged in. Please login to Docker Hub:${NC}"
    docker login
fi
echo -e "${GREEN}✓ Logged in to Docker Hub${NC}"
echo ""

# Build the image
echo -e "${YELLOW}Step 2: Building Docker image...${NC}"
echo "This may take 10-15 minutes depending on your internet connection..."
docker build -t ${FULL_IMAGE} .
echo -e "${GREEN}✓ Image built successfully${NC}"
echo ""

# Show image size
echo -e "${YELLOW}Image details:${NC}"
docker images ${FULL_IMAGE}
echo ""

# Push to Docker Hub
echo -e "${YELLOW}Step 3: Pushing image to Docker Hub...${NC}"
echo "This may take 10-20 minutes depending on your internet connection..."
docker push ${FULL_IMAGE}
echo -e "${GREEN}✓ Image pushed successfully${NC}"
echo ""

# Show final instructions
echo -e "${GREEN}============================================================================${NC}"
echo -e "${GREEN}SUCCESS! Image is ready for cloud GPU training${NC}"
echo -e "${GREEN}============================================================================${NC}"
echo ""
echo -e "${YELLOW}Your image:${NC} ${FULL_IMAGE}"
echo ""
echo -e "${YELLOW}Quick start on cloud GPU:${NC}"
echo ""
echo "docker run --gpus all --shm-size=16g \\"
echo "  -e WANDB_API_KEY=<your-key> \\"
echo "  -v /path/to/datasets:/workspace/datasets:ro \\"
echo "  -v /path/to/checkpoints:/workspace/checkpoints \\"
echo "  ${FULL_IMAGE} \\"
echo "  python train.py --stage 2 --epochs 100 --n_way 2 --n_query 4 --use_wandb"
echo ""
echo -e "${YELLOW}See DOCKER_DEPLOYMENT.md for detailed cloud GPU instructions${NC}"
echo ""
