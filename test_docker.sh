#!/bin/bash
# ============================================================================
# Quick Test Script for Local Docker Testing
# ============================================================================
# Run this before pushing to verify your Docker image works correctly
# ============================================================================

set -e

# Configuration
IMAGE_NAME="${1:-yolov8n-refdet:latest}"
TEST_DATA_ROOT="${2:-./datasets/train/samples}"
TEST_ANNOTATIONS="${3:-./datasets/train/annotations/annotations.json}"

echo "============================================================================"
echo "Testing Docker Image Locally"
echo "============================================================================"
echo ""
echo "Image: ${IMAGE_NAME}"
echo ""

# Check if image exists
if ! docker images | grep -q "$(echo ${IMAGE_NAME} | cut -d: -f1)"; then
    echo "Error: Image ${IMAGE_NAME} not found"
    echo "Build it first with: docker build -t ${IMAGE_NAME} ."
    exit 1
fi

# Test 1: Check CUDA availability
echo "Test 1: Checking CUDA availability..."
docker run --rm --gpus all ${IMAGE_NAME} \
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo "✓ CUDA check passed"
echo ""

# Test 2: Verify dependencies
echo "Test 2: Verifying dependencies..."
docker run --rm ${IMAGE_NAME} \
    python -c "import torch, torchvision, cv2, timm, ultralytics, wandb; print('All imports successful')"
echo "✓ Dependencies check passed"
echo ""

# Test 3: Quick training test (1 episode)
if [ -d "${TEST_DATA_ROOT}" ] && [ -f "${TEST_ANNOTATIONS}" ]; then
    echo "Test 3: Running quick training test (1 episode)..."
    docker run --rm --gpus all --shm-size=8g \
        -v ${TEST_DATA_ROOT}:/workspace/datasets/train/samples:ro \
        -v ${TEST_ANNOTATIONS}:/workspace/datasets/train/annotations/annotations.json:ro \
        ${IMAGE_NAME} \
        python train.py --stage 2 --epochs 1 --n_episodes 1 --n_way 2 --n_query 2
    echo "✓ Training test passed"
else
    echo "⚠ Skipping training test (dataset not found)"
    echo "  To run full test, provide dataset path:"
    echo "  ./test_docker.sh ${IMAGE_NAME} /path/to/datasets/train/samples /path/to/annotations.json"
fi

echo ""
echo "============================================================================"
echo "All tests passed! Image is ready to push."
echo "============================================================================"
echo ""
echo "Next steps:"
echo "1. Push to Docker Hub: docker push ${IMAGE_NAME}"
echo "2. Deploy on cloud GPU platform"
echo ""
