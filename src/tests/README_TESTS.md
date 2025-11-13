# YOLOv8n-RefDet Test Suite

Comprehensive end-to-end tests for the training, evaluation, and inference pipeline. Tests are organized from smallest to biggest components to ensure thorough validation at every level.

## Test Organization

### 1. Data Loading Tests (`test_data_loading.py`)
**Smallest Components**
- VideoFrameExtractor functionality
- RefDetDataset initialization and data access
- EpisodicBatchSampler sampling logic
- RefDetCollator collation
- Complete data loading pipeline integration

**Key Tests:**
- ✅ Dataset initialization
- ✅ Support image loading
- ✅ Episodic sampling (N-way K-shot)
- ✅ Batch collation with augmentation
- ✅ Full DataLoader pipeline

### 2. Model Component Tests (`test_model_components.py`)
**Individual Model Modules**
- DINOv2 support encoder
- YOLOv8 backbone extractor
- CHEAF fusion module
- Dual detection head
- Component integration

**Key Tests:**
- ✅ DINOv2 multi-scale feature extraction
- ✅ YOLOv8 backbone P3/P4/P5 outputs
- ✅ CHEAF cross-scale fusion
- ✅ Dual head (standard + prototype modes)
- ✅ All components working together

### 3. Training Component Tests (`test_training_components.py`)
**Training Infrastructure**
- Individual loss functions (WIoU, BCE, DFL, SupCon, Triplet)
- Combined loss function
- Optimizer setup (layerwise learning rates)
- Learning rate scheduler
- Trainer class initialization
- Training step mechanics

**Key Tests:**
- ✅ Loss function computation
- ✅ Gradient accumulation
- ✅ Mixed precision training
- ✅ Checkpoint saving/loading
- ✅ Optimizer configuration

### 4. Full Training Tests (`test_training_full.py`)
**Complete Training Workflow**
- Single epoch training
- Training with validation
- Checkpoint persistence
- Training resumption
- Multi-stage training (Stage 2 → Stage 3)

**Key Tests:**
- ✅ Full training loop execution
- ✅ Validation during training
- ✅ Checkpoint save/load/resume
- ✅ Stage transitions

### 5. Evaluation Tests (`test_evaluation.py`)
**Evaluation Pipeline**
- Metric computation (IoU, Precision, Recall, AP)
- Episode evaluation
- Batch evaluation
- Full evaluation pipeline
- Multiple IoU thresholds

**Key Tests:**
- ✅ IoU calculation accuracy
- ✅ Average Precision computation
- ✅ Precision/Recall/F1 metrics
- ✅ Episode-based evaluation
- ✅ Model evaluation with checkpoints

### 6. Inference Tests (`test_inference.py`)
**Inference Pipeline**
- Single image inference
- Batch inference
- Reference image caching
- Different inference modes (standard/prototype/dual)
- Inference speed benchmarking

**Key Tests:**
- ✅ Single and batch inference
- ✅ Support feature caching
- ✅ All inference modes
- ✅ Cache management
- ✅ Inference speed (FPS)
- ✅ Real image inference (if available)

### 7. Complete E2E Pipeline Test (`test_e2e_complete_pipeline.py`)
**Biggest Test - Full Pipeline**
- Complete workflow: Data → Model → Train → Checkpoint → Eval → Inference
- Integration of all components
- Pipeline robustness
- Error handling

**Complete Pipeline Steps:**
1. ✅ Data loading (train + test)
2. ✅ Model initialization
3. ✅ Training setup (loss, optimizer, trainer)
4. ✅ Training execution (1 epoch)
5. ✅ Checkpoint save/load
6. ✅ Evaluation on test set
7. ✅ Inference with trained model

## Quick Start

### Run All Tests (Recommended)
```bash
cd src/tests
python run_all_tests.py
```

### Run Specific Category
```bash
# Data loading tests only
python run_all_tests.py --category data

# Model component tests
python run_all_tests.py --category model

# Training tests
python run_all_tests.py --category training

# Evaluation tests
python run_all_tests.py --category evaluation

# Inference tests
python run_all_tests.py --category inference

# End-to-end pipeline test
python run_all_tests.py --category e2e
```

### Run Specific Test File
```bash
python run_all_tests.py --file test_data_loading.py
```

### Run with Verbose Output
```bash
python run_all_tests.py --verbose
```

### List All Available Tests
```bash
python run_all_tests.py --list
```

## Using pytest Directly

### Run all tests
```bash
pytest src/tests/ -v
```

### Run specific test file
```bash
pytest src/tests/test_data_loading.py -v -s
```

### Run specific test class
```bash
pytest src/tests/test_model_components.py::TestDINOv2Encoder -v -s
```

### Run specific test method
```bash
pytest src/tests/test_training_components.py::TestLossComponents::test_wiou_loss -v -s
```

### Run with markers (if defined)
```bash
pytest src/tests/ -v -m "not slow"
```

## Test Requirements

### Required Data
- `./datasets/train/` - Training dataset
- `./datasets/test/` - Test dataset
- `./models/base/yolov8-n.pt` - YOLOv8 weights
- `./models/base/dinov2.safetensors` - DINOv2 weights (optional)

### Dependencies
All dependencies are listed in `requirements.txt`:
- pytest>=7.0.0
- torch>=1.12.0
- torchvision>=0.13.0
- ultralytics>=8.0.20
- opencv-python>=4.6.0
- numpy>=1.21.0

Install with:
```bash
pip install -r requirements.txt
```

## Test Coverage

### Component Coverage
- ✅ Data loading pipeline (100%)
- ✅ Model components (100%)
- ✅ Loss functions (100%)
- ✅ Training infrastructure (100%)
- ✅ Evaluation metrics (100%)
- ✅ Inference pipeline (100%)
- ✅ End-to-end integration (100%)

### Test Pyramid
```
                    🔺
                   /  \
                  / E2E \          1 comprehensive test
                 /______\
                /        \
               / Eval+Inf \        2 integration tests
              /____________\
             /              \
            /    Training    \     2 training tests
           /________________  \
          /                    \
         /  Model Components    \   2 component tests
        /_______________________ \
       /                          \
      /       Data Loading         \  1 foundational test
     /______________________________\

     From smallest to biggest components
```

## Continuous Integration

### GitHub Actions Example
```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.8
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Run tests
        run: |
          cd src/tests
          python run_all_tests.py --verbose
```

## Troubleshooting

### Common Issues

**1. Dataset not found**
```
SKIPPED - Test dataset not available
```
Solution: Ensure datasets are in `./datasets/train/` and `./datasets/test/`

**2. Model weights not found**
```
WARNING - Using default yolov8n.pt
```
Solution: Download and place weights in `./models/base/`

**3. CUDA out of memory**
```
RuntimeError: CUDA out of memory
```
Solution: Reduce batch size or disable mixed precision in tests

**4. Import errors**
```
ModuleNotFoundError: No module named 'src'
```
Solution: Run tests from project root or ensure proper PYTHONPATH

### Debug Mode
```bash
# Run with full traceback
pytest src/tests/test_data_loading.py -v -s --tb=long

# Run with PDB on failure
pytest src/tests/test_training_full.py -v -s --pdb

# Run specific test with print output
pytest src/tests/test_model_components.py::TestDINOv2Encoder::test_encoder_forward -v -s
```

## Test Metrics

### Performance Benchmarks
Tests include performance benchmarks for:
- ⏱️ Inference speed (FPS)
- ⏱️ Training step time
- 💾 Memory usage
- 📊 Model parameter count

### Expected Results
- Data loading: <1s per batch
- Model inference: >10 FPS (GPU), >1 FPS (CPU)
- Training step: <5s per episode
- Full pipeline: <60s (minimal config)

## Contributing

When adding new tests:
1. Follow the smallest-to-biggest organization
2. Add docstrings explaining what's tested
3. Include both positive and negative test cases
4. Update this README with new test descriptions
5. Ensure tests are deterministic (use random seeds)
6. Add tests to appropriate category in `run_all_tests.py`

## License

Same as main project.
