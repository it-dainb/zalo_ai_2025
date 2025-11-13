# Agent Guidelines for YOLOv8n-RefDet

## Build/Test Commands
- **Run all tests**: `cd src/tests && python run_all_tests.py` or `pytest src/tests/ -v`
- **Run single test**: `pytest src/tests/test_<name>.py -v -s`
- **Run specific test method**: `pytest src/tests/test_<file>.py::TestClass::test_method -v -s`
- **Train model**: `python train.py --stage 2 --epochs 100 --n_way 2 --n_query 4`
- **Evaluate model**: `python evaluate.py --checkpoint <path> --test_data_root ./datasets/test/samples`

## Code Style
- **Imports**: Group stdlib, third-party (torch, cv2, numpy), then local (src.*); use absolute imports from `src/`
- **Docstrings**: Use triple-quoted docstrings with module overview at top; include Args/Returns for functions/classes
- **Types**: Use type hints for function signatures (`-> Type`, `: Type`); import from `typing` (Dict, List, Optional, Tuple)
- **Naming**: snake_case for functions/variables, PascalCase for classes, UPPER_CASE for constants
- **Error Handling**: Use try/except with specific exceptions; validate inputs early; provide informative error messages
- **Architecture**: Module structure: models/, datasets/, losses/, training/, augmentations/, tests/
- **Comments**: Use inline comments for complex logic; include paper references for loss functions and models
- **Device**: Always specify `.to(device)` for tensors/models; support both CPU and CUDA
- **Paths**: Use `pathlib.Path` for file paths, not string concatenation

## Project Context
This is a PyTorch-based few-shot UAV object detection system using YOLOv8n backbone + DINOv2 encoder with 3-stage training pipeline (base, meta-learning, fine-tuning).
