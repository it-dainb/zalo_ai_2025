"""
Pytest Configuration
====================

Configure pytest settings and fixtures for the test suite.
"""

import pytest
import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture(scope="session")
def project_root():
    """Get project root directory"""
    return Path(__file__).parent.parent.parent


@pytest.fixture(scope="session")
def device():
    """Get compute device for tests"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture(scope="session")
def cuda_available():
    """Check if CUDA is available"""
    return torch.cuda.is_available()


def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "data: marks tests that require dataset"
    )
    config.addinivalue_line(
        "markers", "e2e: marks end-to-end integration tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection"""
    # Skip GPU tests if CUDA not available
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="CUDA not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)


# Test session info
def pytest_sessionstart(session):
    """Print session start info"""
    print("\n" + "="*70)
    print("YOLOv8n-RefDet Test Suite")
    print("="*70)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("="*70 + "\n")


def pytest_sessionfinish(session, exitstatus):
    """Print session finish info"""
    print("\n" + "="*70)
    print("Test Session Complete")
    print("="*70 + "\n")
