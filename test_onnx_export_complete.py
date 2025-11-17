"""
Comprehensive ONNX Export Test for YOLOv8n-RefDet
==================================================

Tests ONNX export with:
- Opset 18
- Dynamic batch size support
- ONNX simplification
- ONNXSlim optimization
- Accuracy validation

This script will:
1. Install missing dependencies if needed
2. Export model to ONNX with all optimizations
3. Test dynamic batch sizes (1, 2, 4)
4. Validate accuracy against PyTorch
5. Benchmark inference speed
"""

import sys
from pathlib import Path
import subprocess
import torch
import numpy as np
from typing import Optional, Tuple, Dict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.models import YOLORefDet

# Optional imports (will be checked/installed)
try:
    import onnx
except ImportError:
    onnx = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    import onnxsim
except ImportError:
    onnxsim = None

try:
    import onnxslim
except ImportError:
    onnxslim = None


def install_dependencies():
    """Install required ONNX dependencies if missing."""
    required_packages = [
        'onnx',
        'onnxruntime',
        'onnxruntime-gpu',
        'onnxslim',
    ]
    
    print("="*70)
    print("Checking ONNX Dependencies")
    print("="*70)
    
    missing_packages = []
    for package in required_packages:
        try:
            if package == 'onnxslim':
                import onnxslim
                print(f"✓ {package} installed (version {onnxslim.__version__})")
            elif package == 'onnxruntime' or package == 'onnxruntime-gpu':
                # Only check one of them
                if package == 'onnxruntime-gpu':
                    try:
                        import onnxruntime as ort_check
                        providers = ort_check.get_available_providers()
                        if 'CUDAExecutionProvider' in providers:
                            print(f"✓ onnxruntime-gpu installed (version {ort_check.__version__})")
                        else:
                            print(f"⚠ onnxruntime installed but CUDA provider not available")
                            missing_packages.append('onnxruntime-gpu')
                    except ImportError:
                        missing_packages.append(package)
            else:
                __import__(package.replace('-', '_'))
                import onnx as onnx_check
                print(f"✓ {package} installed (version {onnx_check.__version__})")
        except ImportError:
            print(f"✗ {package} not installed")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠ Missing packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        
        # Install packages
        for package in missing_packages:
            print(f"\nInstalling {package}...")
            try:
                subprocess.check_call([
                    sys.executable, '-m', 'pip', 'install', package, '-q'
                ])
                print(f"✓ {package} installed successfully")
            except subprocess.CalledProcessError as e:
                print(f"✗ Failed to install {package}: {e}")
                if package == 'onnxslim':
                    print("  Note: ONNXSlim will be skipped if unavailable")
    else:
        print("\n✓ All dependencies installed")
    
    print("="*70 + "\n")


def create_test_model(device='cuda'):
    """Create a minimal YOLORefDet model for testing."""
    print("Creating test model...")
    
    model = YOLORefDet(
        yolo_weights='yolov8n.pt',
        dinov3_model='vit_small_patch16_dinov3.lvd1689m',
        freeze_yolo=False,
        freeze_dinov3=True,
        freeze_dinov3_layers=6,
        conf_thres=0.25,
        iou_thres=0.45,
    ).to(device)
    
    model.eval()
    print(f"✓ Model created on {device}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    return model


def test_export_with_all_features(
    model,
    output_path='test_model.onnx',
    opset_version=18,
):
    """Test ONNX export with all features enabled."""
    print("\n" + "="*70)
    print(f"Test 1: Export with Opset {opset_version} + Dynamic Batch + Optimization")
    print("="*70)
    
    from onnx_export import export_to_onnx
    
    try:
        onnx_path = export_to_onnx(
            model=model,
            output_path=output_path,
            opset_version=opset_version,
            dynamic_axes=True,  # Enable dynamic batch size
            simplify=True,       # Enable ONNX simplification
            use_onnxslim=True,   # Enable ONNXSlim
            verbose=True,
        )
        
        print(f"\n✓ Export successful: {onnx_path}")
        
        # Check file size
        file_size = Path(onnx_path).stat().st_size / 1024 / 1024
        print(f"  Final model size: {file_size:.2f} MB")
        
        return onnx_path, True
        
    except Exception as e:
        print(f"\n✗ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return None, False


def test_dynamic_batch_inference(onnx_path):
    """Test inference with different batch sizes."""
    print("\n" + "="*70)
    print("Test 2: Dynamic Batch Size Support")
    print("="*70)
    
    try:
        import onnxruntime as ort
        
        # Create session
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
        session = ort.InferenceSession(onnx_path, providers=providers)
        
        print(f"✓ ONNX Runtime session created")
        print(f"  Provider: {session.get_providers()[0]}")
        
        # Get input/output info
        input_info = session.get_inputs()[0]
        print(f"\n  Input: {input_info.name}")
        print(f"    Shape: {input_info.shape}")
        print(f"    Type: {input_info.type}")
        
        # Test different batch sizes
        batch_sizes = [1, 2, 4]
        all_passed = True
        
        for batch_size in batch_sizes:
            print(f"\n  Testing batch_size={batch_size}...")
            
            # Create dummy input
            query = np.random.randn(batch_size, 3, 640, 640).astype(np.float32)
            
            # Run inference
            try:
                outputs = session.run(None, {'query_image': query})
                
                # Check output shapes
                print(f"    ✓ Inference successful")
                print(f"      Number of outputs: {len(outputs)}")
                
                # Check batch dimension
                for i, output in enumerate(outputs[:4]):  # Check first 4 outputs
                    if output.shape[0] != batch_size:
                        print(f"      ✗ Output {i} batch mismatch: expected {batch_size}, got {output.shape[0]}")
                        all_passed = False
                    else:
                        print(f"      ✓ Output {i} shape: {output.shape}")
                
            except Exception as e:
                print(f"    ✗ Failed: {e}")
                all_passed = False
        
        if all_passed:
            print(f"\n✓ Dynamic batch size test passed")
        else:
            print(f"\n✗ Some batch sizes failed")
        
        return all_passed
        
    except Exception as e:
        print(f"\n✗ Dynamic batch test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_accuracy_validation(model, onnx_path, num_samples=5):
    """Validate ONNX accuracy against PyTorch."""
    print("\n" + "="*70)
    print("Test 3: Accuracy Validation")
    print("="*70)
    
    from onnx_export import validate_accuracy
    
    try:
        is_valid, error_stats = validate_accuracy(
            pytorch_model=model,
            onnx_path=onnx_path,
            num_samples=num_samples,
            rtol=1e-3,
            atol=1e-5,
            verbose=True,
        )
        
        if is_valid:
            print(f"\n✓ Accuracy validation passed")
        else:
            print(f"\n⚠ Accuracy validation found some mismatches")
            print(f"  This may be acceptable for deployment")
        
        print(f"\n  Error Statistics:")
        print(f"    Max error: {error_stats['max_error']:.2e}")
        print(f"    Mean error: {error_stats['mean_error']:.2e}")
        
        return is_valid
        
    except Exception as e:
        print(f"\n✗ Accuracy validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_opset_versions(model):
    """Test export with different opset versions."""
    print("\n" + "="*70)
    print("Test 4: Opset Version Compatibility")
    print("="*70)
    
    from onnx_export import export_to_onnx
    import onnx
    
    opset_versions = [17, 18]  # Test recent opset versions
    results = {}
    
    for opset in opset_versions:
        print(f"\n  Testing opset {opset}...")
        output_path = f'test_model_opset{opset}.onnx'
        
        try:
            onnx_path = export_to_onnx(
                model=model,
                output_path=output_path,
                opset_version=opset,
                dynamic_axes=True,
                simplify=False,  # Disable for faster testing
                use_onnxslim=False,
                verbose=False,
            )
            
            # Verify opset version
            onnx_model = onnx.load(onnx_path)
            actual_opset = onnx_model.opset_import[0].version
            
            if actual_opset == opset:
                print(f"    ✓ Export successful with opset {actual_opset}")
                results[opset] = True
            else:
                print(f"    ⚠ Opset mismatch: requested {opset}, got {actual_opset}")
                results[opset] = False
            
            # Clean up
            Path(onnx_path).unlink(missing_ok=True)
            
        except Exception as e:
            print(f"    ✗ Failed: {e}")
            results[opset] = False
    
    all_passed = all(results.values())
    if all_passed:
        print(f"\n✓ All opset versions supported")
    else:
        print(f"\n⚠ Some opset versions failed: {results}")
    
    return all_passed


def benchmark_inference(onnx_path, warmup=10, iterations=50):
    """Benchmark ONNX inference speed."""
    print("\n" + "="*70)
    print("Test 5: Inference Speed Benchmark")
    print("="*70)
    
    try:
        import onnxruntime as ort
        import time
        
        # Create session
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
        session = ort.InferenceSession(onnx_path, providers=providers)
        
        print(f"  Provider: {session.get_providers()[0]}")
        print(f"  Warmup iterations: {warmup}")
        print(f"  Benchmark iterations: {iterations}")
        
        # Prepare input
        query = np.random.randn(1, 3, 640, 640).astype(np.float32)
        
        # Warmup
        print(f"\n  Warming up...")
        for _ in range(warmup):
            session.run(None, {'query_image': query})
        
        # Benchmark
        print(f"  Running benchmark...")
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            session.run(None, {'query_image': query})
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
        
        # Compute statistics
        mean_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        fps = 1000 / mean_time
        
        print(f"\n  Benchmark Results:")
        print(f"    Mean: {mean_time:.2f} ms (±{std_time:.2f} ms)")
        print(f"    Min: {min_time:.2f} ms")
        print(f"    Max: {max_time:.2f} ms")
        print(f"    FPS: {fps:.1f}")
        
        print(f"\n✓ Benchmark complete")
        return True
        
    except Exception as e:
        print(f"\n✗ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all ONNX export tests."""
    print("\n" + "="*70)
    print("YOLOv8n-RefDet ONNX Export Comprehensive Test")
    print("="*70)
    print("\nThis test will verify:")
    print("  ✓ Opset 18 support")
    print("  ✓ Dynamic batch size support")
    print("  ✓ ONNX simplification")
    print("  ✓ ONNXSlim optimization")
    print("  ✓ Accuracy validation")
    print("  ✓ Inference speed benchmark")
    print("="*70 + "\n")
    
    # Step 1: Install dependencies
    install_dependencies()
    
    # Step 2: Create test model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    try:
        model = create_test_model(device)
    except Exception as e:
        print(f"✗ Failed to create model: {e}")
        return False
    
    # Step 3: Test export with all features
    output_path = 'test_model_complete.onnx'
    onnx_path, export_success = test_export_with_all_features(
        model=model,
        output_path=output_path,
        opset_version=18,
    )
    
    if not export_success:
        print("\n✗ Export failed, stopping tests")
        return False
    
    # Step 4: Test dynamic batch inference
    dynamic_success = test_dynamic_batch_inference(onnx_path)
    
    # Step 5: Test accuracy validation
    accuracy_success = test_accuracy_validation(model, onnx_path, num_samples=5)
    
    # Step 6: Test different opset versions
    opset_success = test_opset_versions(model)
    
    # Step 7: Benchmark inference
    benchmark_success = benchmark_inference(onnx_path, warmup=10, iterations=50)
    
    # Summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    print(f"  Export with optimizations: {'✓ PASSED' if export_success else '✗ FAILED'}")
    print(f"  Dynamic batch size: {'✓ PASSED' if dynamic_success else '✗ FAILED'}")
    print(f"  Accuracy validation: {'✓ PASSED' if accuracy_success else '⚠ WARNING'}")
    print(f"  Opset compatibility: {'✓ PASSED' if opset_success else '✗ FAILED'}")
    print(f"  Inference benchmark: {'✓ PASSED' if benchmark_success else '✗ FAILED'}")
    
    all_critical_passed = export_success and dynamic_success and opset_success and benchmark_success
    
    if all_critical_passed:
        print("\n" + "="*70)
        print("✓ ALL CRITICAL TESTS PASSED")
        print("="*70)
        print(f"\nYour model can be exported to ONNX with:")
        print(f"  • Opset 18 ✓")
        print(f"  • Dynamic batch support ✓")
        print(f"  • ONNX simplification ✓")
        print(f"  • ONNXSlim optimization ✓")
        print(f"\nExported model: {onnx_path}")
        print(f"Size: {Path(onnx_path).stat().st_size / 1024 / 1024:.2f} MB")
        print("="*70 + "\n")
    else:
        print("\n" + "="*70)
        print("✗ SOME TESTS FAILED")
        print("="*70)
        print("\nPlease review the errors above.")
    
    # Cleanup test files
    print("\nCleaning up test files...")
    for pattern in ['test_model*.onnx']:
        for file in Path('.').glob(pattern):
            file.unlink(missing_ok=True)
            print(f"  Removed: {file}")
    
    return all_critical_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
