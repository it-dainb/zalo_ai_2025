"""
ONNX Export Script for YOLOv8n-RefDet with PSALM Fusion
========================================================

Exports PyTorch model to ONNX with:
- Opset 18 (latest stable)
- ONNXSlim optimization
- 100% accuracy validation
- Prototype-only detection outputs (8 tensors)
- Dynamic batching support
- Cached support features (query_image only input)

Usage:
    # Export with default settings
    python onnx_export.py --checkpoint path/to/checkpoint.pth
    
    # Export with dynamic batch size
    python onnx_export.py --checkpoint path/to/checkpoint.pth --dynamic
    
    # Export with simplification only (no onnxslim)
    python onnx_export.py --checkpoint path/to/checkpoint.pth --no-slim
    
    # Export without validation
    python onnx_export.py --checkpoint path/to/checkpoint.pth --no-validate
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.models import YOLORefDet


class ONNXExportWrapper(nn.Module):
    """
    Wrapper for ONNX export to handle prototype detection outputs.
    
    This wrapper simplifies the model for ONNX export by:
    1. Flattening dictionary outputs to tuples
    2. Handling cached support features
    3. Providing cleaner input/output signatures
    
    Note: YOLORefDet only outputs prototype detections (no standard YOLOv8 head).
    """
    
    def __init__(
        self,
        model: YOLORefDet,
        cache_support: bool = True,
    ):
        """
        Args:
            model: YOLOv8nRefDet model
            cache_support: Whether to cache support features
        """
        super().__init__()
        self.model = model
        self.cache_support = cache_support
        self._support_cached = False
    
    def forward(self, query_image: torch.Tensor, support_images: Optional[torch.Tensor] = None):
        """
        Forward pass for ONNX export.
        
        Args:
            query_image: (B, 3, 640, 640)
            support_images: (K, 3, 256, 256) or None to use cache
        
        Returns:
            Tuple of 8 outputs (prototype detection only):
            - proto_boxes_p2, proto_boxes_p3, proto_boxes_p4, proto_boxes_p5 (4 tensors)
            - proto_sim_p2, proto_sim_p3, proto_sim_p4, proto_sim_p5 (4 tensors)
        """
        if self.cache_support and support_images is not None:
            # Cache support features on first call
            if not self._support_cached:
                self.model.set_reference_images(support_images, average_prototypes=True)
                self._support_cached = True
        
        # Forward through model (always returns prototype outputs)
        outputs = self.model(
            query_image,
            support_images=support_images if not self.cache_support else None,
            use_cache=self.cache_support,
        )
        
        # Flatten outputs to tuple (ONNX-friendly)
        # 8 outputs (4 scales × 2 outputs)
        return (
            *outputs['prototype_boxes'],   # 4 tensors
            *outputs['prototype_sim'],     # 4 tensors
        )


def export_to_onnx(
    model: YOLORefDet,
    output_path: str,
    opset_version: int = 18,
    dynamic_axes: bool = False,
    simplify: bool = True,
    use_onnxslim: bool = True,
    verbose: bool = True,
) -> str:
    """
    Export PyTorch model to ONNX format.
    
    Args:
        model: YOLOv8nRefDet model
        output_path: Output ONNX file path
        opset_version: ONNX opset version (default: 18)
        dynamic_axes: Enable dynamic batch size
        simplify: Apply ONNX simplification
        use_onnxslim: Apply ONNXSlim optimization (requires onnxslim package)
        verbose: Print detailed logs
    
    Returns:
        Path to exported ONNX file
    """
    model.eval()
    device = next(model.parameters()).device
    
    if verbose:
        print("\n" + "="*70)
        print(f"ONNX Export Configuration")
        print("="*70)
        print(f"  Opset version: {opset_version}")
        print(f"  Dynamic axes: {dynamic_axes}")
        print(f"  Simplification: {simplify}")
        print(f"  ONNXSlim: {use_onnxslim}")
        print(f"  Output: {output_path}")
        print("="*70 + "\n")
    
    # Prepare wrapper (always cache support features for ONNX export)
    cache_support = True
    wrapper = ONNXExportWrapper(model, cache_support=cache_support)
    wrapper.eval()
    
    # Prepare dummy inputs
    batch_size = 1
    query_dummy = torch.randn(batch_size, 3, 640, 640, device=device)
    support_dummy = torch.randn(3, 3, 256, 256, device=device)
    
    # Cache support features if needed
    if cache_support and support_dummy is not None:
        model.set_reference_images(support_dummy, average_prototypes=True)
    
    # Define input/output names
    input_names = ['query_image']
    
    # Output names: 8 prototype detection outputs (4 scales × 2 outputs)
    scales = ['p2', 'p3', 'p4', 'p5']
    output_names = (
        [f'proto_boxes_{s}' for s in scales] +
        [f'proto_sim_{s}' for s in scales]
    )
    
    # Dynamic axes configuration
    dynamic_axes_config = None
    if dynamic_axes:
        dynamic_axes_config = {
            'query_image': {0: 'batch_size'},
        }
        # Add dynamic axes for all outputs
        for name in output_names:
            dynamic_axes_config[name] = {0: 'batch_size'}
    
    if verbose:
        print("[1/5] Exporting to ONNX...")
        print(f"  Input names: {input_names}")
        print(f"  Output names: {len(output_names)} tensors")
    
    # Export to ONNX
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (query_dummy,),  # Only query_image input (support cached)
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes_config,
            opset_version=opset_version,
            do_constant_folding=True,
            verify=True,
            report=True,
            external_data=False,
        )
    
    if verbose:
        print(f"✓ Initial ONNX export complete")
        onnx_model = onnx.load(output_path)
        print(f"  Model size: {Path(output_path).stat().st_size / 1024 / 1024:.2f} MB")
    
    # Step 2: ONNX simplification
    if simplify:
        if verbose:
            print("\n[2/5] Applying ONNX simplification...")
        try:
            import onnxsim
            
            onnx_model = onnx.load(output_path)
            onnx_model_sim, check = onnxsim.simplify(
                onnx_model,
                dynamic_input_shape=dynamic_axes,
                input_shapes={'query_image': [batch_size, 3, 640, 640]} if not dynamic_axes else None,
            )
            
            if check:
                onnx.save(onnx_model_sim, output_path)
                if verbose:
                    print(f"✓ ONNX simplification complete")
                    print(f"  Model size: {Path(output_path).stat().st_size / 1024 / 1024:.2f} MB")
            else:
                if verbose:
                    print("⚠ ONNX simplification validation failed, using original model")
        except ImportError:
            if verbose:
                print("⚠ onnxsim not installed, skipping simplification")
                print("  Install: pip install onnx-simplifier")
    
    # Step 3: ONNXSlim optimization
    if use_onnxslim:
        if verbose:
            print("\n[3/5] Applying ONNXSlim optimization...")
        try:
            import onnxslim
            
            slim_path = output_path.replace('.onnx', '_slim.onnx')
            onnxslim.slim(output_path, slim_path)
            
            # Compare sizes
            original_size = Path(output_path).stat().st_size / 1024 / 1024
            slim_size = Path(slim_path).stat().st_size / 1024 / 1024
            
            if verbose:
                print(f"✓ ONNXSlim optimization complete")
                print(f"  Original: {original_size:.2f} MB")
                print(f"  Slimmed: {slim_size:.2f} MB")
                print(f"  Reduction: {(1 - slim_size/original_size)*100:.1f}%")
            
            # Use slimmed model
            output_path = slim_path
            
        except ImportError:
            if verbose:
                print("⚠ onnxslim not installed, skipping optimization")
                print("  Install: pip install onnxslim")
        except Exception as e:
            if verbose:
                print(f"⚠ ONNXSlim optimization failed: {e}")
    
    if verbose:
        print(f"\n[4/5] Verifying ONNX model...")
    
    # Verify ONNX model
    try:
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        if verbose:
            print("✓ ONNX model verification passed")
    except Exception as e:
        print(f"✗ ONNX model verification failed: {e}")
        raise
    
    return output_path


def validate_accuracy(
    pytorch_model: YOLORefDet,
    onnx_path: str,
    num_samples: int = 10,
    rtol: float = 1e-3,
    atol: float = 1e-5,
    verbose: bool = True,
) -> Tuple[bool, Dict[str, float]]:
    """
    Validate ONNX model accuracy against PyTorch model.
    
    Args:
        pytorch_model: Original PyTorch model
        onnx_path: Path to ONNX model
        num_samples: Number of test samples
        rtol: Relative tolerance
        atol: Absolute tolerance
        verbose: Print detailed logs
    
    Returns:
        (is_valid, error_stats) tuple
    """
    if verbose:
        print(f"\n[5/5] Validating accuracy (target: 100% match)...")
    
    pytorch_model.eval()
    device = next(pytorch_model.parameters()).device
    
    # Create ONNX Runtime session
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
    ort_session = ort.InferenceSession(onnx_path, providers=providers)
    
    # Prepare wrapper (always cache support features)
    cache_support = True
    wrapper = ONNXExportWrapper(pytorch_model, cache_support=cache_support)
    wrapper.eval()
    
    max_errors = []
    mean_errors = []
    all_match = True
    
    for i in range(num_samples):
        # Generate random inputs
        query = torch.randn(1, 3, 640, 640, device=device)
        support = torch.randn(3, 3, 256, 256, device=device)
        
        # Cache support features on first iteration
        if i == 0:
            pytorch_model.set_reference_images(support, average_prototypes=True)
        
        # PyTorch inference
        with torch.no_grad():
            pytorch_outputs = wrapper(query)
        
        # ONNX inference (only query_image input)
        ort_inputs = {'query_image': query.cpu().numpy()}
        
        onnx_outputs = ort_session.run(None, ort_inputs)
        
        # Compare outputs
        for j, (pt_out, onnx_out) in enumerate(zip(pytorch_outputs, onnx_outputs)):
            pt_np = pt_out.cpu().numpy()
            
            # Compute errors
            max_error = np.abs(pt_np - onnx_out).max()
            mean_error = np.abs(pt_np - onnx_out).mean()
            
            max_errors.append(max_error)
            mean_errors.append(mean_error)
            
            # Check if within tolerance
            if not np.allclose(pt_np, onnx_out, rtol=rtol, atol=atol):
                all_match = False
                if verbose and i == 0:  # Only print first mismatch
                    print(f"  ⚠ Output {j}: max_error={max_error:.2e}, mean_error={mean_error:.2e}")
    
    # Compute statistics
    error_stats = {
        'max_error': np.max(max_errors),
        'mean_error': np.mean(mean_errors),
        'max_mean_error': np.max(mean_errors),
    }
    
    if verbose:
        if all_match:
            print(f"✓ Accuracy validation passed (100% match)")
        else:
            print(f"⚠ Some outputs exceeded tolerance")
        print(f"  Max error: {error_stats['max_error']:.2e}")
        print(f"  Mean error: {error_stats['mean_error']:.2e}")
        print(f"  Tolerance: rtol={rtol}, atol={atol}")
    
    return all_match, error_stats


def main():
    parser = argparse.ArgumentParser(description='Export YOLOv8n-RefDet to ONNX')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (default: None, use pretrained weights)')
    parser.add_argument('--output', type=str, default='yolov8n_refdet.onnx',
                        help='Output ONNX file path (default: yolov8n_refdet.onnx)')
    parser.add_argument('--opset', type=int, default=18,
                        help='ONNX opset version (default: 18)')
    parser.add_argument('--dynamic', action='store_true',
                        help='Enable dynamic batch size')
    parser.add_argument('--no-simplify', action='store_true',
                        help='Disable ONNX simplification')
    parser.add_argument('--no-slim', action='store_true',
                        help='Disable ONNXSlim optimization')
    parser.add_argument('--validate', action='store_true', default=True,
                        help='Validate accuracy after export (default: True)')
    parser.add_argument('--num-samples', type=int, default=10,
                        help='Number of samples for validation (default: 10)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for export (default: cuda)')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}\n")
    
    # Load model
    print("Loading YOLOv8n-RefDet model...")
    model = YOLORefDet(
        yolo_weights='yolov8n.pt',
        dinov3_model='vit_small_patch16_dinov3.lvd1689m',
        freeze_yolo=False,
        freeze_dinov3=True,
        freeze_dinov3_layers=6,
        conf_thres=0.25,
        iou_thres=0.45,
    ).to(device)
    
    # Load checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("✓ Checkpoint loaded")
    
    model.eval()
    
    # Export to ONNX
    onnx_path = export_to_onnx(
        model=model,
        output_path=args.output,
        opset_version=args.opset,
        dynamic_axes=args.dynamic,
        simplify=not args.no_simplify,
        use_onnxslim=not args.no_slim,
        verbose=True,
    )
    
    # Validate accuracy
    if args.validate:
        is_valid, error_stats = validate_accuracy(
            pytorch_model=model,
            onnx_path=onnx_path,
            num_samples=args.num_samples,
            verbose=True,
        )
        
        if not is_valid:
            print("\n⚠ Warning: Accuracy validation found some mismatches")
            print("  This may be acceptable depending on your use case")
    
    print("\n" + "="*70)
    print("✓ ONNX Export Complete")
    print("="*70)
    print(f"  Output: {onnx_path}")
    print(f"  Size: {Path(onnx_path).stat().st_size / 1024 / 1024:.2f} MB")
    print(f"  Outputs: 8 prototype detection tensors")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
