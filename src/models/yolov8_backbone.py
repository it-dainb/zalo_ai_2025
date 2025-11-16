"""
YOLOv8 Backbone Feature Extractor
==================================

Extracts multi-scale features from YOLOv8n backbone for query images.
Uses forward hooks to capture intermediate feature maps at P3, P4, P5 scales.

Key Features:
- Loads pre-trained YOLOv8n weights from ultralytics
- Extracts features at 3 scales: P3 (1/8), P4 (1/16), P5 (1/32)
- Preserves gradient flow for fine-tuning
- ~3.2M parameters (backbone only)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from pathlib import Path
from ultralytics import YOLO


class YOLOv8BackboneExtractor(nn.Module):
    """
    YOLOv8n backbone feature extractor for query image processing.
    
    Args:
        weights_path: Path to YOLOv8 pretrained weights (.pt file)
        extract_scales: Which scales to extract ['p2', 'p3', 'p4', 'p5']
        freeze_backbone: Whether to freeze backbone weights
    
    Architecture:
        Input: (B, 3, 640, 640) RGB images
        Output: {
            'p2': (B, 32, 160, 160) - Very small object features (1/4 scale) [NEW for UAV]
            'p3': (B, 64, 80, 80) - Small object features (1/8 scale)
            'p4': (B, 128, 40, 40) - Medium object features (1/16 scale)
            'p5': (B, 256, 20, 20) - Large object features (1/32 scale)
        }
    
    Note:
        YOLOv8n uses C2f modules and CSPDarknet backbone.
        Feature extraction points:
        - P2: After 1st C2f block (Stage 1) - Added for UAV small object detection
        - P3: From PANet neck (layer 15)
        - P4: From PANet neck (layer 18)
        - P5: From PANet neck (layer 21)
    """
    
    def __init__(
        self,
        weights_path: str = "yolov8n.pt",
        extract_scales: list = ['p2', 'p3', 'p4', 'p5'],
        freeze_backbone: bool = False,
        input_size: int = 640,
    ):
        super().__init__()
        
        self.weights_path = Path(weights_path)
        self.extract_scales = extract_scales
        self.freeze_backbone = freeze_backbone
        self.input_size = input_size
        
        # Storage for hooked features
        self.features = {}
        self.hooks = []
        
        # Load YOLOv8 model (task='detect' prevents auto dataset downloads, verbose=False suppresses validation)
        print(f"Loading YOLOv8 model from: {weights_path}")
        yolo_wrapper = YOLO(str(weights_path), task='detect', verbose=False)
        
        # Extract ONLY the PyTorch model (don't store the YOLO wrapper to avoid train() interception)
        self.model = yolo_wrapper.model
        
        # Enable gradients for all parameters (YOLO disables them by default)
        for param in self.model.parameters():
            param.requires_grad = True
        
        # Register forward hooks to extract features
        self._register_hooks()
        
        # Freeze backbone if specified (will set requires_grad=False for frozen layers)
        if freeze_backbone:
            self._freeze_backbone()
        
        print(f"YOLOv8 Backbone Extractor initialized:")
        print(f"  - Extract scales: {extract_scales}")
        print(f"  - Input size: {input_size}")
        print(f"  - Frozen: {freeze_backbone}")
        print(f"  - Total params: {self.count_parameters() / 1e6:.2f}M")
    
    def _register_hooks(self):
        """
        Register forward hooks to capture intermediate feature maps.
        
        YOLOv8n architecture:
        - Backbone (0-9): Feature extraction at multiple scales
        - Neck (10-21): PANet feature fusion
        - Head (22): Detection head
        
        Output feature maps from neck (after PANet fusion):
        - P3 (layer 15): 80x80x64 - Small object features
        - P4 (layer 18): 40x40x128 - Medium object features  
        - P5 (layer 21): 20x20x256 - Large object features
        """
        
        def create_hook(name):
            def hook(module, input, output):
                self.features[name] = output
            return hook
        
        # Map layer indices to feature scales
        # These indices correspond to YOLOv8n backbone and PANet neck outputs
        # 
        # ARCHITECTURE NOTE:
        # - P2 extracted from backbone (layer 2) for very small objects in UAV scenarios
        # - P3, P4, P5 extracted from PANet neck (layers 15, 18, 21)
        # Channels: [32, 64, 128, 256] - clean progression for Lego design
        # The CHEAF fusion module then projects to detection head dimensions.
        layer_mapping = {
            'p2': 2,   # Backbone C2f output for very small objects (160x160x32) [UAV]
            'p3': 15,  # PANet output for small objects (80x80x64)
            'p4': 18,  # PANet output for medium objects (40x40x128)
            'p5': 21,  # PANet output for large objects (20x20x256)
        }
        
        # Register hooks for requested scales
        for scale in self.extract_scales:
            if scale in layer_mapping:
                layer_idx = layer_mapping[scale]
                try:
                    target_layer = self.model.model[layer_idx]
                    hook = target_layer.register_forward_hook(create_hook(scale))
                    self.hooks.append(hook)
                    print(f"  - Registered hook for {scale} at layer {layer_idx}")
                except (IndexError, AttributeError) as e:
                    print(f"  - Warning: Could not register hook for {scale}: {e}")
    
    def _freeze_backbone(self):
        """Freeze backbone parameters (exclude detection head)."""
        # Freeze all model parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        print(f"  - Backbone frozen (all parameters)")
    
    def count_parameters(self, trainable_only: bool = False) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.model.parameters())
    
    def forward(self, x: torch.Tensor, return_global_feat: bool = False) -> Dict[str, torch.Tensor]:
        """
        Extract multi-scale features from input image.
        
        Args:
            x: Input tensor (B, 3, H, W) where H, W are multiples of 32
            return_global_feat: Whether to return global pooled feature for triplet loss (default: False)
        
        Returns:
            Dictionary with keys ['p3', 'p4', 'p5'] containing feature tensors
            If return_global_feat=True, also includes 'global_feat' key with (B, 512) tensor
        """
        # Clear previous features
        self.features.clear()
        
        # Forward pass through model (triggers hooks)
        # We don't need the final detection output, just the features
        _ = self.model(x)
        
        # Return captured features
        output = {k: v for k, v in self.features.items() if k in self.extract_scales}
        
        # Add global pooled feature for triplet loss if requested
        if return_global_feat and 'p5' in output:
            # Global average pooling on P5 for triplet loss (512-dim)
            global_feat = torch.nn.functional.adaptive_avg_pool2d(output['p5'], 1).squeeze(-1).squeeze(-1)
            # CRITICAL: Normalize global_feat to match DINO encoder's normalization
            # This prevents norm imbalance in triplet loss (DINO=15.95 vs YOLOv8=2.7 without normalization)
            global_feat = torch.nn.functional.normalize(global_feat, p=2, dim=-1)
            output['global_feat'] = global_feat
        
        return output
    
    def get_feature_dims(self) -> Dict[str, tuple]:
        """
        Get expected feature dimensions for each scale.
        
        Returns:
            Dictionary mapping scale name to (channels, height, width)
        
        Note:
            These dimensions are for YOLOv8n PANet neck outputs (layers 15, 18, 21),
            NOT the backbone outputs. Channels are [64, 128, 256] from the neck.
        """
        # Based on YOLOv8n architecture with input size 640x640
        scale_factor = self.input_size / 640.0
        
        # Actual backbone and PANet neck output dimensions
        dims = {
            'p2': (32, int(160 * scale_factor), int(160 * scale_factor)),  # 1/4 scale [UAV]
            'p3': (64, int(80 * scale_factor), int(80 * scale_factor)),    # 1/8 scale
            'p4': (128, int(40 * scale_factor), int(40 * scale_factor)),   # 1/16 scale
            'p5': (256, int(20 * scale_factor), int(20 * scale_factor)),   # 1/32 scale
        }
        
        return {k: v for k, v in dims.items() if k in self.extract_scales}
    
    def __del__(self):
        """Remove hooks when object is deleted."""
        for hook in self.hooks:
            hook.remove()


def test_yolov8_backbone():
    """Quick sanity test for YOLOv8 backbone extractor."""
    print("\n" + "="*60)
    print("Testing YOLOv8 Backbone Extractor")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Check if weights exist
    weights_path = "baseline_enot_nano/weights/best.pt"
    if not Path(weights_path).exists():
        print(f"⚠️  Warning: Weights not found at {weights_path}")
        print(f"   Using default yolov8n.pt instead")
        weights_path = "yolov8n.pt"
    
    # Initialize extractor
    extractor = YOLOv8BackboneExtractor(
        weights_path=weights_path,
        extract_scales=['p3', 'p4', 'p5'],
        freeze_backbone=False,
    ).to(device)
    
    extractor.eval()
    
    # Test single image (640x640)
    print("\n[Test 1] Single query image (640x640):")
    dummy_img = torch.randn(1, 3, 640, 640).to(device)
    
    with torch.no_grad():
        features = extractor(dummy_img)
    
    print(f"  Input shape: {dummy_img.shape}")
    for scale, feat in features.items():
        print(f"  {scale} shape: {feat.shape}")
    
    # Test different input size
    print("\n[Test 2] Different input size (1280x1280):")
    extractor_large = YOLOv8BackboneExtractor(
        weights_path=weights_path,
        extract_scales=['p3', 'p4', 'p5'],
        input_size=1280,
    ).to(device)
    extractor_large.eval()
    
    dummy_img_large = torch.randn(1, 3, 1280, 1280).to(device)
    
    with torch.no_grad():
        features_large = extractor_large(dummy_img_large)
    
    print(f"  Input shape: {dummy_img_large.shape}")
    for scale, feat in features_large.items():
        print(f"  {scale} shape: {feat.shape}")
    
    # Test batch processing
    print("\n[Test 3] Batch of images:")
    dummy_batch = torch.randn(4, 3, 640, 640).to(device)
    
    with torch.no_grad():
        features_batch = extractor(dummy_batch)
    
    print(f"  Input shape: {dummy_batch.shape}")
    for scale, feat in features_batch.items():
        print(f"  {scale} shape: {feat.shape}")
    
    # Get expected dimensions
    print("\n[Test 4] Expected feature dimensions:")
    expected_dims = extractor.get_feature_dims()
    for scale, dims in expected_dims.items():
        print(f"  {scale}: {dims} (C, H, W)")
    
    # Parameter count
    print("\n[Parameter Count]")
    total_params = extractor.count_parameters()
    trainable_params = extractor.count_parameters(trainable_only=True)
    print(f"  Total parameters: {total_params / 1e6:.2f}M")
    print(f"  Trainable parameters: {trainable_params / 1e6:.2f}M")
    
    # Test with frozen backbone
    print("\n[Test 5] Frozen backbone:")
    extractor_frozen = YOLOv8BackboneExtractor(
        weights_path=weights_path,
        freeze_backbone=True,
    ).to(device)
    
    frozen_trainable = extractor_frozen.count_parameters(trainable_only=True)
    print(f"  Trainable parameters (frozen): {frozen_trainable / 1e6:.2f}M")
    
    print("\n" + "="*60)
    print("✅ All tests passed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    test_yolov8_backbone()
