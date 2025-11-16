"""
Dual Detection Head
===================

Combines standard YOLOv8 detection head with prototype-based matching head.
Enables detection of both base classes and novel reference-based classes.

Key Features:
- StandardDetectionHead: YOLOv8 native head for base classes
- PrototypeDetectionHead: Cosine similarity matching for novel classes
- DualDetectionHead: Combines both with NMS and score fusion
- Temperature-scaled similarity scores
- ~0.5M parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import torchvision


class Conv(nn.Module):
    """Standard convolution with BatchNorm and activation."""
    
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DFL(nn.Module):
    """
    Distribution Focal Loss (DFL) module.
    Used in YOLOv8 for box regression.
    """
    
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1
    
    def forward(self, x):
        b, c, a = x.shape  # batch, channels, anchors
        # Cast input to match conv weight dtype for mixed precision compatibility
        x_reshaped = x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)
        x_reshaped = x_reshaped.to(self.conv.weight.dtype)
        return self.conv(x_reshaped).view(b, 4, a)


class StandardDetectionHead(nn.Module):
    """
    Standard YOLOv8 detection head for base classes.
    Decoupled head with separate classification and box regression branches.
    
    Args:
        nc: Number of base classes
        ch: List of input channels for each scale [P2, P3, P4, P5]
        reg_max: Maximum value for distribution focal loss (default: 16)
    """
    
    def __init__(self, nc: int = 80, ch: List[int] = [128, 256, 512, 512], reg_max: int = 16):
        super().__init__()
        
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = reg_max  # DFL channels
        self.no = nc + reg_max * 4  # number of outputs per anchor
        self.stride = torch.tensor([4, 8, 16, 32])  # strides for [P2, P3, P4, P5]
        
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], self.nc)  # channels
        
        # Build detection heads for each scale
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c2, 3, padding=1),
                Conv(c2, c2, 3, padding=1),
                nn.Conv2d(c2, 4 * (self.reg_max + 1), 1)
            ) for x in ch
        )
        
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c3, 3, padding=1),
                Conv(c3, c3, 3, padding=1),
                nn.Conv2d(c3, self.nc, 1)
            ) for x in ch
        )
        
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
        
        # Initialize weights for stability
        self._initialize_weights()
        
        print(f"Standard Detection Head initialized:")
        print(f"  - Number of classes: {nc}")
        print(f"  - Detection layers: {self.nl}")
        print(f"  - Input channels: {ch}")
    
    def _initialize_weights(self):
        """Initialize weights for detection head to prevent gradient explosion."""
        for module_list in [self.cv2, self.cv3]:
            for module in module_list:
                for m in module.modules():
                    if isinstance(m, nn.Conv2d):
                        # Use smaller std for final conv layers
                        nn.init.normal_(m.weight, mean=0.0, std=0.01)
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
                    elif isinstance(m, nn.BatchNorm2d):
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)
    
    def forward(self, x: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through standard detection head.
        
        Args:
            x: List of feature tensors [P2, P3, P4, P5]
        
        Returns:
            Tuple of (bbox_predictions, class_predictions)
        """
        box_preds = []
        cls_preds = []
        
        for i in range(self.nl):
            # Box regression
            box_preds.append(self.cv2[i](x[i]))
            # Classification
            cls_preds.append(self.cv3[i](x[i]))
        
        return box_preds, cls_preds


class PrototypeDetectionHead(nn.Module):
    """
    Prototype-based detection head for novel/reference classes.
    Uses cosine similarity matching between features and prototypes.
    
    Args:
        ch: List of input channels for each scale [P2, P3, P4, P5]
        proto_dims: Scale-specific dimensions of prototype features [P2, P3, P4, P5]
        temperature: Temperature scaling factor for similarity scores
    """
    
    def __init__(
        self, 
        ch: List[int] = [128, 256, 512, 512],
        proto_dims: List[int] = [32, 64, 128, 256],  # Scale-specific prototype dimensions
        temperature: float = 10.0,
        reg_max: int = 16,
    ):
        super().__init__()
        
        self.nl = len(ch)
        self.proto_dims = proto_dims  # Store as list for scale-specific dims
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.reg_max = reg_max
        self.stride = torch.tensor([4, 8, 16, 32])
        
        # Feature projection to prototype dimension (scale-specific)
        self.feature_proj = nn.ModuleList([
            nn.Sequential(
                Conv(c, pd, 1),
                nn.Conv2d(pd, pd, 1)
            ) for c, pd in zip(ch, proto_dims)
        ])
        
        # Box regression heads (same as standard head)
        c2 = max((16, ch[0] // 4, self.reg_max * 4))
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c2, 3, padding=1),
                Conv(c2, c2, 3, padding=1),
                nn.Conv2d(c2, 4 * (self.reg_max + 1), 1)
            ) for x in ch
        )
        
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
        
        # Initialize weights for box regression head (critical for stability)
        self._initialize_weights()
        
        # Add gradient clipping hooks to feature_proj layers to prevent NaN gradients
        self._register_gradient_hooks()
        
        # Determine scales dynamically based on number of detection layers
        # If nl < 4, use last nl scales (e.g., 3 layers -> ['p3', 'p4', 'p5'])
        all_scales = ['p2', 'p3', 'p4', 'p5']
        self.scales = all_scales[-self.nl:]
        
        print(f"Prototype Detection Head initialized:")
        print(f"  - Detection layers: {self.nl}")
        print(f"  - Scales: {self.scales}")
        print(f"  - Prototype dims: {proto_dims}")
        print(f"  - Temperature: {temperature}")
    
    def _initialize_weights(self):
        """Initialize weights for box regression head to prevent gradient explosion."""
        for module_list in [self.cv2, self.feature_proj]:
            for module in module_list:
                for m in module.modules():
                    if isinstance(m, nn.Conv2d):
                        # Use smaller std for final conv layers
                        nn.init.normal_(m.weight, mean=0.0, std=0.01)
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
                    elif isinstance(m, nn.BatchNorm2d):
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)
    
    def _register_gradient_hooks(self):
        """
        Register gradient clipping hooks on feature_proj layers.
        This prevents gradient explosion in cls_loss backward pass.
        """
        def gradient_clip_hook(grad):
            """Clamp gradients to [-10, 10] range."""
            if grad is not None:
                return torch.clamp(grad, min=-10.0, max=10.0)
            return grad
        
        # Register hooks on all Conv2d and BatchNorm2d layers in feature_proj
        for i, module in enumerate(self.feature_proj):
            for m in module.modules():
                if isinstance(m, (nn.Conv2d, nn.BatchNorm2d)):
                    if hasattr(m, 'weight') and m.weight is not None:
                        m.weight.register_hook(gradient_clip_hook)
                    if hasattr(m, 'bias') and m.bias is not None:
                        m.bias.register_hook(gradient_clip_hook)
    
    def compute_similarity(
        self, 
        features: torch.Tensor, 
        prototypes: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cosine similarity between features and prototypes.
        
        Args:
            features: (B, C, H, W) feature maps
            prototypes: (K, C) prototype vectors for K classes
        
        Returns:
            Similarity scores (B, K, H, W)
        """
        # Normalize features
        features = F.normalize(features, p=2, dim=1)  # (B, C, H, W)
        
        # Normalize prototypes
        prototypes = F.normalize(prototypes, p=2, dim=1)  # (K, C)
        
        # Compute cosine similarity
        # Reshape features: (B, C, H, W) -> (B, C, H*W)
        b, c, h, w = features.shape
        features_flat = features.reshape(b, c, -1)  # (B, C, H*W)
        
        # Matrix multiplication with broadcasting: (K, C) @ (B, C, H*W) -> (B, K, H*W)
        # PyTorch broadcasts (K, C) to (1, K, C) then to (B, K, C)
        similarity = torch.matmul(prototypes, features_flat)  # (B, K, H*W)
        
        # Reshape back: (B, K, H*W) -> (B, K, H, W)
        similarity = similarity.reshape(b, -1, h, w)
        
        # Temperature scaling
        similarity = similarity * self.temperature
        
        return similarity
    
    def forward(
        self, 
        x: List[torch.Tensor],
        prototypes: Dict[str, torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass through prototype detection head.
        
        Args:
            x: List of feature tensors [P2, P3, P4, P5]
            prototypes: Dict with keys ['p2', 'p3', 'p4', 'p5']
                       Can be either:
                       - (K, proto_dim): K prototype classes (standard usage)
                       - (B, C): Per-batch prototypes where C can vary (single reference per sample)
        
        Returns:
            Tuple of (bbox_predictions, similarity_scores)
        """
        box_preds = []
        sim_scores = []
        
        for i in range(self.nl):
            # Project features to prototype dimension
            feat_proj = self.feature_proj[i](x[i])  # (B, proto_dim, H, W)
            
            # CRITICAL: Clamp feature projections to prevent gradient explosion
            # These go into similarity computation which affects cls_loss gradients
            feat_proj = torch.clamp(feat_proj, min=-10.0, max=10.0)
            
            # Compute similarity with prototypes
            scale_key = self.scales[i]
            if scale_key in prototypes and prototypes[scale_key] is not None:
                proto = prototypes[scale_key]  # (K, C) or (B, C)
                
                # Prototypes should already be projected to proto_dims[i] by the encoder
                # Check scale-specific dimension match
                expected_dim = self.proto_dims[i]
                if proto.shape[-1] != expected_dim:
                    raise ValueError(
                        f"Prototype dimension mismatch at {scale_key}: "
                        f"expected {expected_dim}, got {proto.shape[-1]}. "
                        f"Prototypes should be projected to proto_dims by the support encoder."
                    )
                
                # Compute cosine similarity
                # If proto has same batch size as features, it's per-sample prototypes
                # Otherwise, it's K class prototypes
                B = feat_proj.shape[0]
                K = proto.shape[0]
                
                if K == B and B > 1:
                    # Per-sample prototypes: compute similarity and take diagonal
                    sim = self.compute_similarity(feat_proj, proto)  # (B, B, H, W)
                    # Extract diagonal: each sample's similarity with its own prototype
                    sim = torch.stack([sim[i, i:i+1] for i in range(B)], dim=0)  # (B, 1, H, W)
                else:
                    # K class prototypes: standard similarity computation
                    sim = self.compute_similarity(feat_proj, proto)  # (B, K, H, W)
                
                sim_scores.append(sim)
            else:
                # No prototypes for this scale, return zeros
                sim_scores.append(torch.zeros_like(feat_proj[:, :1]))
            
            # Box regression (shared with features)
            box_preds.append(self.cv2[i](x[i]))
        
        return box_preds, sim_scores


class DualDetectionHead(nn.Module):
    """
    Dual detection head combining standard and prototype-based detection.
    
    Args:
        nc_base: Number of base classes (standard head)
        ch: List of input channels [P2, P3, P4, P5]
        proto_dims: Scale-specific prototype dimensions [P2, P3, P4, P5]
        temperature: Temperature for similarity scaling
        conf_thres: Confidence threshold for detections
        iou_thres: IoU threshold for NMS
    """
    
    def __init__(
        self,
        nc_base: int = 80,
        ch: List[int] = [128, 256, 512, 512],
        proto_dims: List[int] = [32, 64, 128, 256],  # Scale-specific prototype dimensions
        temperature: float = 10.0,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        reg_max: int = 16,
    ):
        super().__init__()
        
        self.nc_base = nc_base
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        # Standard head for base classes
        self.standard_head = StandardDetectionHead(nc=nc_base, ch=ch, reg_max=reg_max)
        
        # Prototype head for novel classes
        self.prototype_head = PrototypeDetectionHead(
            ch=ch, 
            proto_dims=proto_dims, 
            temperature=temperature,
            reg_max=reg_max
        )
        
        print(f"Dual Detection Head initialized:")
        print(f"  - Base classes: {nc_base}")
        print(f"  - Prototype dims: {proto_dims}")
        print(f"  - Total params: {self.count_parameters() / 1e6:.2f}M")
    
    def count_parameters(self, trainable_only: bool = False) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def forward(
        self,
        features: Dict[str, torch.Tensor],
        prototypes: Optional[Dict[str, torch.Tensor]] = None,
        mode: str = 'dual',
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through dual head.
        
        Args:
            features: Dict with keys ['p2', 'p3', 'p4', 'p5']
            prototypes: Dict with prototype features for each scale (optional)
            mode: 'standard', 'prototype', or 'dual' (default)
        
        Returns:
            Dictionary with detection outputs
        """
        # Convert dict to list for compatibility with heads
        # Use dynamic scales from prototype_head (single source of truth)
        feat_list = [features[scale] for scale in self.prototype_head.scales]
        
        outputs = {}
        
        # Standard head (base classes) - only if nc_base > 0
        if mode in ['standard', 'dual'] and self.nc_base > 0:
            std_boxes, std_cls = self.standard_head(feat_list)
            outputs['standard_boxes'] = std_boxes
            outputs['standard_cls'] = std_cls
        
        # Prototype head (novel classes)
        if mode in ['prototype', 'dual'] and prototypes is not None:
            proto_boxes, proto_sim = self.prototype_head(feat_list, prototypes)
            outputs['prototype_boxes'] = proto_boxes
            outputs['prototype_sim'] = proto_sim
        
        return outputs


def test_dual_head():
    """Quick sanity test for dual detection head."""
    print("\n" + "="*60)
    print("Testing Dual Detection Head")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Initialize dual head
    dual_head = DualDetectionHead(
        nc_base=80,
        ch=[128, 256, 512, 512],
        proto_dims=[32, 64, 128, 256],  # Scale-specific prototype dimensions
        temperature=10.0,
    ).to(device)
    
    dual_head.eval()
    
    # Mock features (from CHEAF fusion) - 4 scales [P2, P3, P4, P5]
    features = {
        'p2': torch.randn(1, 128, 160, 160).to(device),
        'p3': torch.randn(1, 256, 80, 80).to(device),
        'p4': torch.randn(1, 512, 40, 40).to(device),
        'p5': torch.randn(1, 512, 20, 20).to(device),
    }
    
    # Mock prototypes (3 novel classes) - scale-specific dimensions
    prototypes = {
        'p2': torch.randn(3, 32).to(device),   # P2: 32 dims (NEW for UAV small objects)
        'p3': torch.randn(3, 64).to(device),   # P3: 64 dims
        'p4': torch.randn(3, 128).to(device),  # P4: 128 dims
        'p5': torch.randn(3, 256).to(device),  # P5: 256 dims
    }
    
    print("[Test 1] Dual mode (standard + prototype):")
    with torch.no_grad():
        outputs = dual_head(features, prototypes, mode='dual')
    
    print("  Standard head outputs:")
    print(f"    Boxes: {len(outputs['standard_boxes'])} scales")
    for i, boxes in enumerate(outputs['standard_boxes']):
        print(f"      Scale {i}: {boxes.shape}")
    print(f"    Classes: {len(outputs['standard_cls'])} scales")
    for i, cls in enumerate(outputs['standard_cls']):
        print(f"      Scale {i}: {cls.shape}")
    
    print("  Prototype head outputs:")
    print(f"    Boxes: {len(outputs['prototype_boxes'])} scales")
    print(f"    Similarities: {len(outputs['prototype_sim'])} scales")
    for i, sim in enumerate(outputs['prototype_sim']):
        print(f"      Scale {i}: {sim.shape}")
    
    print("\n[Test 2] Standard mode only:")
    with torch.no_grad():
        outputs_std = dual_head(features, mode='standard')
    
    print(f"  Has standard outputs: {'standard_boxes' in outputs_std}")
    print(f"  Has prototype outputs: {'prototype_boxes' in outputs_std}")
    
    print("\n[Test 3] Prototype mode only:")
    with torch.no_grad():
        outputs_proto = dual_head(features, prototypes, mode='prototype')
    
    print(f"  Has standard outputs: {'standard_boxes' in outputs_proto}")
    print(f"  Has prototype outputs: {'prototype_boxes' in outputs_proto}")
    
    print("\n[Test 4] Batch processing:")
    batch_features = {
        'p2': torch.randn(4, 128, 160, 160).to(device),
        'p3': torch.randn(4, 256, 80, 80).to(device),
        'p4': torch.randn(4, 512, 40, 40).to(device),
        'p5': torch.randn(4, 512, 20, 20).to(device),
    }
    
    with torch.no_grad():
        outputs_batch = dual_head(batch_features, prototypes, mode='dual')
    
    print(f"  Batch size: 4")
    print(f"  Standard boxes[0] shape: {outputs_batch['standard_boxes'][0].shape}")
    print(f"  Prototype sim[0] shape: {outputs_batch['prototype_sim'][0].shape}")
    
    # Parameter count
    print("\n[Parameter Count]")
    total_params = dual_head.count_parameters()
    std_params = dual_head.standard_head.parameters()
    proto_params = dual_head.prototype_head.parameters()
    
    std_count = sum(p.numel() for p in std_params)
    proto_count = sum(p.numel() for p in proto_params)
    
    print(f"  Total parameters: {total_params / 1e6:.2f}M")
    print(f"  Standard head: {std_count / 1e6:.2f}M")
    print(f"  Prototype head: {proto_count / 1e6:.2f}M")
    
    print("\n" + "="*60)
    print("✅ All tests passed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    test_dual_head()
