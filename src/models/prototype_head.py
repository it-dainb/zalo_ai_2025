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
    ):
        super().__init__()
        
        self.nl = len(ch)
        self.proto_dims = proto_dims  # Store as list for scale-specific dims
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.stride = torch.tensor([4, 8, 16, 32])
        
        # Feature projection to prototype dimension (scale-specific)
        self.feature_proj = nn.ModuleList([
            nn.Sequential(
                Conv(c, pd, 1),
                nn.Conv2d(pd, pd, 1)
            ) for c, pd in zip(ch, proto_dims)
        ])
        
        # Box regression heads: Direct bbox prediction (4 coordinates)
        c2 = max((16, ch[0] // 4))
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c2, 3, padding=1),
                Conv(c2, c2, 3, padding=1),
                nn.Conv2d(c2, 4, 1)  # 4 bbox coordinates (direct prediction)
            ) for x in ch
        )
        
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
        Register gradient clipping hooks on feature_proj and cv2 layers.
        This prevents gradient explosion in cls_loss and bbox_loss backward pass.
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
        
        # CRITICAL: Also register hooks on cv2 box regression layers
        # These are prone to gradient explosion from WIoU loss
        for i, module in enumerate(self.cv2):
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
            
            # Box regression: l, t, r, b are DISTANCES (always positive)
            # Use ReLU to enforce positive values, then clamp to prevent explosion
            box_pred = self.cv2[i](x[i])
            box_pred = F.relu(box_pred)  # Force positive (distances can't be negative)
            box_pred = torch.clamp(box_pred, min=0.0, max=10.0)
            box_preds.append(box_pred)
        
        return box_preds, sim_scores
    
    def count_parameters(self, trainable_only: bool = False) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
