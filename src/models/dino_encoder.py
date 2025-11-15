"""
DINOv3 Support Encoder
======================

Extracts reference features from support images using DINOv3 ViT-Small.
Produces class prototypes for reference-based detection.

Key Features:
- Uses timm's DINOv3 ViT-S/16 (pretrained on LVD-1689M)
- Extracts CLS token as global prototype
- Multi-scale projection for YOLOv8 feature alignment
- L2 normalization for cosine similarity matching
- ~21.6M parameters with rotary position embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
import timm


class DINOSupportEncoder(nn.Module):
    """
    DINOv3-based support encoder for reference image feature extraction.
    
    Args:
        model_name: DINOv3 model variant from timm (default: lvd1689m pretrained)
                   - "vit_small_patch16_dinov3.lvd1689m": 256×256, rotary embeddings
        output_dims: List of output dimensions matching YOLOv8 scales [P2, P3, P4, P5]
                    Default: [32, 64, 128, 256] to match YOLOv8n backbone channels
        freeze_backbone: Whether to freeze DINO weights
        freeze_layers: Number of initial transformer blocks to freeze (0-12)
    
    Architecture:
        Input: (B, 3, 256, 256) RGB images - MUST be normalized with ImageNet stats
               mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        Output: {
            'prototype': (B, 384) - Raw CLS token feature
            'p2': (B, 32) - Very small object scale (matches YOLOv8 P2) [NEW for UAV]
            'p3': (B, 64) - Small object scale (matches YOLOv8 P3)
            'p4': (B, 128) - Medium object scale (matches YOLOv8 P4)
            'p5': (B, 256) - Large object scale (matches YOLOv8 P5)
        }
    
    DINOv3 Features:
        - Rotary Position Embeddings (RoPE) instead of learned positional embeddings
        - Patch size 16×16
        - Trained on large dataset (LVD-1689M)
        - Better feature quality with similar parameter count
    
    Note:
        Use get_transforms() to obtain proper preprocessing pipeline.
        Normalization is CRITICAL - features will be incorrect without it.
    """
    
    def __init__(
        self,
        model_name: str = "vit_small_patch16_dinov3.lvd1689m",
        output_dims: Optional[List[int]] = None,  # [P3, P4, P5] channels matching YOLOv8
        freeze_backbone: bool = True,
        freeze_layers: int = 6,
        input_size: int = 256,
    ):
        super().__init__()
        
        # Default output dims match YOLOv8n backbone channels
        if output_dims is None:
            output_dims = [32, 64, 128, 256]  # [P2, P3, P4, P5]
        
        assert len(output_dims) == 4, "output_dims must be [P2, P3, P4, P5] dimensions"
        
        self.output_dims = output_dims
        self.freeze_backbone = freeze_backbone
        self.input_size = input_size
        
        # Load pretrained DINOv3 model from timm
        print(f"Loading DINO model: {model_name}")
        self.dino = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,  # Remove classification head
            img_size=input_size,
        )
        
        # Get feature dimension (384 for ViT-Small)
        self.feat_dim = self.dino.embed_dim
        
        # Freeze backbone if specified
        if freeze_backbone:
            self._freeze_backbone(freeze_layers)
        
        # Multi-scale projection heads matching YOLOv8 dimensions
        # P2: 384 -> 32 channels (very small objects) [NEW for UAV]
        self.proj_p2 = nn.Sequential(
            nn.Linear(self.feat_dim, output_dims[0]),
            nn.LayerNorm(output_dims[0]),
            nn.GELU(),
        )
        
        # P3: 384 -> 64 channels (small objects)
        self.proj_p3 = nn.Sequential(
            nn.Linear(self.feat_dim, output_dims[1]),
            nn.LayerNorm(output_dims[1]),
            nn.GELU(),
        )
        
        # P4: 384 -> 128 channels (medium objects)
        self.proj_p4 = nn.Sequential(
            nn.Linear(self.feat_dim, output_dims[2]),
            nn.LayerNorm(output_dims[2]),
            nn.GELU(),
        )
        
        # P5: 384 -> 256 channels (large objects)
        self.proj_p5 = nn.Sequential(
            nn.Linear(self.feat_dim, output_dims[3]),
            nn.LayerNorm(output_dims[3]),
            nn.GELU(),
        )
        
        # Triplet loss projection head (384 -> 256)
        # This learnable projection matches DINO features to YOLOv8 global feature dim
        # Better than fixed pooling: network learns optimal 256-dim representation
        self.triplet_proj = nn.Sequential(
            nn.Linear(self.feat_dim, 256),  # 384 -> 256
            nn.LayerNorm(256),
        )
        
        # Store data config for transforms
        self.data_config = timm.data.resolve_model_data_config(self.dino)
        
        print(f"DINO Support Encoder initialized:")
        print(f"  - Model: {model_name}")
        print(f"  - Feature dim: {self.feat_dim}")
        print(f"  - Input size: {input_size}×{input_size}")
        print(f"  - Output dims: P2={output_dims[0]}, P3={output_dims[1]}, P4={output_dims[2]}, P5={output_dims[3]} (matching YOLOv8)")
        print(f"  - Frozen layers: {freeze_layers if freeze_backbone else 0}")
        print(f"  - Total params: {self.count_parameters() / 1e6:.2f}M")
        print(f"  - Expected normalization: mean={self.data_config['mean']}, std={self.data_config['std']}")
        print(f"  ✓ Using DINOv3 with rotary embeddings and LVD-1689M pretraining")
    
    def get_transforms(self, is_training: bool = False):
        """
        Get model-specific preprocessing transforms.
        
        Args:
            is_training: Whether to use training-time augmentation
        
        Returns:
            torchvision transforms pipeline with proper normalization
        
        Example:
            >>> encoder = DINOSupportEncoder()
            >>> transforms = encoder.get_transforms()
            >>> img = Image.open('support.jpg')
            >>> img_tensor = transforms(img).unsqueeze(0)
            >>> features = encoder(img_tensor)
        """
        return timm.data.create_transform(**self.data_config, is_training=is_training)
    
    def _freeze_backbone(self, freeze_layers: int):
        """Freeze DINO backbone parameters."""
        # Freeze patch embedding
        for param in self.dino.patch_embed.parameters():
            param.requires_grad = False
        
        # Freeze CLS token
        if hasattr(self.dino, 'cls_token') and self.dino.cls_token is not None:
            self.dino.cls_token.requires_grad = False
        
        # Freeze rotary embeddings (DINOv3 uses RoPE instead of learned positional embeddings)
        if hasattr(self.dino, 'rope'):
            for param in self.dino.rope.parameters():
                param.requires_grad = False
        
        # Freeze specified number of transformer blocks
        if freeze_layers > 0:
            blocks = self.dino.blocks if hasattr(self.dino, 'blocks') else []
            for i, block in enumerate(blocks):
                if i < freeze_layers:
                    for param in block.parameters():
                        param.requires_grad = False
        
        print(f"Frozen first {freeze_layers} transformer blocks")
    
    def count_parameters(self, trainable_only: bool = False) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract CLS token features from DINO model.
        
        Args:
            x: Input tensor (B, 3, H, W) - MUST be normalized with ImageNet mean/std
               Expected: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        
        Returns:
            CLS token features (B, 384)
        
        Note:
            This is equivalent to:
                features = self.dinov2.forward_features(x)
                cls_token = self.dinov2.forward_head(features, pre_logits=True)
            But more explicit about using CLS token.
        """
        # Forward through DINO model
        features = self.dino.forward_features(x)
        
        # Extract CLS token (first token)
        # DINO output shape: (B, N_tokens, feat_dim)
        # For DINOv3 at 256x256: N_tokens = 257 (1 CLS + 256 patches from 16x16 grid)
        cls_token = features[:, 0]  # (B, feat_dim)
        
        return cls_token
    
    def forward(
        self, 
        support_images: torch.Tensor,
        normalize: bool = True,
        return_global_feat: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass to extract support prototypes.
        
        Args:
            support_images: Support/reference images (B, 3, 256, 256) for DINOv3
                          ⚠️  MUST be preprocessed with ImageNet normalization:
                          mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                          Use get_transforms() to obtain proper preprocessing.
            normalize: Whether to L2-normalize prototype features for cosine similarity
            return_global_feat: Whether to return raw CLS token for triplet loss (default: False)
        
        Returns:
            Dictionary containing:
                - 'prototype': (B, 384) raw CLS token
                - 'p2': (B, 32) projection for P2 scale (matches YOLOv8n) [NEW for UAV]
                - 'p3': (B, 64) projection for P3 scale (matches YOLOv8n)
                - 'p4': (B, 128) projection for P4 scale (matches YOLOv8n)
                - 'p5': (B, 256) projection for P5 scale (matches YOLOv8n)
                - 'global_feat': (B, 256) projected feature for triplet loss (if return_global_feat=True)
        
        Example:
            >>> encoder = DINOSupportEncoder()
            >>> transforms = encoder.get_transforms()
            >>> img = transforms(PIL.Image.open('support.jpg')).unsqueeze(0)
            >>> features = encoder(img)
        """
        # Extract CLS token features
        prototype = self.extract_features(support_images)  # (B, 384)
        
        # L2 normalization for cosine similarity
        if normalize:
            prototype = F.normalize(prototype, p=2, dim=-1)
        
        # Multi-scale projections matching YOLOv8 dimensions
        p2_feat = self.proj_p2(prototype)  # (B, 32)
        p3_feat = self.proj_p3(prototype)  # (B, 64)
        p4_feat = self.proj_p4(prototype)  # (B, 128)
        p5_feat = self.proj_p5(prototype)  # (B, 256)
        
        # Normalize projected features as well
        if normalize:
            p2_feat = F.normalize(p2_feat, p=2, dim=-1)
            p3_feat = F.normalize(p3_feat, p=2, dim=-1)
            p4_feat = F.normalize(p4_feat, p=2, dim=-1)
            p5_feat = F.normalize(p5_feat, p=2, dim=-1)
        
        output = {
            'prototype': prototype,
            'p2': p2_feat,
            'p3': p3_feat,
            'p4': p4_feat,
            'p5': p5_feat,
        }
        
        # Add global feature for triplet loss if requested
        # Use learnable projection to match YOLOv8 global feature dimension (256)
        if return_global_feat:
            # Project raw CLS token: 384 -> 256
            # This learnable transformation allows network to learn optimal feature compression
            global_feat = self.triplet_proj(prototype)  # (B, 256)
            output['global_feat'] = global_feat
        
        return output
    
    def compute_average_prototype(
        self,
        support_images: List[torch.Tensor],
        return_global_feat: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute average prototype from multiple support images.
        Useful for few-shot learning with K > 1 support images.
        
        Args:
            support_images: List of K support images, each (1, 3, 256, 256)
            return_global_feat: Whether to return projected global feature (256-dim) for triplet loss (default: False)
        
        Returns:
            Averaged prototypes for all scales, optionally including global_feat (256-dim)
        """
        all_prototypes = []
        
        with torch.no_grad():
            for img in support_images:
                proto = self.forward(img, normalize=True, return_global_feat=return_global_feat)
                all_prototypes.append(proto)
        
        # Average across all support images
        avg_prototype = {
            'prototype': torch.stack([p['prototype'] for p in all_prototypes]).mean(dim=0),
            'p2': torch.stack([p['p2'] for p in all_prototypes]).mean(dim=0),
            'p3': torch.stack([p['p3'] for p in all_prototypes]).mean(dim=0),
            'p4': torch.stack([p['p4'] for p in all_prototypes]).mean(dim=0),
            'p5': torch.stack([p['p5'] for p in all_prototypes]).mean(dim=0),
        }
        
        # Add averaged global feature if requested
        if return_global_feat:
            avg_prototype['global_feat'] = torch.stack([p['global_feat'] for p in all_prototypes]).mean(dim=0)
        
        # Re-normalize after averaging
        for key in avg_prototype:
            avg_prototype[key] = F.normalize(avg_prototype[key], p=2, dim=-1)
        
        return avg_prototype


def test_dinov2_encoder():
    """Quick sanity test for DINO encoder (DINOv3)."""
    print("\n" + "="*60)
    print("Testing DINO Support Encoder (DINOv3)")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Initialize encoder with DINOv3 and YOLOv8-matching dimensions
    encoder = DINOSupportEncoder(
        model_name="vit_small_patch16_dinov3.lvd1689m",
        output_dims=[32, 64, 128, 256],  # Match YOLOv8n backbone [P2, P3, P4, P5]
        freeze_backbone=True,
        freeze_layers=6,
        input_size=256,
    ).to(device)
    
    encoder.eval()
    
    # Test transforms
    print("\n[Test 0] Model transforms:")
    transforms = encoder.get_transforms()
    print(f"  Transforms: {transforms}")
    print(f"  Expected input: 256×256 RGB image (PIL or tensor)")
    print(f"  Normalization: mean={encoder.data_config['mean']}, std={encoder.data_config['std']}")
    
    # Test single image (simulating normalized input)
    print("\n[Test 1] Single support image:")
    # NOTE: In production, use encoder.get_transforms() to preprocess PIL images
    # Here we simulate normalized input
    dummy_img = torch.randn(1, 3, 256, 256).to(device)
    # Apply ImageNet normalization to simulate proper preprocessing
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    dummy_img = (dummy_img - mean) / std
    
    with torch.no_grad():
        output = encoder(dummy_img)
    
    print(f"  Input shape: {dummy_img.shape}")
    print(f"  Prototype shape: {output['prototype'].shape}")
    print(f"  P2 shape: {output['p2'].shape}")
    print(f"  P3 shape: {output['p3'].shape}")
    print(f"  P4 shape: {output['p4'].shape}")
    print(f"  P5 shape: {output['p5'].shape}")
    print(f"  Prototype L2 norm: {output['prototype'].norm(dim=-1).item():.3f} (should be ~1.0)")
    
    # Test batch
    print("\n[Test 2] Batch of support images:")
    dummy_batch = torch.randn(4, 3, 256, 256).to(device)
    dummy_batch = (dummy_batch - mean) / std  # Apply normalization
    
    with torch.no_grad():
        output_batch = encoder(dummy_batch)
    
    print(f"  Input shape: {dummy_batch.shape}")
    print(f"  Prototype shape: {output_batch['prototype'].shape}")
    print(f"  Average L2 norm: {output_batch['prototype'].norm(dim=-1).mean().item():.3f} (should be ~1.0)")
    
    # Test average prototype computation
    print("\n[Test 3] Average prototype from multiple images:")
    support_list = [(torch.randn(1, 3, 256, 256).to(device) - mean) / std for _ in range(3)]
    
    with torch.no_grad():
        avg_proto = encoder.compute_average_prototype(support_list)
    
    print(f"  Number of support images: {len(support_list)}")
    print(f"  Averaged prototype shape: {avg_proto['prototype'].shape}")
    print(f"  Averaged L2 norm: {avg_proto['prototype'].norm(dim=-1).item():.3f} (should be ~1.0)")
    
    # Parameter count
    print("\n[Parameter Count]")
    total_params = encoder.count_parameters()
    trainable_params = encoder.count_parameters(trainable_only=True)
    print(f"  Total parameters: {total_params / 1e6:.2f}M")
    print(f"  Trainable parameters: {trainable_params / 1e6:.2f}M")
    print(f"  Frozen parameters: {(total_params - trainable_params) / 1e6:.2f}M")
    
    print("\n" + "="*60)
    print("✅ All tests passed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    test_dinov2_encoder()
