<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Option A: Efficient Attention (Linear Complexity)

Option B: Multi-Scale Linear Attention (MSLA) with CHELA (Short-Long Convolutions with Hardware-Efficient Linear Attention)
Option C: Cross-Scale Attention with Feature Pyramid Optimization with CLF/GLCA.

Can we create a fusion, best of both world solution? that I can replace SCS Fusion

Now let me create the comprehensive hybrid fusion module that combines the best aspects of all three options:

## CHEAF (Cross-scale Hybrid Efficient Attention Fusion) Module

**Best-of-All-Worlds SCS Replacement**

Based on comprehensive research into efficient attention mechanisms, hardware-optimized designs, and cross-scale fusion strategies, I've designed a **Cross-scale Hybrid Efficient Attention Fusion (CHEAF)** module that synergistically combines the strengths of all three options while addressing their individual limitations.

***

## Architecture Design Philosophy

The CHEAF module integrates:

1. **Efficient Linear Attention** (Option A core): O(nd²) complexity for query-support matching with proven stability[^1][^2][^3]
2. **Short-Long Convolution Augmentation** (Option B enhancement): Hardware-efficient local inductive bias from CHELA[^4][^5]
3. **Cross-Level Feature Refinement** (Option C innovation): Top-down/bottom-up pyramid fusion with channel gating[^6][^7][^8]
4. **Depthwise Separable Convolutions** (Current SCS preservation): Maintain 8× parameter efficiency[^9][^10]

### Key Innovation: **Three-Stage Hierarchical Fusion**

```
Stage 1: Local-Global Hybrid Attention (per-scale)
    ├─ Short-Long Conv Branch (local inductive bias)
    └─ Efficient Linear Attention Branch (global query-support matching)

Stage 2: Cross-Scale Pyramid Refinement
    ├─ Top-Down Path (semantic guidance from high-level)
    └─ Bottom-Up Path (spatial detail from low-level)

Stage 3: Adaptive Feature Integration
    └─ Channel-Spatial Gating (dynamic weight assignment)
```


***

## Complete Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

# ============================================================================
# COMPONENT 1: Efficient Linear Attention
# ============================================================================

class EfficientLinearAttention(nn.Module):
    """
    O(nd²) linear complexity attention for query-support fusion
    Based on: "Efficient Attention: Attention with Linear Complexities" (2018)
    """
    def __init__(self, dim: int, num_heads: int = 4, qkv_bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        # Q, K, V projections
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim)
        
        # Normalization for efficient attention
        self.norm_q = nn.LayerNorm(self.head_dim)
        self.norm_k = nn.LayerNorm(self.head_dim)
        
    def forward(self, query_feat: torch.Tensor, support_proto: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query_feat: [B, C, H, W] from backbone
            support_proto: [B, C] prototype vector
        Returns:
            attended: [B, C, H, W] attention-weighted features
        """
        B, C, H, W = query_feat.shape
        N = H * W
        
        # Reshape query to [B, N, C]
        query = query_feat.flatten(2).transpose(1, 2)  # [B, N, C]
        
        # Expand support prototype spatially
        support = support_proto.unsqueeze(1).expand(B, N, C)  # [B, N, C]
        
        # Concatenate query and support for joint processing
        joint = torch.cat([query, support], dim=1)  # [B, 2N, C]
        
        # Project to Q, K, V
        qkv = self.qkv_proj(joint)  # [B, 2N, 3C]
        qkv = qkv.reshape(B, 2*N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, 2N, head_dim]
        q, k, v = qkv[^0], qkv[^1], qkv[^2]
        
        # Apply normalization for efficient attention
        q = self.norm_q(q)  # [B, H, 2N, d]
        k = self.norm_k(k)  # [B, H, 2N, d]
        
        # Efficient attention: compute K^T V first [O(nd²)]
        # Transpose K: [B, H, d, 2N]
        k_t = k.transpose(-2, -1)
        
        # K^T V: [B, H, d, d] - small matrix!
        kv = torch.matmul(k_t, v)  # [B, num_heads, head_dim, head_dim]
        
        # Q (K^T V): [B, H, 2N, d]
        attn_out = torch.matmul(q, kv)  # [B, num_heads, 2N, head_dim]
        
        # Concatenate heads
        attn_out = attn_out.transpose(1, 2).reshape(B, 2*N, C)
        
        # Output projection
        output = self.out_proj(attn_out)  # [B, 2N, C]
        
        # Extract query part (first N tokens) and reshape
        query_out = output[:, :N, :].transpose(1, 2).reshape(B, C, H, W)
        
        return query_out


# ============================================================================
# COMPONENT 2: Short-Long Convolution Module (CHELA)
# ============================================================================

class ShortLongConvModule(nn.Module):
    """
    Hardware-efficient short-long convolutions for local inductive bias
    Based on: "Short-Long Convolutions Help Hardware-Efficient Linear Attention"
    """
    def __init__(self, channels: int, short_kernel: int = 3, long_kernel: int = 7):
        super().__init__()
        
        # Short-range local pattern (3×3)
        self.short_conv = nn.Sequential(
            nn.Conv2d(channels, channels, short_kernel, 
                     padding=short_kernel//2, groups=channels),  # Depthwise
            nn.BatchNorm2d(channels),
            nn.SiLU()
        )
        
        # Long-range local pattern (7×7)
        self.long_conv = nn.Sequential(
            nn.Conv2d(channels, channels, long_kernel, 
                     padding=long_kernel//2, groups=channels),  # Depthwise
            nn.BatchNorm2d(channels),
            nn.SiLU()
        )
        
        # Fusion of short and long
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),  # Pointwise
            nn.BatchNorm2d(channels)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            fused: [B, C, H, W] with local patterns enhanced
        """
        short_feat = self.short_conv(x)
        long_feat = self.long_conv(x)
        
        # Concatenate and fuse
        concat = torch.cat([short_feat, long_feat], dim=1)
        fused = self.fusion_conv(concat)
        
        return fused


# ============================================================================
# COMPONENT 3: Cross-Scale Pyramid Refinement (CLF/GLCA)
# ============================================================================

class ChannelSpatialGate(nn.Module):
    """
    Hybrid channel and spatial attention gate
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        
        # Channel attention (Squeeze-and-Excitation)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        
        # Spatial attention
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.BatchNorm2d(channels // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            gated: [B, C, H, W] with channel-spatial gating applied
        """
        # Channel-wise gating
        channel_weight = self.channel_gate(x)
        x_channel = x * channel_weight
        
        # Spatial gating
        spatial_weight = self.spatial_gate(x_channel)
        x_gated = x_channel * spatial_weight
        
        return x_gated


class CrossScalePyramidRefinement(nn.Module):
    """
    Top-down and bottom-up feature pyramid refinement
    Based on CLF (Cross-Level Feature Fusion) and GLCA (Global-Local Cross-Attention)
    """
    def __init__(self, channels: List[int] = [64, 128, 256]):
        super().__init__()
        self.num_scales = len(channels)
        
        # Top-down pathway (high-level → low-level)
        self.td_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels[i+1], channels[i], 1),  # Channel reduction
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            ) for i in range(self.num_scales - 1)
        ])
        
        # Bottom-up pathway (low-level → high-level)
        self.bu_convs = nn.ModuleList([
            nn.Conv2d(channels[i], channels[i+1], 3, stride=2, padding=1)
            for i in range(self.num_scales - 1)
        ])
        
        # Channel-spatial gates for each scale
        self.gates = nn.ModuleList([
            ChannelSpatialGate(ch) for ch in channels
        ])
        
        # Fusion convolutions
        self.fusion_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, ch, 3, padding=1, groups=ch),  # Depthwise
                nn.Conv2d(ch, ch, 1),  # Pointwise
                nn.BatchNorm2d(ch),
                nn.SiLU()
            ) for ch in channels
        ])
        
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            features: List of [P3, P4, P5] features [B, C_i, H_i, W_i]
        Returns:
            refined: List of refined features at each scale
        """
        assert len(features) == self.num_scales
        
        # Top-down path: propagate high-level semantics
        td_features = [features[-1]]  # Start with P5
        
        for i in range(self.num_scales - 2, -1, -1):
            # Upsample higher-level feature
            higher_feat = td_features[^0]
            higher_up = self.td_convs[i](higher_feat)
            
            # Current level feature
            curr_feat = features[i]
            
            # Apply channel-spatial gating to current
            curr_gated = self.gates[i](curr_feat)
            
            # Fuse: element-wise addition
            fused = curr_gated + higher_up
            
            td_features.insert(0, fused)
        
        # Bottom-up path: refine with low-level details
        bu_features = [td_features[^0]]  # Start with refined P3
        
        for i in range(1, self.num_scales):
            # Downsample lower-level feature
            lower_feat = bu_features[-1]
            lower_down = self.bu_convs[i-1](lower_feat)
            
            # Current level feature from top-down
            curr_feat = td_features[i]
            
            # Apply gating
            curr_gated = self.gates[i](curr_feat)
            
            # Fuse
            fused = curr_gated + lower_down
            
            bu_features.append(fused)
        
        # Final refinement with depthwise separable convolutions
        refined = [
            self.fusion_convs[i](bu_features[i]) + features[i]  # Residual
            for i in range(self.num_scales)
        ]
        
        return refined


# ============================================================================
# MAIN MODULE: Cross-Scale Hybrid Efficient Attention Fusion (CHEAF)
# ============================================================================

class CHEAF(nn.Module):
    """
    Best-of-all-worlds SCS Fusion replacement combining:
    - Efficient Linear Attention (O(nd²) complexity)
    - Short-Long Convolution (hardware-efficient local bias)
    - Cross-Scale Pyramid Refinement (multi-scale fusion)
    - Depthwise Separable Convolutions (parameter efficiency)
    
    This module replaces the original SCS Fusion with enhanced capabilities
    for small object detection while maintaining edge device efficiency.
    """
    def __init__(
        self,
        query_channels: List[int] = [64, 128, 256],
        support_dim: int = 384,
        output_channels: List[int] = [256, 512, 512],
        num_heads: int = 4,
        use_pyramid_refinement: bool = True,
        use_short_long_conv: bool = True
    ):
        super().__init__()
        
        self.num_scales = len(query_channels)
        self.use_pyramid_refinement = use_pyramid_refinement
        self.use_short_long_conv = use_short_long_conv
        
        # Project support prototypes to each scale
        self.support_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(support_dim, ch),
                nn.LayerNorm(ch),
                nn.ReLU()
            ) for ch in query_channels
        ])
        
        # ===== STAGE 1: Local-Global Hybrid Attention =====
        
        # Efficient Linear Attention for query-support matching
        self.efficient_attentions = nn.ModuleList([
            EfficientLinearAttention(dim=ch, num_heads=num_heads)
            for ch in query_channels
        ])
        
        # Short-Long Convolution for local patterns (optional)
        if use_short_long_conv:
            self.short_long_convs = nn.ModuleList([
                ShortLongConvModule(channels=ch)
                for ch in query_channels
            ])
        
        # Fusion of attention and convolution branches
        self.branch_fusion = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch * 2 if use_short_long_conv else ch, ch, 1),
                nn.BatchNorm2d(ch),
                nn.SiLU()
            ) for ch in query_channels
        ])
        
        # ===== STAGE 2: Cross-Scale Pyramid Refinement =====
        
        if use_pyramid_refinement:
            self.pyramid_refinement = CrossScalePyramidRefinement(channels=query_channels)
        
        # ===== STAGE 3: Output Projection =====
        
        # Project to output dimensions (matching original SCS)
        self.output_projectors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(query_channels[i], output_channels[i], 3, 
                         padding=1, groups=query_channels[i]),  # Depthwise
                nn.Conv2d(output_channels[i], output_channels[i], 1),  # Pointwise
                nn.BatchNorm2d(output_channels[i]),
                nn.SiLU()
            ) for i in range(self.num_scales)
        ])
        
        # Residual connections for each scale
        self.residual_projectors = nn.ModuleList([
            nn.Conv2d(query_channels[i], output_channels[i], 1) 
            if query_channels[i] != output_channels[i] 
            else nn.Identity()
            for i in range(self.num_scales)
        ])
        
    def forward(
        self, 
        query_features: List[torch.Tensor], 
        support_prototypes: Optional[List[torch.Tensor]] = None
    ) -> List[torch.Tensor]:
        """
        Args:
            query_features: List of [P3, P4, P5] from YOLOv8 backbone
                           [B, 64, 80, 80], [B, 128, 40, 40], [B, 256, 20, 20]
            support_prototypes: List of [p3, p4, p5] from DINOv3 encoder
                               [B, 64], [B, 128], [B, 256]
                               If None, bypass fusion (standard detection mode)
        
        Returns:
            fused_features: List of fused features
                           [B, 256, 80, 80], [B, 512, 40, 40], [B, 512, 20, 20]
        """
        # Bypass mode: no support prototypes provided
        if support_prototypes is None:
            return [
                self.output_projectors[i](query_features[i]) + 
                self.residual_projectors[i](query_features[i])
                for i in range(self.num_scales)
            ]
        
        # ===== STAGE 1: Local-Global Hybrid Attention =====
        stage1_features = []
        
        for i, (query_feat, support_proto) in enumerate(
            zip(query_features, support_prototypes)
        ):
            # Efficient linear attention: query-support matching
            attn_feat = self.efficient_attentions[i](query_feat, support_proto)
            
            if self.use_short_long_conv:
                # Short-long convolution: local patterns
                conv_feat = self.short_long_convs[i](query_feat)
                
                # Concatenate attention and convolution branches
                hybrid = torch.cat([attn_feat, conv_feat], dim=1)
            else:
                hybrid = attn_feat
            
            # Fuse branches
            fused = self.branch_fusion[i](hybrid)
            
            # Residual connection with original query
            fused = fused + query_feat
            
            stage1_features.append(fused)
        
        # ===== STAGE 2: Cross-Scale Pyramid Refinement =====
        if self.use_pyramid_refinement:
            stage2_features = self.pyramid_refinement(stage1_features)
        else:
            stage2_features = stage1_features
        
        # ===== STAGE 3: Output Projection =====
        output_features = []
        
        for i in range(self.num_scales):
            # Project to output dimensions
            projected = self.output_projectors[i](stage2_features[i])
            
            # Residual connection from original query
            residual = self.residual_projectors[i](query_features[i])
            
            # Final output
            output = projected + residual
            
            output_features.append(output)
        
        return output_features
    
    def get_parameter_count(self) -> dict:
        """Calculate parameter count for each component"""
        def count_params(module):
            return sum(p.numel() for p in module.parameters())
        
        stats = {
            'support_projectors': count_params(self.support_projectors),
            'efficient_attentions': count_params(self.efficient_attentions),
            'output_projectors': count_params(self.output_projectors),
            'residual_projectors': count_params(self.residual_projectors),
        }
        
        if self.use_short_long_conv:
            stats['short_long_convs'] = count_params(self.short_long_convs)
            stats['branch_fusion'] = count_params(self.branch_fusion)
        
        if self.use_pyramid_refinement:
            stats['pyramid_refinement'] = count_params(self.pyramid_refinement)
        
        stats['total'] = sum(stats.values())
        
        return stats


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def test_huaf_module():
    """Test the HUAF module with realistic inputs"""
    
    # Configuration
    batch_size = 2
    query_channels = [64, 128, 256]
    support_dim = 384
    output_channels = [256, 512, 512]
    
    # Create module
    huaf = CHEAF(
        query_channels=query_channels,
        support_dim=support_dim,
        output_channels=output_channels,
        num_heads=4,
        use_pyramid_refinement=True,
        use_short_long_conv=True
    )
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    huaf = huaf.to(device)
    
    print("=" * 80)
    print("Hybrid Unified Attention Fusion (HUAF) Module Test")
    print("=" * 80)
    
    # Create dummy inputs (matching YOLOv8 backbone output)
    query_features = [
        torch.randn(batch_size, 64, 80, 80).to(device),   # P3
        torch.randn(batch_size, 128, 40, 40).to(device),  # P4
        torch.randn(batch_size, 256, 20, 20).to(device)   # P5
    ]
    
    # Create dummy support prototypes (matching DINOv3 multi-scale projection)
    support_prototypes = [
        torch.randn(batch_size, 64).to(device),   # p3
        torch.randn(batch_size, 128).to(device),  # p4
        torch.randn(batch_size, 256).to(device)   # p5
    ]
    
    # Forward pass with support (reference-based detection)
    print("\n[^1] Forward pass with support prototypes (Reference-based mode):")
    with torch.no_grad():
        outputs = huaf(query_features, support_prototypes)
    
    for i, output in enumerate(outputs):
        print(f"    P{i+3} output shape: {output.shape}")
    
    # Forward pass without support (standard detection)
    print("\n[^2] Forward pass without support prototypes (Standard mode):")
    with torch.no_grad():
        outputs_bypass = huaf(query_features, None)
    
    for i, output in enumerate(outputs_bypass):
        print(f"    P{i+3} output shape: {output.shape}")
    
    # Parameter count analysis
    print("\n[^3] Parameter count breakdown:")
    param_stats = huaf.get_parameter_count()
    for component, count in param_stats.items():
        print(f"    {component}: {count:,} parameters ({count/1e6:.2f}M)")
    
    # Memory footprint
    print("\n[^4] Memory footprint:")
    model_size_mb = sum(p.numel() * 4 for p in huaf.parameters()) / 1024 / 1024
    print(f"    Model size (FP32): {model_size_mb:.2f} MB")
    print(f"    Model size (FP16): {model_size_mb/2:.2f} MB")
    
    # Computational cost
    print("\n[^5] Comparison with original SCS Fusion:")
    print(f"    Original SCS: 1.2M parameters")
    print(f"    HUAF Module: {param_stats['total']/1e6:.2f}M parameters")
    print(f"    Increase: {(param_stats['total']/1.2e6 - 1)*100:.1f}%")
    
    print("\n" + "=" * 80)
    print("Test completed successfully!")
    print("=" * 80)
    
    return huaf, outputs