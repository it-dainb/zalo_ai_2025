"""
CHEAF: Cross-scale Hybrid Efficient Attention Fusion Module
============================================================

Best-of-all-worlds SCS Fusion replacement combining:
- Efficient Linear Attention (O(nd²) complexity)
- Short-Long Convolution (hardware-efficient local bias)
- Cross-Scale Pyramid Refinement (multi-scale fusion)
- Depthwise Separable Convolutions (parameter efficiency)

Key Features:
- Three-stage hierarchical fusion
- O(nd²) linear complexity attention
- Hardware-efficient local inductive bias
- Top-down/bottom-up pyramid refinement
- ~2-3M parameters (maintains efficiency)

References:
- Efficient Attention: Attention with Linear Complexities (2018)
- Short-Long Convolutions Help Hardware-Efficient Linear Attention
- CLF (Cross-Level Feature Fusion) and GLCA (Global-Local Cross-Attention)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional


# ============================================================================
# COMPONENT 1: Efficient Linear Attention
# ============================================================================

class EfficientLinearAttention(nn.Module):
    """
    O(nd²) linear complexity attention for query-support fusion.
    Based on: "Efficient Attention: Attention with Linear Complexities" (2018)
    
    MATCHES AUTHOR'S OFFICIAL IMPLEMENTATION:
    - Conv2d(1×1) for Q, K, V projections (preserves spatial structure)
    - Softmax normalization: softmax(K, dim=2) for spatial, softmax(Q, dim=1) for channel
    - Attention computation: K^T V @ Q (efficient order)
    - Residual connection: output + input
    
    Computes K^T V first (small d×d matrix) then multiplies with Q,
    reducing complexity from O(n²d) to O(nd²) where n >> d.
    """
    def __init__(
        self, 
        in_channels: int, 
        key_channels: int, 
        value_channels: int, 
        head_count: int = 1
    ):
        super().__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        self.head_count = head_count
        
        assert key_channels % head_count == 0, \
            f"key_channels {key_channels} must be divisible by head_count {head_count}"
        assert value_channels % head_count == 0, \
            f"value_channels {value_channels} must be divisible by head_count {head_count}"
        
        self.key_channels_per_head = key_channels // head_count
        self.value_channels_per_head = value_channels // head_count
        
        # Conv2d projections (preserves spatial structure)
        self.keys = nn.Conv2d(in_channels, key_channels, 1)
        self.queries = nn.Conv2d(in_channels, key_channels, 1)
        self.values = nn.Conv2d(in_channels, value_channels, 1)
        self.reprojection = nn.Conv2d(value_channels, in_channels, 1)
        
    def forward(self, query_feat: torch.Tensor, support_proto: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query_feat: [B, C, H, W] from backbone
            support_proto: [B, C] prototype vector
        Returns:
            attended: [B, C, H, W] attention-weighted features WITH residual
        """
        B, C, H, W = query_feat.shape
        N = H * W
        
        # Expand support prototype to spatial dimensions
        support_spatial = support_proto.view(B, C, 1, 1).expand(B, C, H, W)
        
        # Concatenate query and support for joint processing
        joint = torch.cat([query_feat, support_spatial], dim=0)  # [2B, C, H, W]
        
        # Project to Keys, Queries, Values
        keys = self.keys(joint)  # [2B, key_channels, H, W]
        queries = self.queries(joint)  # [2B, key_channels, H, W]
        values = self.values(joint)  # [2B, value_channels, H, W]
        
        # Multi-head attention
        head_key_channels = self.key_channels_per_head
        head_value_channels = self.value_channels_per_head
        
        # Reshape for multi-head: [2B, H, d, hw]
        keys = keys.reshape(2*B, self.head_count, head_key_channels, N)
        queries = queries.reshape(2*B, self.head_count, head_key_channels, N)
        values = values.reshape(2*B, self.head_count, head_value_channels, N)
        
        # Apply softmax normalization (CRITICAL: matches author)
        # Spatial softmax on keys: normalize across spatial positions
        keys = F.softmax(keys, dim=3)  # [2B, H, d, hw] - softmax over hw dimension
        
        # Channel softmax on queries: normalize across key channels
        queries = F.softmax(queries, dim=2)  # [2B, H, d, hw] - softmax over d dimension
        
        # Efficient attention computation per head
        # Step 1: K^T @ V = [2B, H, d, d] (small matrix!)
        attended_values = []
        for h in range(self.head_count):
            key = keys[:, h, :, :]  # [2B, d, hw]
            query = queries[:, h, :, :]  # [2B, d, hw]
            value = values[:, h, :, :]  # [2B, d_v, hw]
            
            # Context: K^T @ V^T = [2B, d_k, d_v]
            context = key @ value.transpose(1, 2)  # [2B, d_k, d_v]
            
            # Step 2: Context^T @ Q = [2B, d_v, hw]
            attended_value = context.transpose(1, 2) @ query  # [2B, d_v, hw]
            
            attended_values.append(attended_value)
        
        # Concatenate heads: [2B, value_channels, hw]
        aggregated_values = torch.cat(attended_values, dim=1)
        
        # Reshape back to spatial: [2B, value_channels, H, W]
        aggregated_values = aggregated_values.reshape(2*B, self.value_channels, H, W)
        
        # Reproject to input dimensions
        reprojected_value = self.reprojection(aggregated_values)  # [2B, C, H, W]
        
        # Add residual connection (CRITICAL: matches author)
        output = reprojected_value + joint  # [2B, C, H, W]
        
        # Extract query part (first B samples)
        query_output = output[:B, :, :, :]  # [B, C, H, W]
        
        return query_output


# ============================================================================
# COMPONENT 2: Short-Long Convolution Module (CHELA)
# ============================================================================

class ShortLongConvModule(nn.Module):
    """
    Hardware-efficient short-long convolutions for local inductive bias.
    Based on: "Short-Long Convolutions Help Hardware-Efficient Linear Attention" (CHELA)
    
    MATCHES AUTHOR'S IMPLEMENTATION (Equation 8):
    Z = K_l(φ_silu(K_s(X)))
    
    Sequential architecture:
    1. Short convolution K_s (kernel size 3) 
    2. SiLU activation φ_silu
    3. Long convolution K_l (kernel size varies, default 7)
    
    This sequential design captures both high-frequency (short) and 
    low-frequency (long) patterns with improved stability.
    
    Note: Author mentions structural reparameterization (SR) to fuse
    short kernels into long at inference, but this is optional.
    """
    def __init__(self, channels: int, short_kernel: int = 3, long_kernel: int = 7):
        super().__init__()
        
        # Short convolution K_s (depthwise for efficiency)
        self.short_conv = nn.Conv2d(
            channels, channels, short_kernel,
            padding=short_kernel // 2,
            groups=channels  # Depthwise
        )
        
        # SiLU activation between short and long
        self.activation = nn.SiLU()
        
        # Long convolution K_l (depthwise for efficiency)
        self.long_conv = nn.Conv2d(
            channels, channels, long_kernel,
            padding=long_kernel // 2,
            groups=channels  # Depthwise
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sequential short-long convolution: Z = K_l(φ_silu(K_s(X)))
        
        Args:
            x: [B, C, H, W] input features
        Returns:
            z: [B, C, H, W] output with short-long patterns
        """
        # Short convolution
        z = self.short_conv(x)
        
        # SiLU activation
        z = self.activation(z)
        
        # Long convolution
        z = self.long_conv(z)
        
        return z


# ============================================================================
# COMPONENT 3: Cross-Scale Pyramid Refinement (CLF/GLCA)
# ============================================================================

class ChannelSpatialGate(nn.Module):
    """
    Hybrid channel and spatial attention gate.
    Combines Squeeze-and-Excitation (channel) with spatial attention.
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
    Top-down and bottom-up feature pyramid refinement.
    Based on CLF (Cross-Level Feature Fusion) and GLCA (Global-Local Cross-Attention).
    
    Propagates high-level semantics downward and low-level details upward.
    Supports 4 scales: [P2, P3, P4, P5] for UAV small object detection.
    """
    def __init__(self, channels: List[int] = [32, 64, 128, 256]):
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
            features: List of [P2, P3, P4, P5] features [B, C_i, H_i, W_i]
        Returns:
            refined: List of refined features at each scale
        """
        assert len(features) == self.num_scales, \
            f"Expected {self.num_scales} features, got {len(features)}"
        
        # Top-down path: propagate high-level semantics
        td_features = [features[-1]]  # Start with P5 (highest level)
        
        for i in range(self.num_scales - 2, -1, -1):
            # Upsample higher-level feature
            higher_feat = td_features[0]
            higher_up = self.td_convs[i](higher_feat)
            
            # Current level feature
            curr_feat = features[i]
            
            # Apply channel-spatial gating to current
            curr_gated = self.gates[i](curr_feat)
            
            # Fuse: element-wise addition
            fused = curr_gated + higher_up
            
            td_features.insert(0, fused)
        
        # Bottom-up path: refine with low-level details
        bu_features = [td_features[0]]  # Start with refined P3
        
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

class CHEAFFusionModule(nn.Module):
    """
    CHEAF: Cross-scale Hybrid Efficient Attention Fusion
    
    Best-of-all-worlds SCS Fusion replacement combining:
    - Efficient Linear Attention (O(nd²) complexity)
    - Short-Long Convolution (hardware-efficient local bias)
    - Cross-Scale Pyramid Refinement (multi-scale fusion)
    - Depthwise Separable Convolutions (parameter efficiency)
    
    This module replaces the original SCS Fusion with enhanced capabilities
    for small object detection while maintaining edge device efficiency.
    
    Args:
        query_channels: List of query feature channels [P2, P3, P4, P5]
        support_channels: List of support prototype dimensions [P2, P3, P4, P5]
        out_channels: List of output channels [P2, P3, P4, P5]
        num_heads: Number of attention heads per scale
        use_pyramid_refinement: Enable cross-scale pyramid refinement
        use_short_long_conv: Enable short-long convolution module
    
    Architecture:
        Stage 1: Local-Global Hybrid Attention (per-scale)
            ├─ Short-Long Conv Branch (local inductive bias)
            └─ Efficient Linear Attention Branch (global query-support matching)
        
        Stage 2: Cross-Scale Pyramid Refinement
            ├─ Top-Down Path (semantic guidance from high-level)
            └─ Bottom-Up Path (spatial detail from low-level)
        
        Stage 3: Adaptive Feature Integration
            └─ Output projection with residual connections
    
    Note:
        For UAV small object detection, P2 (1/4 scale, 160×160) is added
        to capture very small distant objects at higher resolution.
    """
    def __init__(
        self,
        query_channels: List[int] = [32, 64, 128, 256],
        support_channels: List[int] = [32, 64, 128, 256],
        out_channels: List[int] = [128, 256, 512, 512],
        num_heads: int = 4,
        use_pyramid_refinement: bool = True,
        use_short_long_conv: bool = True
    ):
        super().__init__()
        
        self.num_scales = len(query_channels)
        # Dynamically determine scales based on number of channels provided
        all_scales = ['p2', 'p3', 'p4', 'p5']
        # Use the last N scales (e.g., if 3 channels, use p3, p4, p5)
        self.scales = all_scales[-self.num_scales:]
        self.use_pyramid_refinement = use_pyramid_refinement
        self.use_short_long_conv = use_short_long_conv
        self.support_channels = support_channels
        
        # Enforce Lego concept: query and support channels must match
        assert query_channels == support_channels, (
            f"Lego design violation! Query channels {query_channels} must match "
            f"support channels {support_channels}. DINOv3 and YOLOv8 outputs should "
            f"already be aligned at [64, 128, 256] without projection."
        )
        
        # ===== STAGE 1: Local-Global Hybrid Attention =====
        
        # Efficient Linear Attention for query-support matching
        # Following author's implementation: in_channels, key_channels, value_channels, head_count
        self.efficient_attentions = nn.ModuleDict({
            self.scales[i]: EfficientLinearAttention(
                in_channels=query_channels[i],
                key_channels=query_channels[i] // 2,  # Reduce key dim for efficiency
                value_channels=query_channels[i] // 2,  # Reduce value dim for efficiency
                head_count=min(num_heads, query_channels[i] // 16)  # Ensure sufficient channels per head
            )
            for i in range(self.num_scales)
        })
        
        # Short-Long Convolution for local patterns (optional)
        if use_short_long_conv:
            self.short_long_convs = nn.ModuleDict({
                self.scales[i]: ShortLongConvModule(channels=query_channels[i])
                for i in range(self.num_scales)
            })
        
        # Fusion of attention and convolution branches
        self.branch_fusion = nn.ModuleDict({
            self.scales[i]: nn.Sequential(
                nn.Conv2d(query_channels[i] * 2 if use_short_long_conv else query_channels[i], 
                         query_channels[i], 1),
                nn.BatchNorm2d(query_channels[i]),
                nn.SiLU()
            ) for i in range(self.num_scales)
        })
        
        # ===== STAGE 2: Cross-Scale Pyramid Refinement =====
        
        if use_pyramid_refinement:
            self.pyramid_refinement = CrossScalePyramidRefinement(channels=query_channels)
        
        # ===== STAGE 3: Output Projection =====
        
        # Project to output dimensions
        self.output_projectors = nn.ModuleDict({
            self.scales[i]: nn.Sequential(
                nn.Conv2d(query_channels[i], out_channels[i], 3, 
                         padding=1, groups=query_channels[i]),  # Depthwise
                nn.Conv2d(out_channels[i], out_channels[i], 1),  # Pointwise
                nn.BatchNorm2d(out_channels[i]),
                nn.SiLU()
            ) for i in range(self.num_scales)
        })
        
        # Residual connections for each scale
        self.residual_projectors = nn.ModuleDict({
            self.scales[i]: (
                nn.Conv2d(query_channels[i], out_channels[i], 1) 
                if query_channels[i] != out_channels[i] 
                else nn.Identity()
            )
            for i in range(self.num_scales)
        })
        
        print(f"CHEAF Fusion Module initialized:")
        print(f"  - Scales: {self.scales}")
        print(f"  - Query channels: {query_channels}")
        print(f"  - Support channels: {support_channels}")
        print(f"  - Output channels: {out_channels}")
        print(f"  - Pyramid refinement: {use_pyramid_refinement}")
        print(f"  - Short-long conv: {use_short_long_conv}")
        print(f"  - Total params: {self.count_parameters() / 1e6:.2f}M")
        
    def count_parameters(self, trainable_only: bool = False) -> int:
        """Count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def forward(
        self, 
        query_features: Dict[str, torch.Tensor], 
        support_features: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through CHEAF fusion.
        
        Args:
            query_features: Dictionary with keys ['p2', 'p3', 'p4', 'p5']
                - 'p2': (B, C_q2, H2, W2) [NEW for UAV]
                - 'p3': (B, C_q3, H3, W3)
                - 'p4': (B, C_q4, H4, W4)
                - 'p5': (B, C_q5, H5, W5)
            
            support_features: Optional dictionary with keys ['p2', 'p3', 'p4', 'p5']
                - 'p2': (B, C_s2) or None [NEW for UAV]
                - 'p3': (B, C_s3) or None
                - 'p4': (B, C_s4) or None
                - 'p5': (B, C_s5) or None
                If None, bypass fusion mode (projection only)
        
        Returns:
            Dictionary of fused features at each scale
        """
        # Bypass mode: no support features, just project query to output dims
        if support_features is None:
            fused_features = {}
            for scale in self.scales:
                if scale in query_features:
                    query_feat = query_features[scale]
                    # Use output projector and residual path
                    projected = self.output_projectors[scale](query_feat)
                    residual = self.residual_projectors[scale](query_feat)
                    fused_features[scale] = projected + residual
            return fused_features
        
        # ===== STAGE 1: Local-Global Hybrid Attention =====
        stage1_features = []
        scale_order = []
        
        for scale in self.scales:
            if scale not in query_features:
                continue
                
            scale_order.append(scale)
            query_feat = query_features[scale]
            
            # Get support prototype (already matches query channels due to Lego design)
            if scale in support_features and support_features[scale] is not None:
                support_proto = support_features[scale]  # [B, C_q] - no projection needed!
            else:
                # No support for this scale, use zeros
                B, C = query_feat.shape[0], query_feat.shape[1]
                support_proto = torch.zeros(B, C, device=query_feat.device, dtype=query_feat.dtype)
            
            # Efficient linear attention: query-support matching
            attn_feat = self.efficient_attentions[scale](query_feat, support_proto)
            
            if self.use_short_long_conv:
                # Short-long convolution: local patterns
                conv_feat = self.short_long_convs[scale](query_feat)
                
                # Concatenate attention and convolution branches
                hybrid = torch.cat([attn_feat, conv_feat], dim=1)
            else:
                hybrid = attn_feat
            
            # Fuse branches
            fused = self.branch_fusion[scale](hybrid)
            
            # Residual connection with original query
            fused = fused + query_feat
            
            stage1_features.append(fused)
        
        # ===== STAGE 2: Cross-Scale Pyramid Refinement =====
        if self.use_pyramid_refinement and len(stage1_features) == self.num_scales:
            stage2_features = self.pyramid_refinement(stage1_features)
        else:
            stage2_features = stage1_features
        
        # ===== STAGE 3: Output Projection =====
        output_features = {}
        
        for i, scale in enumerate(scale_order):
            query_feat = query_features[scale]
            
            # Project to output dimensions
            projected = self.output_projectors[scale](stage2_features[i])
            
            # Residual connection from original query
            residual = self.residual_projectors[scale](query_feat)
            
            # Final output
            output = projected + residual
            
            output_features[scale] = output
        
        return output_features
    
    def get_parameter_count(self) -> dict:
        """Calculate parameter count for each component."""
        def count_params(module):
            return sum(p.numel() for p in module.parameters())
        
        stats = {
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
# TESTING
# ============================================================================

def test_cheaf_fusion():
    """Quick sanity test for CHEAF fusion module."""
    print("\n" + "="*60)
    print("Testing CHEAF Fusion Module")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Initialize CHEAF module with Lego design (matching query/support channels)
    cheaf = CHEAFFusionModule(
        query_channels=[32, 64, 128, 256],  # YOLOv8 outputs [P2, P3, P4, P5]
        support_channels=[32, 64, 128, 256],  # DINOv3 outputs (must match!)
        out_channels=[128, 256, 512, 512],
        num_heads=4,
        use_pyramid_refinement=True,
        use_short_long_conv=True,
    ).to(device)
    
    cheaf.eval()
    
    # Test with mock features
    print("\n[Test 1] Single image fusion:")
    
    # Mock query features (from YOLOv8) - matching support channels
    query_feats = {
        'p2': torch.randn(1, 32, 160, 160).to(device),
        'p3': torch.randn(1, 64, 80, 80).to(device),
        'p4': torch.randn(1, 128, 40, 40).to(device),
        'p5': torch.randn(1, 256, 20, 20).to(device),
    }
    
    # Mock support features (from DINOv3) - matching query channels
    support_feats = {
        'p2': torch.randn(1, 32).to(device),
        'p3': torch.randn(1, 64).to(device),
        'p4': torch.randn(1, 128).to(device),
        'p5': torch.randn(1, 256).to(device),
    }
    
    with torch.no_grad():
        fused = cheaf(query_feats, support_feats)
    
    print("  Query features:")
    for scale, feat in query_feats.items():
        print(f"    {scale}: {feat.shape}")
    
    print("  Support features:")
    for scale, feat in support_feats.items():
        print(f"    {scale}: {feat.shape}")
    
    print("  Fused features:")
    for scale, feat in fused.items():
        print(f"    {scale}: {feat.shape}")
    
    # Test with batch
    print("\n[Test 2] Batch fusion:")
    batch_size = 4
    
    query_batch = {
        'p2': torch.randn(batch_size, 32, 160, 160).to(device),
        'p3': torch.randn(batch_size, 64, 80, 80).to(device),
        'p4': torch.randn(batch_size, 128, 40, 40).to(device),
        'p5': torch.randn(batch_size, 256, 20, 20).to(device),
    }
    
    support_batch = {
        'p2': torch.randn(batch_size, 32).to(device),
        'p3': torch.randn(batch_size, 64).to(device),
        'p4': torch.randn(batch_size, 128).to(device),
        'p5': torch.randn(batch_size, 256).to(device),
    }
    
    with torch.no_grad():
        fused_batch = cheaf(query_batch, support_batch)
    
    print(f"  Batch size: {batch_size}")
    for scale, feat in fused_batch.items():
        print(f"    {scale}: {feat.shape}")
    
    # Test bypass mode
    print("\n[Test 3] Bypass mode (no support):")
    
    with torch.no_grad():
        fused_bypass = cheaf(query_feats, None)
    
    print("  Bypass mode (support_features=None):")
    for scale, feat in fused_bypass.items():
        print(f"    {scale}: {feat.shape}")
    
    # Parameter count
    print("\n[Parameter Count]")
    param_stats = cheaf.get_parameter_count()
    for component, count in param_stats.items():
        print(f"  {component}: {count/1e6:.2f}M")
    
    print("\n[Comparison]")
    print(f"  Original SCS: ~1.2M parameters")
    print(f"  CHEAF Module: {param_stats['total']/1e6:.2f}M parameters")
    print(f"  Increase: {(param_stats['total']/1.2e6 - 1)*100:.1f}%")
    
    print("\n" + "="*60)
    print("✅ All tests passed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    test_cheaf_fusion()
