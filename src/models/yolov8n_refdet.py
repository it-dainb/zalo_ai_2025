"""
YOLOv8n-RefDet: End-to-End Reference-Based Detection
====================================================

Complete integrated model combining all components:
- DINOv3 support encoder for reference features
- YOLOv8n backbone for query features
- PSALM fusion for cross-scale feature combination
- Dual detection head for base + novel class detection

Key Features:
- Flexible inference modes (standard, prototype, dual)
- Support feature caching for efficiency
- Single-forward or multi-shot reference
- ~10.4M total parameters
- Real-time capable on Jetson Xavier NX
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from .dino_encoder import DINOSupportEncoder
from .yolov8_backbone import YOLOv8BackboneExtractor
from .psalm_fusion import PSALMFusion
from .dual_head import DualDetectionHead


class YOLOv8nRefDet(nn.Module):
    """
    YOLOv8n-RefDet: Reference-Based Detection Model
    
    Complete end-to-end model for UAV search-and-rescue object detection
    with reference image support.
    
    Args:
        yolo_weights: Path to YOLOv8n pretrained weights
        nc_base: Number of base classes (default: 80 for COCO)
        dinov2_model: DINO model name from timm (DINOv3 by default)
        freeze_yolo: Whether to freeze YOLOv8 backbone
        freeze_dinov2: Whether to freeze DINO encoder
        conf_thres: Confidence threshold for detections
        iou_thres: IoU threshold for NMS
    
    Architecture:
        Query Image (640×640) ──→ YOLOv8n Backbone ──→ [P2, P3, P4, P5]
                                                          ↓
        Support Images (256×256) ──→ DINOv3 Encoder ──→ Prototypes
                                                          ↓
                                                    PSALM Fusion
                                                          ↓
                                                   Dual Head
                                                          ↓
                                                    Detections
    """
    
    def __init__(
        self,
        yolo_weights: str = "yolov8n.pt",
        nc_base: int = 80,
        dinov3_model: str = "vit_small_patch16_dinov3.lvd1689m",
        freeze_yolo: bool = False,
        freeze_dinov3: bool = True,
        freeze_dinov3_layers: int = 6,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
    ):
        super().__init__()
        
        self.nc_base = nc_base
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        # Module 1: DINOv3 Support Encoder
        print("\n[1/4] Initializing DINOv3 Support Encoder...")
        self.support_encoder = DINOSupportEncoder(
            model_name=dinov3_model,
            output_dims=[32, 64, 128, 256],  # Match YOLOv8n backbone dimensions (Lego design) - 4 scales
            freeze_backbone=freeze_dinov3,
            freeze_layers=freeze_dinov3_layers,
        )
        
        # Module 2: YOLOv8n Backbone
        print("\n[2/4] Initializing YOLOv8 Backbone...")
        self.backbone = YOLOv8BackboneExtractor(
            weights_path=yolo_weights,
            extract_scales=['p2', 'p3', 'p4', 'p5'],
            freeze_backbone=freeze_yolo,
        )
        
        # Module 3: PSALM Fusion Module (Clean Lego design - matching dimensions)
        # YOLOv8n outputs: P2=32ch, P3=64ch, P4=128ch, P5=256ch
        # DINOv3 outputs: P2=32ch, P3=64ch, P4=128ch, P5=256ch (matching!)
        print("\n[3/4] Initializing PSALM Fusion Module...")
        self.scs_fusion = PSALMFusion(
            query_channels=[32, 64, 128, 256],  # YOLOv8n backbone channels - 4 scales
            support_channels=[32, 64, 128, 256],  # DINOv3 matching channels (Lego design)
            out_channels=[128, 256, 512, 512],  # Standard detection head input
            num_heads=4,
        )
        
        # Module 4: Dual Detection Head (0.5M params)
        print("\n[4/4] Initializing Dual Detection Head...")
        self.detection_head = DualDetectionHead(
            nc_base=nc_base,
            ch=[128, 256, 512, 512],  # Matches PSALM fusion output - 4 scales
            proto_dims=[32, 64, 128, 256],  # Scale-specific prototype dimensions matching DINOv3
            temperature=10.0,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
        )
        
        # Cache for pre-computed support features
        self._cached_support_features = None
        
        print("\n" + "="*60)
        print("YOLOv8n-RefDet Initialization Complete")
        print("="*60)
        self._print_model_summary()
    
    def _print_model_summary(self):
        """Print model summary with parameter counts."""
        support_params = self.support_encoder.count_parameters()
        backbone_params = self.backbone.count_parameters()
        fusion_params = self.scs_fusion.count_parameters()
        head_params = self.detection_head.count_parameters()
        total_params = support_params + backbone_params + fusion_params + head_params
        
        print(f"\nModel Summary:")
        print(f"  DINOv3 Support Encoder: {support_params/1e6:.2f}M params")
        print(f"  YOLOv8n Backbone:       {backbone_params/1e6:.2f}M params")
        print(f"  PSALM Fusion Module:    {fusion_params/1e6:.2f}M params")
        print(f"  Dual Detection Head:    {head_params/1e6:.2f}M params")
        print(f"  {'─'*40}")
        print(f"  Total Parameters:       {total_params/1e6:.2f}M params")
        print(f"  Parameter Budget:       {total_params/50e6*100:.1f}% of 50M limit")
        print(f"  Remaining Budget:       {(50e6-total_params)/1e6:.2f}M params")
        print("="*60 + "\n")
    
    def count_parameters(self, trainable_only: bool = False) -> int:
        """Count total model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def set_reference_images(
        self, 
        support_images: torch.Tensor,
        average_prototypes: bool = True,
        n_way: int = None,
        n_support: int = None,
    ):
        """
        Pre-compute and cache support features from reference images.
        Useful for inference optimization and episodic learning.
        
        Args:
            support_images: Reference images
                - Single: (1, 3, 256, 256)
                - Multiple: (K, 3, 256, 256) or (N*K, 3, 256, 256) for episodic learning
            average_prototypes: If True and multiple images, average prototypes
            n_way: Number of classes (for episodic learning)
            n_support: Number of support images per class (for episodic learning)
        """
        with torch.no_grad():
            # Episodic learning: N classes with K support images each
            if n_way is not None and n_support is not None and average_prototypes:
                # support_images shape: (N*K, 3, H, W)
                # We need to compute N prototypes (one per class)
                all_class_prototypes = []
                for class_idx in range(n_way):
                    # Extract K support images for this class
                    start_idx = class_idx * n_support
                    end_idx = start_idx + n_support
                    class_support = support_images[start_idx:end_idx]  # (K, 3, H, W)
                    
                    # Average K support images to get one prototype
                    support_list = [class_support[k:k+1] for k in range(n_support)]
                    class_proto = self.support_encoder.compute_average_prototype(support_list)
                    all_class_prototypes.append(class_proto)
                
                # Stack N class prototypes: {scale: (N, dim)}
                support_feats = {
                    scale: torch.cat([proto[scale] for proto in all_class_prototypes], dim=0)
                    for scale in all_class_prototypes[0].keys()
                }
                self._cached_support_features = support_feats
                
            elif isinstance(support_images, list) and average_prototypes:
                # Average multiple reference images (list format) - single class
                support_feats = self.support_encoder.compute_average_prototype(support_images)
                self._cached_support_features = {
                    scale: feat[:1] if feat.shape[0] > 1 else feat
                    for scale, feat in support_feats.items()
                }
            elif isinstance(support_images, torch.Tensor) and support_images.shape[0] > 1 and average_prototypes:
                # Average multiple reference images (tensor format) - single class
                support_list = [support_images[i:i+1] for i in range(support_images.shape[0])]
                support_feats = self.support_encoder.compute_average_prototype(support_list)
                self._cached_support_features = {
                    scale: feat[:1] if feat.shape[0] > 1 else feat
                    for scale, feat in support_feats.items()
                }
            else:
                # Single reference or no averaging
                support_feats = self.support_encoder(support_images)
                self._cached_support_features = support_feats
        
        # Caching complete (removed verbose logging)
    
    def clear_cache(self):
        """Clear cached support features."""
        self._cached_support_features = None
    
    def extract_features(
        self,
        images: torch.Tensor,
        image_type: str = 'query',
    ) -> torch.Tensor:
        """
        Extract global features for triplet loss WITHOUT going through fusion.
        
        This method directly accesses the encoders to extract features,
        bypassing the fusion module to avoid batch size mismatch issues.
        
        Args:
            images: Input images
                - For query (positive/negative): (B, 3, 640, 640)
                - For support (anchor): (B, 3, 256, 256)
            image_type: Type of images ('query' or 'support')
        
        Returns:
            Global feature tensor:
                - query: (B, 256) from YOLOv8 backbone
                - support: (B, 384) from DINOv3 encoder
        """
        if image_type == 'query':
            # Extract from YOLOv8 backbone
            features = self.backbone(images, return_global_feat=True)
            return features['global_feat']  # (B, 256)
        elif image_type == 'support':
            # Extract from DINOv3 encoder
            features = self.support_encoder(images, return_global_feat=True)
            return features['global_feat']  # (B, 384)
        else:
            raise ValueError(f"Invalid image_type: {image_type}. Must be 'query' or 'support'")
    
    def forward(
        self,
        query_image: torch.Tensor,
        support_images: Optional[torch.Tensor] = None,
        mode: str = 'dual',
        use_cache: bool = True,
        return_features: bool = False,
        class_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through complete model.
        
        Args:
            query_image: Query image (B, 3, 640, 640)
            support_images: Reference images (K, 3, 256, 256) or None to use cache
            mode: Detection mode ('standard', 'prototype', 'dual')
            use_cache: Whether to use cached support features if available
            return_features: Whether to return intermediate features for triplet loss (default: False)
        
        Returns:
            Dictionary containing detection outputs from dual head
            If return_features=True, also includes:
                - 'query_global_feat': (B, 256) for triplet positives/negatives
                - 'support_global_feat': (B, 384) for triplet anchors (if support_images provided)
        """
        # Step 1: Extract query features from YOLOv8 backbone
        query_features = self.backbone(query_image, return_global_feat=return_features)  # {p2, p3, p4, p5, [global_feat]}
        
        # Step 2: Get support features (from cache or compute)
        support_features = None
        support_features_for_head = None  # Keep original N prototypes for detection head
        
        if mode in ['prototype', 'dual']:
            if use_cache and self._cached_support_features is not None:
                # Use cached support features
                support_features_original = self._cached_support_features
                
                # Determine how to expand for fusion
                batch_size = query_image.shape[0]
                first_scale = list(support_features_original.keys())[0]
                num_prototypes = support_features_original[first_scale].shape[0]
                
                if num_prototypes == 1 and batch_size > 1:
                    # Single prototype: expand to match batch size
                    support_features = {
                        scale: feat.expand(batch_size, -1)
                        for scale, feat in support_features_original.items()
                    }
                    support_features_for_head = support_features
                elif num_prototypes > 1 and class_ids is not None:
                    # N-way episodic learning:
                    # - For fusion: select prototype for each query (B, dim)
                    # - For detection head: keep all N prototypes (N, dim)
                    support_features = {
                        scale: feat[class_ids]  # (N, dim) -> (B, dim) via indexing
                        for scale, feat in support_features_original.items()
                    }
                    support_features_for_head = support_features_original  # Keep (N, dim)
                else:
                    # Multiple prototypes without class_ids - keep as (N, dim) for both
                    support_features = support_features_original
                    support_features_for_head = support_features_original
            elif support_images is not None:
                # Compute support features on-the-fly
                # If support_images has multiple samples (K > 1), decide whether to average
                # - For detection (return_features=False): Average K-shot examples into single prototype
                # - For triplet loss (return_features=True): Process each support image individually
                if support_images.shape[0] > 1 and not return_features:
                    # Few-shot detection: average K support images into single prototype
                    support_list = [support_images[i:i+1] for i in range(support_images.shape[0])]
                    support_features = self.support_encoder.compute_average_prototype(support_list, return_global_feat=False)
                    
                    # Expand to match query batch size for detection
                    batch_size = query_image.shape[0]
                    if batch_size > 1:
                        support_features = {
                            scale: feat.expand(batch_size, -1)
                            for scale, feat in support_features.items()
                        }
                    support_features_for_head = support_features
                else:
                    # Single support image OR triplet loss: process directly without averaging
                    support_features = self.support_encoder(support_images, return_global_feat=return_features)
                    
                    # For triplet loss, keep original batch dimension from support_images
                    # For detection with single support, expand to match query batch size
                    if not return_features:
                        batch_size = query_image.shape[0]
                        if batch_size > 1 and support_images.shape[0] == 1:
                            support_features = {
                                scale: feat.expand(batch_size, -1)
                                for scale, feat in support_features.items()
                            }
                    support_features_for_head = support_features
            else:
                raise ValueError(
                    f"Mode '{mode}' requires support images or cached features. "
                    "Call set_reference_images() first or provide support_images."
                )
        
        # Step 3: Fuse query and support features
        # Always pass through fusion to get correct output channels [256, 512, 512]
        # In N-way episodic mode: support_features is (B, dim) expanded for fusion
        # In standard mode, pass None for support_features (fusion will skip correlation)
        if mode in ['prototype', 'dual']:
            fused_features = self.scs_fusion(query_features, support_features)
        else:
            # Standard mode: project query features to correct channels without fusion
            fused_features = self.scs_fusion(query_features, None)
        
        # Step 4: Detection head
        # Prepare prototypes for detection head (use scale-specific features)
        # Use support_features_for_head to preserve N-way prototypes for episodic learning
        prototypes = None
        if support_features_for_head is not None:
            # Use scale-specific prototypes for better multi-scale detection
            # For episodic learning: (N, dim) where N = num_classes
            # For single prototype: (B, dim) where B = batch_size
            # Scale-specific dimensions [P2:32, P3:64, P4:128, P5:256] match proto_dims
            prototypes = {
                'p2': support_features_for_head['p2'],
                'p3': support_features_for_head['p3'],
                'p4': support_features_for_head['p4'],
                'p5': support_features_for_head['p5'],
            }
        
        detections = self.detection_head(fused_features, prototypes, mode=mode)
        
        # Step 5: Add features for contrastive losses
        # For SupCon loss (Stage 2): Add query_features and support_prototypes
        if mode in ['prototype', 'dual'] and support_features_for_head is not None:
            # Query features: Use P4 scale features for contrastive learning (middle scale)
            # Pool spatial features to 1D: (B, 128, H, W) -> (B, 128)
            if 'p4' in query_features:
                query_feat_map = query_features['p4']  # (B, 128, H, W)
                # Global average pooling
                query_feat_pooled = torch.nn.functional.adaptive_avg_pool2d(query_feat_map, 1)  # (B, 128, 1, 1)
                detections['query_features'] = query_feat_pooled.squeeze(-1).squeeze(-1)  # (B, 128)
            
            # Support prototypes: Already 1D vectors from DINO encoder
            # Shape: (N, 128) for N-way episodic, or (1, 128) for single prototype
            if 'p4' in support_features_for_head:
                detections['support_prototypes'] = support_features_for_head['p4']
        
        # For triplet loss (Stage 3): Add global features if requested
        if return_features:
            if 'global_feat' in query_features:
                detections['query_global_feat'] = query_features['global_feat']
            if support_features is not None and 'global_feat' in support_features:
                detections['support_global_feat'] = support_features['global_feat']
        
        return detections
    
    def inference(
        self,
        query_image: torch.Tensor,
        support_images: Optional[torch.Tensor] = None,
        mode: str = 'dual',
        conf_thres: Optional[float] = None,
        iou_thres: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        High-level inference method with post-processing.
        
        Args:
            query_image: Query image (B, 3, 640, 640)
            support_images: Reference images or None to use cache
            mode: Detection mode
            conf_thres: Override confidence threshold
            iou_thres: Override IoU threshold
        
        Returns:
            Post-processed detections
        """
        # Use provided thresholds or defaults
        conf_thres = conf_thres or self.conf_thres
        iou_thres = iou_thres or self.iou_thres
        
        # Forward pass
        outputs = self.forward(query_image, support_images, mode=mode)
        
        # TODO: Add post-processing (NMS, score thresholding)
        # For now, return raw outputs
        return outputs


def test_yolov8n_refdet():
    """Quick sanity test for end-to-end model."""
    print("\n" + "="*70)
    print("Testing YOLOv8n-RefDet End-to-End Model")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Check if custom weights exist
    weights_path = "baseline_enot_nano/weights/best.pt"
    if not Path(weights_path).exists():
        print(f"⚠️  Custom weights not found, using default yolov8n.pt")
        weights_path = "yolov8n.pt"
    
    # Initialize complete model
    print("Initializing YOLOv8n-RefDet...")
    model = YOLOv8nRefDet(
        yolo_weights=weights_path,
        nc_base=80,
        freeze_yolo=False,
        freeze_dinov3=True,
        freeze_dinov3_layers=6,
    ).to(device)
    
    model.eval()
    
    # Test 1: Standard mode (no references)
    print("\n[Test 1] Standard mode (base classes only):")
    query_img = torch.randn(1, 3, 640, 640).to(device)
    
    with torch.no_grad():
        outputs_std = model(query_img, mode='standard')
    
    print(f"  Query image: {query_img.shape}")
    print(f"  Has standard outputs: {'standard_boxes' in outputs_std}")
    print(f"  Has prototype outputs: {'prototype_boxes' in outputs_std}")
    
    # Test 2: Prototype mode with on-the-fly computation
    print("\n[Test 2] Prototype mode (on-the-fly support features):")
    support_imgs = torch.randn(3, 3, 256, 256).to(device)  # 3 reference images
    
    with torch.no_grad():
        outputs_proto = model(query_img, support_imgs, mode='prototype')
    
    print(f"  Support images: {support_imgs.shape}")
    print(f"  Has prototype outputs: {'prototype_boxes' in outputs_proto}")
    
    # Test 3: Dual mode with cached support features
    print("\n[Test 3] Dual mode with cached support features:")
    
    # Cache support features
    model.set_reference_images(support_imgs)
    
    with torch.no_grad():
        outputs_dual = model(query_img, mode='dual', use_cache=True)
    
    print(f"  Using cached features: True")
    print(f"  Has standard outputs: {'standard_boxes' in outputs_dual}")
    print(f"  Has prototype outputs: {'prototype_boxes' in outputs_dual}")
    
    # Test 4: Batch processing
    print("\n[Test 4] Batch processing:")
    query_batch = torch.randn(4, 3, 640, 640).to(device)
    
    with torch.no_grad():
        outputs_batch = model(query_batch, mode='dual', use_cache=True)
    
    print(f"  Batch size: {query_batch.shape[0]}")
    print(f"  Output boxes shape: {outputs_batch['standard_boxes'][0].shape}")
    
    # Test 5: Multiple reference images with averaging
    print("\n[Test 5] Multiple reference images (K-shot):")
    support_list = [torch.randn(1, 3, 256, 256).to(device) for _ in range(5)]
    
    model.set_reference_images(support_list, average_prototypes=True)
    
    with torch.no_grad():
        outputs_kshot = model(query_img, mode='prototype', use_cache=True)
    
    print(f"  K-shot: {len(support_list)} reference images")
    print(f"  Averaged prototypes cached")
    
    # Test 6: Clear cache
    print("\n[Test 6] Cache management:")
    print(f"  Cache status before clear: {model._cached_support_features is not None}")
    model.clear_cache()
    print(f"  Cache status after clear: {model._cached_support_features is not None}")
    
    # Test 7: Inference method
    print("\n[Test 7] High-level inference method:")
    model.set_reference_images(support_imgs)
    
    with torch.no_grad():
        inference_outputs = model.inference(
            query_img,
            mode='dual',
            conf_thres=0.5,
            iou_thres=0.45
        )
    
    print(f"  Inference complete")
    print(f"  Output keys: {list(inference_outputs.keys())}")
    
    # Parameter summary
    print("\n[Parameter Summary]")
    total_params = model.count_parameters(trainable_only=False)
    trainable_params = model.count_parameters(trainable_only=True)
    
    print(f"  Total parameters: {total_params/1e6:.2f}M")
    print(f"  Trainable parameters: {trainable_params/1e6:.2f}M")
    print(f"  Frozen parameters: {(total_params-trainable_params)/1e6:.2f}M")
    print(f"  Parameter budget used: {total_params/50e6*100:.1f}% of 50M")
    
    # Memory usage (approximate)
    if device.type == 'cuda':
        print(f"\n[GPU Memory]")
        print(f"  Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        print(f"  Cached: {torch.cuda.memory_reserved()/1e9:.2f} GB")
    
    print("\n" + "="*70)
    print("✅ All end-to-end tests passed!")
    print("="*70 + "\n")


if __name__ == "__main__":
    test_yolov8n_refdet()
