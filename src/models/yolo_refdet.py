"""
YOLOv8n-RefDet: End-to-End Reference-Based Detection
====================================================

Complete integrated model combining all components:
- DINOv3 support encoder for reference features
- YOLOv8n backbone for query features
- PSALM fusion for cross-scale feature combination
- Prototype detection head for novel class detection

Key Features:
- Support feature caching for efficiency
- Single-forward or multi-shot reference
- ~29.8M total parameters
- Real-time capable on Jetson Xavier NX
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from .dino_encoder import DINOSupportEncoder
from .yolo_backbone import YOLOBackboneExtractor
from .psalm_fusion import PSALMFusion
from .prototype_head import PrototypeDetectionHead


class YOLORefDet(nn.Module):
    """
    YOLOv8n-RefDet: Reference-Based Detection Model
    
    Complete end-to-end model for UAV search-and-rescue object detection
    with reference image support.
    
    Args:
        yolo_weights: Path to YOLOv8n pretrained weights
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
                                                  Prototype Head
                                                          ↓
                                                    Detections
    """
    
    def __init__(
        self,
        yolo_weights: str = "yolov8n.pt",
        dinov3_model: str = "vit_small_patch16_dinov3.lvd1689m",
        freeze_yolo: bool = False,
        freeze_dinov3: bool = True,
        freeze_dinov3_layers: int = 6,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
    ):
        super().__init__()
        
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
        self.backbone = YOLOBackboneExtractor(
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
        
        # Module 4: Prototype Detection Head (~1.7M params)
        print("\n[4/4] Initializing Prototype Detection Head...")
        self.detection_head = PrototypeDetectionHead(
            ch=[128, 256, 512, 512],  # Matches PSALM fusion output - 4 scales
            proto_dims=[32, 64, 128, 256],  # Scale-specific prototype dimensions matching DINOv3
            temperature=10.0,
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
        print(f"  DINOv3 Support Encoder:  {support_params/1e6:.2f}M params")
        print(f"  YOLOv8n Backbone:        {backbone_params/1e6:.2f}M params")
        print(f"  PSALM Fusion Module:     {fusion_params/1e6:.2f}M params")
        print(f"  Prototype Detection Head:{head_params/1e6:.2f}M params")
        print(f"  {'─'*40}")
        print(f"  Total Parameters:        {total_params/1e6:.2f}M params")
        print(f"  Parameter Budget:        {total_params/50e6*100:.1f}% of 50M limit")
        print(f"  Remaining Budget:        {(50e6-total_params)/1e6:.2f}M params")
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
        allow_gradients: bool = False,
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
            allow_gradients: If True, allow gradients to flow (for training). If False, use no_grad (for inference)
        """
        # Use context manager conditionally based on allow_gradients
        ctx = torch.enable_grad() if allow_gradients else torch.no_grad()
        with ctx:
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
                - support: (B, 256) from DINOv3 encoder (384→256 via triplet_proj)
        """
        if image_type == 'query':
            # Extract from YOLOv8 backbone
            features = self.backbone(images, return_global_feat=True)
            return features['global_feat']  # (B, 256)
        elif image_type == 'support':
            # Extract from DINOv3 encoder
            features = self.support_encoder(images, return_global_feat=True)
            return features['global_feat']  # (B, 256) - projected via triplet_proj
        else:
            raise ValueError(f"Invalid image_type: {image_type}. Must be 'query' or 'support'")
    
    def forward(
        self,
        query_image: torch.Tensor,
        support_images: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        return_features: bool = False,
        class_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through complete model.
        
        Args:
            query_image: Query image (B, 3, 640, 640)
            support_images: Reference images (K, 3, 256, 256) or None to use cache
            use_cache: Whether to use cached support features if available
            return_features: Whether to return intermediate features for triplet loss (default: False)
            class_ids: Class IDs for episodic learning (optional)
        
        Returns:
            Dictionary containing detection outputs from prototype head
            If return_features=True, also includes:
                - 'query_global_feat': (B, 256) for triplet positives/negatives
                - 'support_global_feat': (B, 384) for triplet anchors (if support_images provided)
        """
        # Step 1: Extract query features from YOLOv8 backbone
        query_features = self.backbone(query_image, return_global_feat=return_features)  # {p2, p3, p4, p5, [global_feat]}
        
        # Step 2: Get support features (from cache or compute)
        support_features = None
        support_features_for_head = None  # Keep original N prototypes for detection head
        
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
                "Prototype detection requires support images or cached features. "
                "Call set_reference_images() first or provide support_images."
            )
        
        # Step 3: Fuse query and support features
        fused_features = self.scs_fusion(query_features, support_features)
        
        # Step 4: Detection head
        # Prepare prototypes for detection head (use scale-specific features)
        # Use support_features_for_head to preserve N-way prototypes for episodic learning
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
        
        # CRITICAL: Clamp fused features before detection head to prevent gradient explosion
        # This prevents extreme values from propagating into detection head and causing NaN gradients
        fused_features_clamped = {
            scale: torch.clamp(feat, min=-10.0, max=10.0)
            for scale, feat in fused_features.items()
        }
        
        # Convert dict to list for detection head (expects list [P2, P3, P4, P5])
        fused_features_list = [
            fused_features_clamped['p2'],
            fused_features_clamped['p3'],
            fused_features_clamped['p4'],
            fused_features_clamped['p5'],
        ]
        
        # Detection head returns (box_preds, sim_scores) tuple
        box_preds, sim_scores = self.detection_head(fused_features_list, prototypes)
        
        # Convert to dict format expected by rest of pipeline
        detections = {
            'prototype_boxes': box_preds,
            'prototype_sim': sim_scores,
        }
        
        # Add fused features to outputs for CPE loss (ROI feature extraction)
        detections['fused_features'] = fused_features
        
        # Step 5: Add features for contrastive losses
        # For SupCon loss (Stage 2): Add query_features and support_prototypes
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
        conf_thres: Optional[float] = None,
        iou_thres: Optional[float] = None,
        max_det: int = 300,
        return_raw: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        High-level inference method with post-processing for UAV video streams.
        
        **Efficient UAV Inference Workflow:**
        1. First frame: Call with support_images to cache reference features
        2. Subsequent frames: Call with support_images=None (uses cache)
        3. Clear cache when switching to new target: model.clear_cache()
        
        Example UAV video processing:
        ```python
        model.eval()
        
        # Load reference images once (K-shot)
        ref_images = load_reference_images()  # (K, 3, 256, 256)
        
        # Process video stream
        for frame in video_stream:
            frame_tensor = preprocess(frame)  # (1, 3, 640, 640)
            
            # First frame: cache references
            if frame_idx == 0:
                detections = model.inference(frame_tensor, support_images=ref_images)
            else:
                # Use cached features (fast!)
                detections = model.inference(frame_tensor)
            
            # Draw bboxes
            draw_detections(frame, detections)
        ```
        
        Args:
            query_image: Query image tensor (B, 3, 640, 640)
                - For video: typically B=1 (single frame)
                - Pre-normalized to [0, 1] range
            support_images: Reference images (K, 3, 256, 256) or None
                - If provided: Cache these reference features
                - If None: Use previously cached features (efficient for video)
                - K can be 1 (single ref) or multiple (K-shot averaging)
            conf_thres: Confidence threshold (default: 0.25)
            iou_thres: IoU threshold for NMS (default: 0.45)
            max_det: Maximum detections per image (default: 300)
            return_raw: If True, return raw model outputs without post-processing
        
        Returns:
            Dictionary with post-processed detections:
                - 'bboxes': (B, N, 4) in xyxy format [x1, y1, x2, y2]
                - 'scores': (B, N) confidence scores [0, 1]
                - 'class_ids': (B, N) predicted class indices
                - 'num_detections': (B,) number of valid detections per image
                
                Where N = max number of detections (padded)
                Valid detections are where scores > 0
        """
        # Use provided thresholds or defaults
        conf_thres = conf_thres or self.conf_thres
        iou_thres = iou_thres or self.iou_thres
        
        # Ensure model is in eval mode
        was_training = self.training
        if was_training:
            self.eval()
        
        with torch.no_grad():
            # Cache support features if provided
            if support_images is not None:
                # Process and cache support features for efficient video stream processing
                if support_images.shape[0] > 1:
                    # K-shot: average multiple support images into single prototype
                    support_list = [support_images[i:i+1] for i in range(support_images.shape[0])]
                    support_feats = self.support_encoder.compute_average_prototype(
                        support_list, 
                        return_global_feat=False
                    )
                else:
                    # Single-shot: just encode
                    support_feats = self.support_encoder(support_images, return_global_feat=False)
                
                # Cache for subsequent frames
                self._cached_support_features = support_feats
            
            # Forward pass
            raw_outputs = self.forward(
                query_image, 
                support_images=None if support_images is not None else None,  # Always use cache after setting
                use_cache=True,  # Always use cache in inference
            )
            
            # Return raw outputs if requested (for debugging or custom post-processing)
            if return_raw:
                if was_training:
                    self.train()
                return raw_outputs
            
            # Post-process detections (NMS, thresholding)
            detections = self._postprocess_detections(
                raw_outputs,
                conf_thres=conf_thres,
                iou_thres=iou_thres,
                max_det=max_det,
            )
        
        # Restore training mode if needed
        if was_training:
            self.train()
        
        return detections
    
    def _postprocess_detections(
        self,
        model_outputs: Dict[str, torch.Tensor],
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        max_det: int = 300,
    ) -> Dict[str, torch.Tensor]:
        """
        Post-process raw model outputs to extract final detections.
        
        Follows YOLOv8 inference pipeline:
        1. Decode bbox predictions from anchor offsets to xyxy coordinates
        2. Apply confidence thresholding
        3. Apply NMS per class
        4. Limit to max_det detections
        
        Args:
            model_outputs: Raw model outputs from forward pass
            conf_thres: Confidence threshold
            iou_thres: IoU threshold for NMS
            max_det: Maximum number of detections
        
        Returns:
            Post-processed detections dictionary
        """
        from torchvision.ops import nms
        
        # Extract prototype head outputs
        boxes_list = model_outputs['prototype_boxes']  # List of (B, 4, H, W) per scale
        scores_list = model_outputs['prototype_sim']   # List of (B, K, H, W) per scale
        
        device = boxes_list[0].device
        batch_size = boxes_list[0].shape[0]
        
        # Strides for each scale [P2, P3, P4, P5]
        strides = torch.tensor([4, 8, 16, 32], device=device, dtype=torch.float32)
        
        # Concatenate all scales
        box_list_flat = []
        scores_list_flat = []
        
        for boxes, scores in zip(boxes_list, scores_list):
            B, C, H, W = boxes.shape
            box_list_flat.append(boxes.view(B, C, -1))  # (B, 4, H*W)
            scores_list_flat.append(scores.view(B, scores.shape[1], -1))  # (B, K, H*W)
        
        # Concatenate across scales
        box_cat = torch.cat(box_list_flat, dim=2)  # (B, 4, total_anchors)
        scores_cat = torch.cat(scores_list_flat, dim=2)  # (B, K, total_anchors)
        
        # Generate anchor points
        anchor_points_list = []
        stride_tensor_list = []
        for i, (boxes_scale, stride) in enumerate(zip(boxes_list, strides)):
            _, _, h, w = boxes_scale.shape
            sy = torch.arange(h, device=device, dtype=torch.float32) + 0.5
            sx = torch.arange(w, device=device, dtype=torch.float32) + 0.5
            sy, sx = torch.meshgrid(sy, sx, indexing='ij')
            anchor_points_list.append(torch.stack((sx, sy), -1).view(-1, 2))
            stride_tensor_list.append(torch.full((h * w, 1), stride.item(), device=device, dtype=torch.float32))
        
        anchor_points = torch.cat(anchor_points_list)  # (total_anchors, 2)
        stride_tensor = torch.cat(stride_tensor_list)  # (total_anchors, 1)
        
        # Decode bbox predictions from ltrb offsets to xyxy coordinates
        box_preds = box_cat.permute(0, 2, 1)  # (B, total_anchors, 4) [l, t, r, b]
        
        anchor_x = anchor_points[:, 0:1] * stride_tensor
        anchor_y = anchor_points[:, 1:2] * stride_tensor
        
        eps = 1e-4
        decoded_boxes = torch.stack([
            anchor_x.squeeze(1) - box_preds[:, :, 0] * stride_tensor.squeeze(1),
            anchor_y.squeeze(1) - box_preds[:, :, 1] * stride_tensor.squeeze(1),
            anchor_x.squeeze(1) + box_preds[:, :, 2] * stride_tensor.squeeze(1) + eps,
            anchor_y.squeeze(1) + box_preds[:, :, 3] * stride_tensor.squeeze(1) + eps,
        ], dim=2)  # (B, total_anchors, 4)
        
        # Apply sigmoid to scores and get class predictions
        scores_cat = scores_cat.sigmoid().permute(0, 2, 1)  # (B, total_anchors, K)
        max_scores, class_ids = scores_cat.max(dim=-1)  # (B, total_anchors)
        
        # Apply confidence threshold and NMS per batch
        final_boxes = []
        final_scores = []
        final_class_ids = []
        
        for b in range(batch_size):
            # Filter by confidence
            conf_mask = max_scores[b] >= conf_thres
            boxes_b = decoded_boxes[b][conf_mask]
            scores_b = max_scores[b][conf_mask]
            class_ids_b = class_ids[b][conf_mask]
            
            if len(boxes_b) == 0:
                # No detections above threshold
                final_boxes.append(torch.zeros((0, 4), device=device))
                final_scores.append(torch.zeros(0, device=device))
                final_class_ids.append(torch.zeros(0, dtype=torch.long, device=device))
                continue
            
            # Apply NMS
            keep_indices = nms(boxes_b, scores_b, iou_thres)
            keep_indices = keep_indices[:max_det]  # Limit to max_det
            
            final_boxes.append(boxes_b[keep_indices])
            final_scores.append(scores_b[keep_indices])
            final_class_ids.append(class_ids_b[keep_indices])
        
        # Pad to same length for batching
        max_dets = max(len(b) for b in final_boxes) if final_boxes else 0
        max_dets = min(max(max_dets, 1), max_det)  # At least 1, at most max_det
        
        padded_boxes = torch.zeros((batch_size, max_dets, 4), device=device)
        padded_scores = torch.zeros((batch_size, max_dets), device=device)
        padded_class_ids = torch.zeros((batch_size, max_dets), dtype=torch.long, device=device)
        num_detections = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        for b in range(batch_size):
            n = len(final_boxes[b])
            if n > 0:
                n = min(n, max_dets)
                padded_boxes[b, :n] = final_boxes[b][:n]
                padded_scores[b, :n] = final_scores[b][:n]
                padded_class_ids[b, :n] = final_class_ids[b][:n]
                num_detections[b] = n
        
        return {
            'bboxes': padded_boxes,
            'scores': padded_scores,
            'class_ids': padded_class_ids,
            'num_detections': num_detections,
        }


def test_yolov8n_refdet():
    """Quick sanity test for end-to-end model (prototype-only mode)."""
    print("\n" + "="*70)
    print("Testing YOLOv8n-RefDet End-to-End Model (Prototype-Only)")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Check if custom weights exist
    weights_path = "baseline_enot_nano/weights/best.pt"
    if not Path(weights_path).exists():
        print(f"⚠️  Custom weights not found, using default yolov8n.pt")
        weights_path = "yolov8n.pt"
    
    # Initialize complete model (prototype-only, no base classes)
    print("Initializing YOLOv8n-RefDet...")
    model = YOLORefDet(
        yolo_weights=weights_path,
        freeze_yolo=False,
        freeze_dinov3=True,
        freeze_dinov3_layers=6,
    ).to(device)
    
    model.eval()
    
    # Test 1: Prototype detection with on-the-fly computation
    print("\n[Test 1] Prototype detection (on-the-fly support features):")
    query_img = torch.randn(1, 3, 640, 640).to(device)
    support_imgs = torch.randn(3, 3, 256, 256).to(device)  # 3 reference images
    
    with torch.no_grad():
        outputs = model(query_img, support_imgs)
    
    print(f"  Query image: {query_img.shape}")
    print(f"  Support images: {support_imgs.shape}")
    print(f"  Has prototype outputs: {'prototype_boxes' in outputs}")
    
    # Test 2: Cached support features
    print("\n[Test 2] Prototype detection with cached support features:")
    
    # Cache support features
    model.set_reference_images(support_imgs)
    
    with torch.no_grad():
        outputs_cached = model(query_img, use_cache=True)
    
    print(f"  Using cached features: True")
    print(f"  Has prototype outputs: {'prototype_boxes' in outputs_cached}")
    
    # Test 3: Batch processing
    print("\n[Test 3] Batch processing:")
    query_batch = torch.randn(4, 3, 640, 640).to(device)
    
    with torch.no_grad():
        outputs_batch = model(query_batch, use_cache=True)
    
    print(f"  Batch size: {query_batch.shape[0]}")
    print(f"  Output boxes shape: {outputs_batch['prototype_boxes'][0].shape}")
    
    # Test 4: Multiple reference images with averaging (K-shot)
    print("\n[Test 4] Multiple reference images (K-shot):")
    support_list = [torch.randn(1, 3, 256, 256).to(device) for _ in range(5)]
    
    # Convert list to tensor for set_reference_images
    support_tensor = torch.cat(support_list, dim=0)  # (5, 3, 256, 256)
    model.set_reference_images(support_tensor, average_prototypes=True)
    
    with torch.no_grad():
        outputs_kshot = model(query_img, use_cache=True)
    
    print(f"  K-shot: {len(support_list)} reference images")
    print(f"  Averaged prototypes cached")
    
    # Test 5: Clear cache
    print("\n[Test 5] Cache management:")
    print(f"  Cache status before clear: {model._cached_support_features is not None}")
    model.clear_cache()
    print(f"  Cache status after clear: {model._cached_support_features is not None}")
    
    # Test 6: Inference method
    print("\n[Test 6] High-level inference method:")
    model.set_reference_images(support_imgs)
    
    with torch.no_grad():
        inference_outputs = model.inference(
            query_img,
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
