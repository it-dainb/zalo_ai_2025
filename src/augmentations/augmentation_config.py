"""
Augmentation Configuration and Hyperparameters
Stage-specific augmentation schedules for 3-stage training.

UPGRADED: Now uses AlbumentationsX (10-23x faster augmentation)
=============================================================

Hybrid Augmentation Strategy:
1. Query Path (Drone Frames):
   - Ultralytics: Mosaic, MixUp (optimized for YOLO, +5-7% mAP)
   - AlbumentationsX: Color, geometric, blur, erasing (10-23x faster)
   
2. Support Path (Reference Images):
   - AlbumentationsX: Weak/strong modes for DINOv3
   - Image size: 256×256 (optimal for ViT-S/14 with reg4 tokens)
   
3. Temporal Path (Video Sequences):
   - AlbumentationsX: Consistency window for smooth augmentation
   - Prevents flickering across consecutive frames

New AlbumentationsX Features:
- PlanckianJitter: Realistic lighting simulation (color temperature)
- AdvancedBlur: Motion blur, defocus, glass blur
- Erasing: Random erasing (better than CoarseDropout)
- D4 group: All 90° rotations + flips (preserves grid structure)
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class AugmentationConfig:
    """
    Configuration for data augmentation pipeline.
    
    UPGRADED: Now uses AlbumentationsX for 10-23x faster augmentation
    - Query path: Hybrid (Ultralytics Mosaic/MixUp + AlbumentationsX color/geometric)
    - Support path: AlbumentationsX with semantic-preserving weak/strong modes
    - Temporal path: AlbumentationsX with consistency window for video sequences
    """
    
    # Query augmentation (drone frames) - YOLOv8 detection
    query_img_size: int = 640  # YOLOv8n standard input size
    query_mosaic_prob: float = 1.0  # Ultralytics Mosaic (most impactful: +5-7% mAP)
    query_mixup_prob: float = 0.15  # Ultralytics MixUp
    query_mixup_alpha: float = 0.1  # Beta distribution parameter for MixUp
    
    # Support augmentation (reference images) - DINOv3 encoding
    support_img_size: int = 256  # DINOv3 ViT-S/14 with reg4 tokens
    support_mode: str = "weak"  # "weak" (training) or "strong" (contrastive learning)
    
    # Temporal consistency (video sequences)
    temporal_consistency_window: int = 8  # Frames sharing same augmentation params
    video_frame_stride: int = 1  # Sample every N frames (1 = all frames)
    video_sequence_length: int = 8  # Frames per training sequence
    video_sequence_overlap: int = 4  # Overlap between consecutive sequences
    
    # Geometric augmentations (AlbumentationsX)
    flip_horizontal: float = 0.5
    flip_vertical: float = 0.5
    rotate_90: float = 0.5  # D4 group augmentation (0°, 90°, 180°, 270°)
    affine_scale: tuple = (0.8, 1.2)
    affine_rotate: tuple = (-15, 15)
    affine_translate: tuple = (-0.1, 0.1)
    affine_shear: tuple = (-5, 5)
    
    # Photometric augmentations (AlbumentationsX)
    hsv_hue: tuple = (-15, 15)
    hsv_saturation: tuple = (-30, 30)
    hsv_value: tuple = (-30, 30)
    brightness: tuple = (-0.3, 0.3)
    contrast: tuple = (0.7, 1.3)
    planckian_jitter_prob: float = 0.3  # NEW: Realistic lighting simulation
    planckian_temp_range: tuple = (3000, 15000)  # Color temperature (K)
    
    # Blur and noise (AlbumentationsX)
    blur_prob: float = 0.2
    advanced_blur_prob: float = 0.2  # NEW: Motion blur, defocus, glass blur
    advanced_blur_limit: tuple = (3, 7)
    noise_prob: float = 0.2
    noise_std_range: tuple = (0.1, 0.3)  # Gaussian noise std
    
    # Cutout/Erasing (AlbumentationsX)
    erasing_prob: float = 0.3  # NEW: Random erasing (replaces CoarseDropout)
    erasing_scale: tuple = (0.01, 0.1)  # Area fraction
    erasing_ratio: tuple = (0.3, 3.3)  # Aspect ratio
    
    # Feature space augmentation (for DINOv2 embeddings)
    feature_noise_std: float = 0.1
    feature_dropout: float = 0.1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            # Image sizes
            'query_img_size': self.query_img_size,
            'support_img_size': self.support_img_size,
            
            # Query augmentations (Ultralytics + AlbumentationsX)
            'query_mosaic_prob': self.query_mosaic_prob,
            'query_mixup_prob': self.query_mixup_prob,
            'query_mixup_alpha': self.query_mixup_alpha,
            
            # Support augmentations (AlbumentationsX)
            'support_mode': self.support_mode,
            
            # Temporal consistency
            'temporal_consistency_window': self.temporal_consistency_window,
            'video_frame_stride': self.video_frame_stride,
            'video_sequence_length': self.video_sequence_length,
            'video_sequence_overlap': self.video_sequence_overlap,
            
            # AlbumentationsX upgrades
            'planckian_jitter_prob': self.planckian_jitter_prob,
            'advanced_blur_prob': self.advanced_blur_prob,
            'erasing_prob': self.erasing_prob,
        }


def get_stage_config(stage: str = "stage1") -> AugmentationConfig:
    """
    Get stage-specific augmentation configuration.
    
    UPGRADED: Now uses AlbumentationsX (10-23x faster) with new augmentations:
    - PlanckianJitter: Realistic lighting simulation
    - AdvancedBlur: Motion blur, defocus, glass blur
    - Erasing: Random erasing (better than CoarseDropout)
    - D4 group: All 90° rotations + flips
    
    Stage 1: Base training with aggressive augmentation
    Stage 2: Few-shot training with medium augmentation
    Stage 3: Fine-tuning with weak augmentation
    
    Args:
        stage: One of "stage1", "stage2", "stage3"
        
    Returns:
        AugmentationConfig instance
    """
    if stage == "stage1":
        return AugmentationConfig(
            # Image sizes
            query_img_size=640,
            support_img_size=518,  # DINOv2 optimal size with reg4
            
            # Query augmentation - STRONG (Ultralytics + AlbumentationsX)
            query_mosaic_prob=1.0,  # Ultralytics Mosaic: +5-7% mAP
            query_mixup_prob=0.15,  # Ultralytics MixUp
            query_mixup_alpha=0.1,
            
            # Support augmentation - WEAK (AlbumentationsX)
            support_mode="weak",  # Preserve semantic content for DINOv2
            
            # Geometric - AGGRESSIVE (AlbumentationsX)
            flip_horizontal=0.5,
            flip_vertical=0.5,
            rotate_90=0.5,  # D4 group augmentation
            affine_scale=(0.8, 1.2),
            affine_rotate=(-15, 15),
            affine_translate=(-0.1, 0.1),
            affine_shear=(-5, 5),
            
            # Photometric - STRONG (AlbumentationsX)
            hsv_hue=(-15, 15),
            hsv_saturation=(-30, 30),
            hsv_value=(-30, 30),
            brightness=(-0.3, 0.3),
            contrast=(0.7, 1.3),
            planckian_jitter_prob=0.3,  # NEW: Realistic lighting
            planckian_temp_range=(3000, 15000),
            
            # Blur and noise - MODERATE (AlbumentationsX)
            blur_prob=0.2,
            advanced_blur_prob=0.2,  # NEW: Motion/defocus/glass blur
            advanced_blur_limit=(3, 7),
            noise_prob=0.2,
            noise_std_range=(0.1, 0.3),
            
            # Erasing - MODERATE (AlbumentationsX)
            erasing_prob=0.3,  # NEW: Random erasing
            erasing_scale=(0.01, 0.1),
            erasing_ratio=(0.3, 3.3),
            
            # Temporal consistency
            temporal_consistency_window=8,
            video_frame_stride=1,
            video_sequence_length=8,
            video_sequence_overlap=4,
            
            # Feature space augmentation
            feature_noise_std=0.1,
            feature_dropout=0.1,
        )
    
    elif stage == "stage2":
        return AugmentationConfig(
            # Image sizes
            query_img_size=640,
            support_img_size=518,  # DINOv2 optimal size
            
            # Query augmentation - MEDIUM (Reduced for episodic learning)
            query_mosaic_prob=0.5,  # Reduce mosaic (conflicts with episodic sampling)
            query_mixup_prob=0.0,   # Disable mixup (confuses prototype learning)
            query_mixup_alpha=0.1,
            
            # Support augmentation - WEAK (AlbumentationsX)
            support_mode="weak",  # Preserve prototypes
            
            # Geometric - MODERATE (AlbumentationsX)
            flip_horizontal=0.5,
            flip_vertical=0.0,  # Disable vertical flip
            rotate_90=0.3,  # Reduced D4
            affine_scale=(0.85, 1.15),
            affine_rotate=(-10, 10),
            affine_translate=(-0.05, 0.05),
            affine_shear=(0, 0),  # Disable shear
            
            # Photometric - MODERATE (AlbumentationsX)
            hsv_hue=(-10, 10),
            hsv_saturation=(-20, 20),
            hsv_value=(-20, 20),
            brightness=(-0.2, 0.2),
            contrast=(0.8, 1.2),
            planckian_jitter_prob=0.2,  # Reduced lighting simulation
            planckian_temp_range=(3000, 9000),
            
            # Blur and noise - LIGHT (AlbumentationsX)
            blur_prob=0.1,
            advanced_blur_prob=0.1,
            advanced_blur_limit=(3, 5),
            noise_prob=0.1,
            noise_std_range=(0.05, 0.2),
            
            # Erasing - LIGHT
            erasing_prob=0.1,
            erasing_scale=(0.01, 0.05),
            erasing_ratio=(0.3, 3.3),
            
            # Temporal consistency
            temporal_consistency_window=8,
            video_frame_stride=1,
            video_sequence_length=8,
            video_sequence_overlap=4,
            
            # Feature space augmentation
            feature_noise_std=0.05,
            feature_dropout=0.05,
        )
    
    else:  # stage3
        return AugmentationConfig(
            # Image sizes
            query_img_size=640,
            support_img_size=518,  # DINOv2 optimal size
            
            # Query augmentation - WEAK (Minimal for fine-tuning)
            query_mosaic_prob=0.3,  # Light mosaic
            query_mixup_prob=0.0,   # No mixup
            query_mixup_alpha=0.1,
            
            # Support augmentation - MINIMAL (AlbumentationsX)
            support_mode="weak",
            
            # Geometric - LIGHT (AlbumentationsX)
            flip_horizontal=0.3,
            flip_vertical=0.0,
            rotate_90=0.0,  # No rotation
            affine_scale=(0.9, 1.1),
            affine_rotate=(-5, 5),
            affine_translate=(0, 0),
            affine_shear=(0, 0),
            
            # Photometric - LIGHT (AlbumentationsX)
            hsv_hue=(-5, 5),
            hsv_saturation=(-10, 10),
            hsv_value=(-10, 10),
            brightness=(-0.1, 0.1),
            contrast=(0.9, 1.1),
            planckian_jitter_prob=0.0,  # Disable for stability
            planckian_temp_range=(3000, 9000),
            
            # Blur and noise - MINIMAL (AlbumentationsX)
            blur_prob=0.0,
            advanced_blur_prob=0.0,
            advanced_blur_limit=(3, 5),
            noise_prob=0.0,
            noise_std_range=(0.05, 0.1),
            
            # Erasing - DISABLED
            erasing_prob=0.0,
            erasing_scale=(0.01, 0.05),
            erasing_ratio=(0.3, 3.3),
            
            # Temporal consistency (longer window for stability)
            temporal_consistency_window=16,  # Longer consistency for fine-tuning
            video_frame_stride=1,
            video_sequence_length=8,
            video_sequence_overlap=4,
            
            # Feature space augmentation (minimal)
            feature_noise_std=0.0,
            feature_dropout=0.0,
        )


def get_yolov8_augmentation_params(stage: str = "stage1") -> Dict[str, Any]:
    """
    Get YOLOv8-compatible augmentation parameters for Ultralytics integration.
    
    NOTE: We use hybrid approach:
    - Ultralytics: Mosaic, MixUp, CopyPaste (optimized for YOLO)
    - AlbumentationsX: Color, blur, geometric transforms (10-23x faster)
    
    These params control only the Ultralytics augmentations.
    AlbumentationsX augmentations are applied separately via QueryAugmentation class.
    
    Args:
        stage: Training stage
        
    Returns:
        Dictionary of YOLOv8 augmentation parameters
    """
    config = get_stage_config(stage)
    
    # Convert to YOLOv8 format (0-1 scale for most params)
    yolo_params = {
        # Image size
        'imgsz': config.query_img_size,
        
        # HSV (handled by AlbumentationsX, but keep for compatibility)
        'hsv_h': config.hsv_hue[1] / 180.0,  # YOLOv8 uses 0-1 scale
        'hsv_s': config.hsv_saturation[1] / 100.0,
        'hsv_v': config.hsv_value[1] / 100.0,
        
        # Geometric (handled by AlbumentationsX, but keep for compatibility)
        'degrees': config.affine_rotate[1],  # Rotation degrees
        'translate': config.affine_translate[1],  # Translation fraction
        'scale': config.affine_scale[1] - 1.0,  # Scale factor (±value)
        'shear': config.affine_shear[1],  # Shear degrees
        'perspective': 0.0,  # Disable perspective (too aggressive)
        'flipud': config.flip_vertical,  # Vertical flip probability
        'fliplr': config.flip_horizontal,  # Horizontal flip probability
        
        # Mosaic/MixUp (Ultralytics implementation - KEEP ENABLED)
        'mosaic': config.query_mosaic_prob,  # Most impactful: +5-7% mAP
        'mixup': config.query_mixup_prob,
        
        # Copy-Paste (disable by default, requires segmentation masks)
        'copy_paste': 0.0,
        
        # Auto augmentation (disable, we use custom AlbumentationsX pipeline)
        'auto_augment': None,
        
        # Close mosaic in last N epochs (for convergence)
        'close_mosaic': 10,
    }
    
    return yolo_params


# Predefined configurations for quick access
STAGE1_CONFIG = get_stage_config("stage1")
STAGE2_CONFIG = get_stage_config("stage2")
STAGE3_CONFIG = get_stage_config("stage3")


def print_config_comparison():
    """Print comparison table of stage configurations."""
    configs = {
        "Stage 1": STAGE1_CONFIG,
        "Stage 2": STAGE2_CONFIG,
        "Stage 3": STAGE3_CONFIG
    }
    
    print("\n" + "="*80)
    print("AUGMENTATION CONFIGURATION COMPARISON")
    print("="*80)
    
    print(f"\n{'Parameter':<30} {'Stage 1':<15} {'Stage 2':<15} {'Stage 3':<15}")
    print("-"*80)
    
    params = [
        ('query_mosaic_prob', 'Mosaic Prob'),
        ('query_mixup_prob', 'MixUp Prob'),
        ('flip_horizontal', 'Flip H Prob'),
        ('flip_vertical', 'Flip V Prob'),
        ('rotate_90', 'D4 Rotation'),
        ('affine_rotate', 'Rotation Range'),
        ('hsv_hue', 'Hue Shift'),
        ('brightness', 'Brightness'),
        ('planckian_jitter_prob', 'Planckian Prob'),
        ('blur_prob', 'Blur Prob'),
        ('advanced_blur_prob', 'Advanced Blur'),
        ('erasing_prob', 'Erasing Prob'),
        ('support_img_size', 'Support Size'),
    ]
    
    for param_name, display_name in params:
        values = []
        for stage_name in ["Stage 1", "Stage 2", "Stage 3"]:
            val = getattr(configs[stage_name], param_name)
            if isinstance(val, tuple):
                values.append(f"{val[0]:.1f} to {val[1]:.1f}")
            else:
                values.append(str(val))
        
        print(f"{display_name:<30} {values[0]:<15} {values[1]:<15} {values[2]:<15}")
    
    print("="*80)


def print_stage_config(stage: str = "stage1"):
    """
    Print detailed configuration for a specific stage.
    
    Args:
        stage: One of "stage1", "stage2", "stage3"
    """
    config = get_stage_config(stage)
    
    print(f"\n{'='*80}")
    print(f"AUGMENTATION CONFIG - {stage.upper()}")
    print(f"{'='*80}\n")
    
    print("IMAGE SIZES:")
    print(f"  Query (YOLO):   {config.query_img_size}x{config.query_img_size}")
    print(f"  Support (DINOv2): {config.support_img_size}x{config.support_img_size}")
    
    print("\nQUERY AUGMENTATIONS (Ultralytics):")
    print(f"  Mosaic:         {config.query_mosaic_prob:.2f}")
    print(f"  MixUp:          {config.query_mixup_prob:.2f}")
    print(f"  MixUp Alpha:    {config.query_mixup_alpha:.2f}")
    
    print("\nSUPPORT AUGMENTATIONS:")
    print(f"  Mode:           {config.support_mode}")
    
    print("\nGEOMETRIC (AlbumentationsX):")
    print(f"  Flip H:         {config.flip_horizontal:.2f}")
    print(f"  Flip V:         {config.flip_vertical:.2f}")
    print(f"  Rotate 90° (D4): {config.rotate_90:.2f}")
    print(f"  Affine Scale:   {config.affine_scale[0]:.2f} to {config.affine_scale[1]:.2f}")
    print(f"  Affine Rotate:  {config.affine_rotate[0]:.1f}° to {config.affine_rotate[1]:.1f}°")
    print(f"  Affine Translate: {config.affine_translate[0]:.2f} to {config.affine_translate[1]:.2f}")
    print(f"  Affine Shear:   {config.affine_shear[0]:.1f}° to {config.affine_shear[1]:.1f}°")
    
    print("\nPHOTOMETRIC (AlbumentationsX):")
    print(f"  HSV Hue:        {config.hsv_hue[0]:.1f}° to {config.hsv_hue[1]:.1f}°")
    print(f"  HSV Saturation: {config.hsv_saturation[0]:.1f} to {config.hsv_saturation[1]:.1f}")
    print(f"  HSV Value:      {config.hsv_value[0]:.1f} to {config.hsv_value[1]:.1f}")
    print(f"  Brightness:     {config.brightness[0]:.2f} to {config.brightness[1]:.2f}")
    print(f"  Contrast:       {config.contrast[0]:.2f} to {config.contrast[1]:.2f}")
    print(f"  Planckian Jitter: {config.planckian_jitter_prob:.2f}")
    print(f"  Temp Range:     {config.planckian_temp_range[0]}K to {config.planckian_temp_range[1]}K")
    
    print("\nBLUR & NOISE (AlbumentationsX):")
    print(f"  Blur:           {config.blur_prob:.2f}")
    print(f"  Advanced Blur:  {config.advanced_blur_prob:.2f}")
    print(f"  Blur Limit:     {config.advanced_blur_limit[0]} to {config.advanced_blur_limit[1]}")
    print(f"  Noise:          {config.noise_prob:.2f}")
    print(f"  Noise Std:      {config.noise_std_range[0]:.2f} to {config.noise_std_range[1]:.2f}")
    
    print("\nERASING (AlbumentationsX):")
    print(f"  Probability:    {config.erasing_prob:.2f}")
    print(f"  Scale:          {config.erasing_scale[0]:.2f} to {config.erasing_scale[1]:.2f}")
    print(f"  Ratio:          {config.erasing_ratio[0]:.2f} to {config.erasing_ratio[1]:.2f}")
    
    print("\nTEMPORAL CONSISTENCY:")
    print(f"  Window:         {config.temporal_consistency_window} frames")
    print(f"  Frame Stride:   {config.video_frame_stride}")
    print(f"  Sequence Length: {config.video_sequence_length}")
    print(f"  Sequence Overlap: {config.video_sequence_overlap}")
    
    print("\nFEATURE SPACE:")
    print(f"  Noise Std:      {config.feature_noise_std:.2f}")
    print(f"  Dropout:        {config.feature_dropout:.2f}")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    # Print configuration comparison
    print_config_comparison()
    
    # Example usage
    print("\n\nExample: Getting Stage 2 configuration")
    config = get_stage_config("stage2")
    print(f"Mosaic probability: {config.query_mosaic_prob}")
    print(f"Support mode: {config.support_mode}")
    print(f"Temporal window: {config.temporal_consistency_window}")
    
    print("\n\nExample: YOLOv8 parameters for Stage 1")
    yolo_params = get_yolov8_augmentation_params("stage1")
    for key, val in yolo_params.items():
        print(f"  {key}: {val}")
