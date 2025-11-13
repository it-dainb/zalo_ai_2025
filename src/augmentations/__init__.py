"""
Data Augmentation Package for YOLOv8n-RefDet
Implements dual-path augmentation strategy for reference-based UAV detection.

UPGRADED: Now uses hybrid approach with Ultralytics + AlbumentationsX:
- Ultralytics (Mosaic/MixUp/CopyPaste) - optimized for YOLO detection
- AlbumentationsX (10-23x faster) - D4 symmetry, PlanckianJitter, AdvancedBlur, Erasing
- Best of both worlds: YOLO-specific augmentations + fast geometric/color transforms
"""

from .query_augmentation import (
    QueryAugmentation, 
    MosaicAugmentation, 
    MixUpAugmentation, 
    CopyPasteAugmentation
)
from .support_augmentation import SupportAugmentation, FeatureSpaceAugmentation
from .temporal_augmentation import TemporalConsistentAugmentation, VideoFrameSampler
from .augmentation_config import (
    AugmentationConfig, 
    get_stage_config, 
    get_yolov8_augmentation_params,
    print_stage_config,
    print_config_comparison,
)

__all__ = [
    # Augmentation classes
    'QueryAugmentation',
    'MosaicAugmentation',
    'MixUpAugmentation',
    'CopyPasteAugmentation',
    'SupportAugmentation',
    'FeatureSpaceAugmentation',
    'TemporalConsistentAugmentation',
    'VideoFrameSampler',
    
    # Configuration
    'AugmentationConfig',
    'get_stage_config',
    'get_yolov8_augmentation_params',
    'print_stage_config',
    'print_config_comparison',
]
