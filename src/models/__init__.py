"""
YOLOv8n-RefDet: Reference-Based Detection for UAV Search-and-Rescue
===================================================================

A hybrid architecture combining:
- YOLOv8n backbone for real-time detection
- DINOv2 encoder for reference feature extraction
- PSALM (Pyramid Support Attention Layer Module) fusion
- Dual detection heads (standard + prototype matching)

Total parameters: ~9.6M
Target FPS: 25-30 FPS on Jetson Xavier NX
"""

__version__ = "0.2.0"
__author__ = "Zalo AI Challenge 2025"

from .dino_encoder import DINOSupportEncoder
from .yolov8_backbone import YOLOv8BackboneExtractor
from .psalm_fusion import PSALMFusion
from .dual_head import DualDetectionHead, PrototypeDetectionHead, StandardDetectionHead
from .yolov8n_refdet import YOLOv8nRefDet

__all__ = [
    "DINOSupportEncoder",
    "YOLOv8BackboneExtractor",
    "PSALMFusion",
    "DualDetectionHead",
    "PrototypeDetectionHead",
    "StandardDetectionHead",
    "YOLOv8nRefDet",
]
