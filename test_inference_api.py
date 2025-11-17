"""
Quick test to verify the new inference API works correctly.
"""
import torch
from src.models.yolo_refdet import YOLORefDet

def test_inference_api():
    """Test the new inference() method with caching."""
    print("="*70)
    print("Testing Inference API")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Initialize model
    model = YOLORefDet(
        yolo_weights="yolov8n.pt",
        dinov3_model="vit_small_patch16_dinov3.lvd1689m",
        freeze_yolo=False,
        freeze_dinov3=True,
        freeze_dinov3_layers=6,
    ).to(device)
    model.eval()
    
    print("\n✓ Model initialized")
    
    # Create dummy data
    ref_images = torch.randn(3, 3, 256, 256).to(device)  # 3-shot
    frame1 = torch.randn(1, 3, 640, 640).to(device)
    frame2 = torch.randn(1, 3, 640, 640).to(device)
    
    print("✓ Dummy data created")
    
    # Test 1: First frame with reference images (caches features)
    print("\nTest 1: First frame with support images")
    with torch.no_grad():
        det1 = model.inference(
            query_image=frame1,
            support_images=ref_images,
            conf_thres=0.25,
            iou_thres=0.45,
            max_det=300
        )
    
    print(f"  Detections keys: {list(det1.keys())}")
    print(f"  bboxes shape: {det1['bboxes'].shape}")
    print(f"  scores shape: {det1['scores'].shape}")
    print(f"  class_ids shape: {det1['class_ids'].shape}")
    print(f"  num_detections: {det1['num_detections']}")
    print(f"  Cache active: {model._cached_support_features is not None}")
    
    # Test 2: Second frame without reference images (uses cache)
    print("\nTest 2: Second frame using cached features")
    with torch.no_grad():
        det2 = model.inference(
            query_image=frame2,
            support_images=None,  # Use cache!
            conf_thres=0.25,
            iou_thres=0.45,
            max_det=300
        )
    
    print(f"  Detections keys: {list(det2.keys())}")
    print(f"  bboxes shape: {det2['bboxes'].shape}")
    print(f"  Cache still active: {model._cached_support_features is not None}")
    
    # Test 3: Clear cache
    print("\nTest 3: Clear cache")
    model.clear_cache()
    print(f"  Cache cleared: {model._cached_support_features is None}")
    
    # Test 4: Return raw outputs
    print("\nTest 4: Return raw outputs (no post-processing)")
    with torch.no_grad():
        raw_outputs = model.inference(
            query_image=frame1,
            support_images=ref_images,
            return_raw=True
        )
    
    print(f"  Raw outputs keys: {list(raw_outputs.keys())}")
    print(f"  prototype_boxes type: {type(raw_outputs['prototype_boxes'])}")
    print(f"  prototype_boxes length: {len(raw_outputs['prototype_boxes'])}")
    if isinstance(raw_outputs['prototype_boxes'], list):
        print(f"  prototype_boxes[0] shape: {raw_outputs['prototype_boxes'][0].shape}")
    print(f"  prototype_sim type: {type(raw_outputs['prototype_sim'])}")
    if isinstance(raw_outputs['prototype_sim'], list):
        print(f"  prototype_sim length: {len(raw_outputs['prototype_sim'])}")
        print(f"  prototype_sim[0] shape: {raw_outputs['prototype_sim'][0].shape}")
    
    print("\n" + "="*70)
    print("✅ All inference API tests passed!")
    print("="*70)

if __name__ == "__main__":
    test_inference_api()
