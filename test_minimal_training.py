"""Minimal test to isolate the CUDA index error."""
import torch
import sys
sys.path.insert(0, '/workspace/zalo_ai')

from src.training.loss_utils import assign_targets_to_anchors

# Simulate detection head outputs
batch_size = 2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
reg_max = 16

# Create fake detection outputs
proto_boxes_list = []
proto_sim_list = []
for stride, h_dim in zip([4, 8, 16, 32], [160, 80, 40, 20]):
    # Boxes: (B, 64, H, W) - FIXED: use 4*reg_max instead of 4*(reg_max+1)
    boxes = torch.randn(batch_size, 4*reg_max, h_dim, h_dim, device=device)
    proto_boxes_list.append(boxes)
    
    # Similarity: (B, K, H, W) where K=4 (n_way=4)
    sim = torch.randn(batch_size, 4, h_dim, h_dim, device=device)
    proto_sim_list.append(sim)

# Create fake targets
target_bboxes = [
    torch.tensor([[100, 100, 200, 200], [300, 300, 400, 400]], device=device, dtype=torch.float32),
    torch.tensor([[150, 150, 250, 250]], device=device, dtype=torch.float32),
]

# CRITICAL: Target classes must be in [0, K-1] for episodic learning
# If n_way=4, valid classes are [0, 1, 2, 3]
target_classes = [
    torch.tensor([0, 1], device=device, dtype=torch.long),  # Two objects in image 0
    torch.tensor([2], device=device, dtype=torch.long),      # One object in image 1
]

print(f"Device: {device}")
print(f"proto_sim shape: {[s.shape for s in proto_sim_list]}")
print(f"K (num_classes) from sim: {proto_sim_list[0].shape[1]}")
print(f"target_classes: {[tc.cpu().numpy() for tc in target_classes]}")

try:
    print("\nCalling assign_targets_to_anchors...")
    result = assign_targets_to_anchors(
        proto_boxes_list=proto_boxes_list,
        proto_sim_list=proto_sim_list,
        target_bboxes=target_bboxes,
        target_classes=target_classes,
        img_size=640,
        reg_max=reg_max,
    )
    
    print(f"\n✓ Success! Assigned {result[0].shape[0]} anchors")
    print(f"  matched_pred_bboxes: {result[0].shape}")
    print(f"  matched_pred_cls_logits: {result[1].shape}")
    print(f"  matched_pred_dfl_dist: {result[2].shape}")
    print(f"  target_cls_onehot: {result[6].shape}")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
