import torch

# Simulate detection head output
stride = 16
H, W = 40, 40  # Feature map size
img_size = 640

# Create anchor grid
grid_y, grid_x = torch.meshgrid(
    torch.arange(H, dtype=torch.float32),
    torch.arange(W, dtype=torch.float32),
    indexing='ij'
)
anchor_x = (grid_x + 0.5) * stride  # Pixel coordinates
anchor_y = (grid_y + 0.5) * stride

# Example: anchor at position (20, 20) -> pixel (328, 328)
anchor_idx_h, anchor_idx_w = 20, 20
anchor_px = anchor_x[anchor_idx_h, anchor_idx_w].item()
anchor_py = anchor_y[anchor_idx_h, anchor_idx_w].item()
print(f"Anchor grid position: ({anchor_idx_h}, {anchor_idx_w})")
print(f"Anchor pixel position: ({anchor_px}, {anchor_py})")

# Simulate raw prediction from detection head (clamped to [-10, 10])
# Option 1: Direct xyxy in stride-normalized space
raw_pred_option1 = torch.tensor([18.0, 18.0, 22.0, 22.0])  # Grid units
print(f"\n=== Option 1: Direct xyxy in grid units ===")
print(f"Raw prediction: {raw_pred_option1}")
decoded_1 = raw_pred_option1 * stride
print(f"After stride multiply: {decoded_1}")
print(f"Range: [{decoded_1.min()}, {decoded_1.max()}]")

# Option 2: ltrb offsets from anchor in stride units
raw_pred_option2 = torch.tensor([-2.0, -2.0, 2.0, 2.0])  # Offsets in stride units
print(f"\n=== Option 2: ltrb offsets from anchor (stride units) ===")
print(f"Raw prediction (offsets): {raw_pred_option2}")
decoded_2 = torch.tensor([
    anchor_px + raw_pred_option2[0] * stride,  # x1 = anchor_x + left_offset * stride
    anchor_py + raw_pred_option2[1] * stride,  # y1 = anchor_y + top_offset * stride
    anchor_px + raw_pred_option2[2] * stride,  # x2 = anchor_x + right_offset * stride
    anchor_py + raw_pred_option2[3] * stride,  # y2 = anchor_y + bottom_offset * stride
])
print(f"Decoded bbox: {decoded_2}")
print(f"BBox size: {decoded_2[2] - decoded_2[0]} x {decoded_2[3] - decoded_2[1]}")

# Option 3: ltrb offsets from anchor in pixel units  
raw_pred_option3 = torch.tensor([-32.0, -32.0, 32.0, 32.0])  # Offsets in pixels
print(f"\n=== Option 3: ltrb offsets from anchor (pixel units) ===")
print(f"Raw prediction (offsets): {raw_pred_option3}")
decoded_3 = torch.tensor([
    anchor_px + raw_pred_option3[0],  # x1 = anchor_x + left_offset
    anchor_py + raw_pred_option3[1],  # y1 = anchor_y + top_offset  
    anchor_px + raw_pred_option3[2],  # x2 = anchor_x + right_offset
    anchor_py + raw_pred_option3[3],  # y2 = anchor_y + bottom_offset
])
print(f"Decoded bbox: {decoded_3}")
print(f"BBox size: {decoded_3[2] - decoded_3[0]} x {decoded_3[3] - decoded_3[1]}")

print(f"\n=== Checking inference code behavior ===")
print(f"Inference code does: decoded_boxes = raw_pred * stride")
print(f"This matches Option 1: Direct xyxy in grid units")
print(f"BUT: With clamping [-10, 10], max value = 10 * 32 = 320 < 640!")
print(f"This can't reach the right/bottom edges of 640x640 image!")
