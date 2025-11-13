"""
Debug script to check gradient flow in the model.
"""
import torch
from src.models.yolov8n_refdet import YOLOv8nRefDet

# Create model
model = YOLOv8nRefDet(
    num_base_classes=1,
    fusion_module='cheaf',
    freeze_dinov3=True,
    freeze_dinov3_layers=6,
    freeze_yolo=False,
)

# Move to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.train()

# Check trainable parameters
print("="*70)
print("Trainable Parameters:")
print("="*70)
total_params = 0
trainable_params = 0
for name, param in model.named_parameters():
    total_params += param.numel()
    if param.requires_grad:
        trainable_params += param.numel()
        if 'support_encoder' in name or 'backbone' in name or 'fusion' in name or 'detection_head' in name:
            print(f"✓ {name}: {param.shape} (requires_grad={param.requires_grad})")

print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Frozen parameters: {total_params - trainable_params:,}")

# Create dummy batch
batch_size = 2
support_images = torch.randn(2, 3, 3, 512, 512).to(device)  # (N_classes, K_shots, C, H, W)
query_images = torch.randn(batch_size, 3, 640, 640).to(device)

# Forward pass
print("\n" + "="*70)
print("Forward Pass:")
print("="*70)

N, K, C, H, W = support_images.shape
support_flat = support_images.reshape(N * K, C, H, W)
model.set_reference_images(support_flat, average_prototypes=True)

outputs = model(query_image=query_images, mode='dual', use_cache=True)

# Create dummy loss
print("\nModel outputs keys:", list(outputs.keys()))

# Simple loss
if 'pred_bboxes' in outputs:
    loss = outputs['pred_bboxes'].sum()
else:
    loss = sum(v.sum() for v in outputs.values() if isinstance(v, torch.Tensor) and v.requires_grad)

print(f"Loss: {loss.item():.4f}")

# Backward
print("\n" + "="*70)
print("Backward Pass:")
print("="*70)
loss.backward()

# Check gradients
has_grad_count = 0
no_grad_count = 0
for name, param in model.named_parameters():
    if param.requires_grad:
        if param.grad is not None:
            has_grad_count += 1
            if param.grad.abs().sum() > 0:
                print(f"✓ {name}: has non-zero gradients")
        else:
            no_grad_count += 1
            print(f"❌ {name}: requires_grad=True but grad is None")

print(f"\nSummary:")
print(f"  Parameters with gradients: {has_grad_count}")
print(f"  Parameters missing gradients: {no_grad_count}")

if no_grad_count > 0:
    print("\n⚠️  WARNING: Some trainable parameters have no gradients!")
    print("   This will cause 'No inf checks were recorded' error with GradScaler")
