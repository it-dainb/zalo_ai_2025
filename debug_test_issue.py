import torch
from models.yolo_refdet import YOLORefDet

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create model
model = YOLORefDet(
    yolo_weights='yolov8n.pt',
    dinov3_model='vit_small_patch16_dinov3.lvd1689m',
    freeze_yolo=False,
    freeze_dinov3=True,
).to(device)

# Create test batch
batch_size = 2
n_way = 2
k_shot = 2

# Create support images: (N=2, K=2, 3, 256, 256)
support_images = torch.randn(n_way, k_shot, 3, 256, 256, device=device)
query_images = torch.randn(batch_size, 3, 640, 640, device=device)

print(f"\n=== Testing episodic support image handling ===")
print(f"Support images shape: {support_images.shape}")
print(f"N-way: {n_way}, K-shot: {k_shot}")

# Flatten support images as the trainer does
N, K, C, H, W = support_images.shape
support_flat = support_images.reshape(N * K, C, H, W)
print(f"Flattened support: {support_flat.shape}")

# Call set_reference_images as trainer does
model.set_reference_images(
    support_flat,
    average_prototypes=True,
    n_way=N,
    n_support=K
)

# Check cached features
print(f"\n=== Cached support features ===")
for scale, feat in model._cached_support_features.items():
    print(f"{scale}: {feat.shape}")

# Forward pass with class_ids
class_ids = torch.tensor([0, 1], device=device)
outputs = model(
    query_image=query_images,
    use_cache=True,
    class_ids=class_ids,
)

# Check output shapes
print(f"\n=== Model outputs ===")
for key, value in outputs.items():
    if isinstance(value, list):
        print(f"{key}: list of {len(value)} items")
        for i, v in enumerate(value):
            print(f"  [{i}]: {v.shape}")
    else:
        print(f"{key}: {value.shape if hasattr(value, 'shape') else value}")

# Extract proto_sim to check K
proto_sim = outputs['proto_sim']
print(f"\n=== Checking K dimension in proto_sim ===")
for i, sim in enumerate(proto_sim):
    B, K, H, W = sim.shape
    print(f"Scale {i}: (B={B}, K={K}, H={H}, W={W})")
    if K != n_way:
        print(f"  ❌ ERROR: Expected K={n_way}, got K={K}")
    else:
        print(f"  ✓ OK: K matches n_way")
