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

# Flatten support images
N, K, C, H, W = support_images.shape
support_flat = support_images.reshape(N * K, C, H, W)

# Call set_reference_images
model.set_reference_images(
    support_flat,
    average_prototypes=True,
    n_way=N,
    n_support=K
)

print(f"\n=== After set_reference_images ===")
print(f"Cached support features (should be N={n_way} prototypes):")
for scale in ['p2', 'p3', 'p4', 'p5']:
    feat = model._cached_support_features[scale]
    print(f"  {scale}: {feat.shape}")

# Now call forward with class_ids
class_ids = torch.tensor([0, 1], device=device)
print(f"\n=== Calling forward with class_ids={class_ids} ===")
print(f"  use_cache=True, class_ids={class_ids}")

# Add instrumentation to check support_features_for_head
import sys
orig_forward = model.forward

def instrumented_forward(*args, **kwargs):
    # Set a breakpoint right after support_features_for_head is set
    import types
    
    # Call original forward
    result = orig_forward(*args, **kwargs)
    return result

outputs = model(
    query_image=query_images,
    use_cache=True,
    class_ids=class_ids,
)

print(f"\n=== Checking prototypes passed to detection head ===")
print("Looking at proto_sim output shape:")
for i, sim in enumerate(outputs['proto_sim']):
    B, K_out, H, W = sim.shape
    print(f"  Scale {i}: (B={B}, K={K_out}, H={H}, W={W})")
    if K_out != n_way:
        print(f"    ❌ ERROR: Expected K={n_way}, got K={K_out}")
        print(f"    This means the detection head received ({K_out}, dim) prototypes instead of ({n_way}, dim)")
