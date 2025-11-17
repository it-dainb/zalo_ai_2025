"""
Test SupCon loss with fixed model outputs
"""
import torch
import sys
sys.path.insert(0, 'src')

from models.yolov8n_refdet import YOLOv8nRefDet
from losses.combined_loss import ReferenceBasedDetectionLoss
from training.loss_utils import prepare_loss_inputs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}\n')

# Initialize model
print('Initializing model...')
model = YOLOv8nRefDet(
    yolo_weights='yolov8n.pt',
    nc_base=80,
    freeze_yolo=False,
    freeze_dinov3=True,
).to(device)

# Initialize loss function for Stage 2 with SupCon enabled
print('Initializing loss function with SupCon...')
loss_fn = ReferenceBasedDetectionLoss(
    stage=2,
    bbox_weight=2.0,
    cls_weight=1.0,
    supcon_weight=1.2,  # ✅ Enable SupCon
    cpe_weight=0.0,
    triplet_weight=0.0,
)

print(f'Stage 2 weights: {loss_fn.weights}\n')

# Create fake batch (4-way, 2-query per class)
print('Creating fake 4-way episode...')
N_way = 4
N_query = 2
batch_size = N_way * N_query  # 8 queries total

# Fake support images (4 classes, 1 shot each)
support_images = torch.randn(N_way, 1, 3, 256, 256).to(device)
query_images = torch.randn(batch_size, 3, 640, 640).to(device)

# Fake targets (2 boxes per query)
target_bboxes = [torch.tensor([[100, 100, 200, 200], [300, 300, 400, 400]]).float().to(device) for _ in range(batch_size)]
target_classes = [torch.tensor([i // N_query, i // N_query]).long().to(device) for i in range(batch_size)]
class_ids = torch.tensor([i // N_query for i in range(batch_size)]).long().to(device)

batch = {
    'support_images': support_images,
    'query_images': query_images,
    'target_bboxes': target_bboxes,
    'target_classes': target_classes,
    'class_ids': class_ids,
    'num_classes': N_way,
}

print(f'Batch: {N_way}-way, {N_query}-query, class_ids: {class_ids.tolist()}\n')

# Forward pass
print('Forward pass through model...')
N, K, C, H, W = support_images.shape
support_flat = support_images.reshape(N * K, C, H, W)

model.set_reference_images(support_flat, average_prototypes=True, n_way=N, n_support=K)

model.eval()
with torch.no_grad():
    model_outputs = model(
        query_image=query_images,
        mode='dual',
        use_cache=True,
        class_ids=class_ids,
    )

print('✅ Model outputs:')
for key in ['prototype_boxes', 'prototype_sim', 'query_features', 'support_prototypes']:
    if key in model_outputs:
        val = model_outputs[key]
        if isinstance(val, list):
            print(f'   {key}: List of {len(val)} tensors, first shape {val[0].shape}')
        else:
            print(f'   {key}: {val.shape}')
    else:
        print(f'   {key}: NOT FOUND ❌')

# Prepare loss inputs
print('\nPreparing loss inputs...')
loss_inputs = prepare_loss_inputs(
    model_outputs=model_outputs,
    batch=batch,
    stage=2,
)

print('✅ Loss inputs keys:')
for key in ['pred_bboxes', 'target_bboxes', 'query_features', 'support_prototypes', 'feature_labels']:
    if key in loss_inputs:
        val = loss_inputs[key]
        print(f'   {key}: {val.shape}')
    else:
        print(f'   {key}: NOT FOUND')

# Compute loss
print('\nComputing loss...')
# Remove diagnostic_data before passing to loss function
diagnostic_data = loss_inputs.pop('diagnostic_data', None)
losses = loss_fn(**loss_inputs)

print('\n✅ Loss components:')
for key, val in losses.items():
    if key != 'total_loss':
        print(f'   {key}: {val.item():.6f}')
print(f'   total_loss: {losses["total_loss"].item():.6f}')

# Check if SupCon loss is non-zero
supcon_loss = losses['supcon_loss'].item()
if supcon_loss > 0:
    print(f'\n🎉 SupCon loss is WORKING: {supcon_loss:.6f}')
else:
    print(f'\n❌ SupCon loss is still ZERO')

print('\n✅ Test complete!')
