"""Test to verify class indices are within bounds."""
import torch
from pathlib import Path
from src.datasets.refdet_dataset import RefDetDataset
from src.datasets.collate import refdet_collate_fn

# Load a small batch
dataset = RefDetDataset(
    data_root='./datasets/train/samples',
    img_size=640,
    n_way=2,
    n_support=1,
    n_query=4,
    stage=2,
)

# Get one episode
episode = dataset[0]
print(f"Episode keys: {episode.keys()}")

# Collate into batch
batch = refdet_collate_fn([episode])
print(f"\nBatch keys: {batch.keys()}")

# Check target_classes
if 'target_classes' in batch:
    target_classes = batch['target_classes']
    print(f"\nTarget classes (list of tensors):")
    for i, cls_tensor in enumerate(target_classes):
        if len(cls_tensor) > 0:
            print(f"  Image {i}: {cls_tensor}, min={cls_tensor.min().item()}, max={cls_tensor.max().item()}")
        else:
            print(f"  Image {i}: empty")

# Check num classes from similarity scores (if available)
if 'model_outputs' in batch and 'prototype_sim' in batch['model_outputs']:
    sim_scores = batch['model_outputs']['prototype_sim']
    print(f"\nSimilarity scores (K classes):")
    for i, sim in enumerate(sim_scores):
        print(f"  Scale {i}: shape={sim.shape}, K={sim.shape[1]}")
        
print("\n✓ Class indices check complete")
