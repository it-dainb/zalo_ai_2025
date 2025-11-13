# Complete Data Flow Analysis: Dataset → Model → Loss → Gradients

## Overview
This document provides a detailed analysis of how data flows through the entire training pipeline, from dataset to gradient backpropagation, with specific focus on integrating triplet training.

## Current Data Flow (Detection Only)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: DATA LOADING                                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ RefDetDataset.__getitem__(idx)                                             │
│ └─ Returns:                                                                 │
│    {                                                                        │
│      'query_frame': (H, W, 3) numpy,      ← Video frame                    │
│      'bboxes': (N, 4) numpy,              ← Object bounding boxes          │
│      'class_id': int,                     ← Class index                    │
│      'video_id': str,                     ← Class name                     │
│      'frame_idx': int,                                                      │
│      'support_images': List[(H,W,3)]      ← Reference images               │
│    }                                                                        │
│                                                                             │
│ ↓                                                                           │
│                                                                             │
│ RefDetCollator.__call__(batch)                                             │
│ ├─ Applies augmentations (query & support paths)                           │
│ ├─ Stacks images into tensors                                              │
│ └─ Returns:                                                                 │
│    {                                                                        │
│      'query_images': (B, 3, 640, 640),     ← Augmented query frames        │
│      'support_images': (N, K, 3, 518, 518), ← Augmented support images     │
│      'target_bboxes': List[(Ni, 4)],       ← Target boxes per image        │
│      'target_classes': List[(Ni,)],        ← Target classes per image      │
│      'class_ids': (B,),                    ← Class IDs                     │
│      'video_ids': List[str],               ← Video names                   │
│      'num_classes': int                    ← N-way                         │
│    }                                                                        │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: MODEL FORWARD PASS                                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ INPUT: batch['query_images'] (B, 3, 640, 640)                              │
│        batch['support_images'] (N, K, 3, 518, 518)                         │
│                                                                             │
│ ┌──────────────────────────────────────────────────────────────┐           │
│ │ Step 1: Support Encoder (DINOv2)                            │           │
│ ├──────────────────────────────────────────────────────────────┤           │
│ │ support_images (N*K, 3, 518, 518)                           │           │
│ │   ↓                                                          │           │
│ │ DINOv2SupportEncoder                                         │           │
│ │   ├─ ViT encoder → (N*K, 384)                               │           │
│ │   ├─ Average K shots → (N, 384)                             │           │
│ │   ├─ Multi-scale projection                                 │           │
│ │   └─ Returns: {                                             │           │
│ │       'p3': (N, 256),  ← Scale-specific prototypes          │           │
│ │       'p4': (N, 256),                                        │           │
│ │       'p5': (N, 256)                                         │           │
│ │     }                                                        │           │
│ └──────────────────────────────────────────────────────────────┘           │
│                                                                             │
│ ┌──────────────────────────────────────────────────────────────┐           │
│ │ Step 2: Query Encoder (YOLOv8 Backbone)                     │           │
│ ├──────────────────────────────────────────────────────────────┤           │
│ │ query_images (B, 3, 640, 640)                               │           │
│ │   ↓                                                          │           │
│ │ YOLOv8BackboneExtractor                                      │           │
│ │   ├─ C2f blocks → multi-scale features                      │           │
│ │   └─ Returns: {                                             │           │
│ │       'p3': (B, 64, 80, 80),    ← Low-level features        │           │
│ │       'p4': (B, 128, 40, 40),   ← Mid-level features        │           │
│ │       'p5': (B, 256, 20, 20)    ← High-level features       │           │
│ │     }                                                        │           │
│ └──────────────────────────────────────────────────────────────┘           │
│                                                                             │
│ ┌──────────────────────────────────────────────────────────────┐           │
│ │ Step 3: CHEAF Fusion                                           │           │
│ ├──────────────────────────────────────────────────────────────┤           │
│ │ query_features: {p3, p4, p5}                                │           │
│ │ support_features: {p3, p4, p5}                              │           │
│ │   ↓                                                          │           │
│ │ SCSFusionModule (for each scale):                           │           │
│ │   ├─ Cross-attention (query ⊗ support)                      │           │
│ │   ├─ Channel-wise correlation                               │           │
│ │   ├─ Spatial correlation                                    │           │
│ │   └─ Returns: {                                             │           │
│ │       'p3': (B, 256, 80, 80),                               │           │
│ │       'p4': (B, 512, 40, 40),                               │           │
│ │       'p5': (B, 512, 20, 20)                                │           │
│ │     }                                                        │           │
│ └──────────────────────────────────────────────────────────────┘           │
│                                                                             │
│ ┌──────────────────────────────────────────────────────────────┐           │
│ │ Step 4: Detection Head                                       │           │
│ ├──────────────────────────────────────────────────────────────┤           │
│ │ fused_features: {p3, p4, p5}                                │           │
│ │ prototypes: {p3, p4, p5}                                    │           │
│ │   ↓                                                          │           │
│ │ DualDetectionHead:                                           │           │
│ │   ├─ For each scale:                                        │           │
│ │   │   ├─ Conv layers                                        │           │
│ │   │   ├─ Box regression head → bboxes                       │           │
│ │   │   ├─ Classification head → logits                       │           │
│ │   │   └─ DFL head → distributions                           │           │
│ │   └─ Returns: {                                             │           │
│ │       'standard_boxes': List[(M1, 4)],   # Per scale       │           │
│ │       'standard_scores': List[(M1, C)],                     │           │
│ │       'prototype_boxes': List[(M2, 4)],                     │           │
│ │       'prototype_scores': List[(M2, K)],                    │           │
│ │       'dfl_distributions': List[(M, 4*17)]                  │           │
│ │     }                                                        │           │
│ └──────────────────────────────────────────────────────────────┘           │
│                                                                             │
│ OUTPUT: detections dict (NO intermediate features!) ← PROBLEM!             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: LOSS COMPUTATION                                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ prepare_loss_inputs(model_outputs, batch, stage)                           │
│ ├─ Matches predictions to targets                                          │
│ ├─ Prepares bbox/cls/dfl targets                                           │
│ └─ Returns: {                                                               │
│     'pred_bboxes': (N, 4),                                                  │
│     'pred_cls_logits': (N, C),                                              │
│     'pred_dfl_dist': (N, 4*17),                                             │
│     'target_bboxes': (N, 4),                                                │
│     'target_cls': (N, C),                                                   │
│     'target_dfl': (N, 4),                                                   │
│     # Missing for contrastive losses:                                      │
│     'query_features': None,         ← NOT AVAILABLE!                       │
│     'support_prototypes': None,     ← NOT AVAILABLE!                       │
│     'proposal_features': None       ← NOT AVAILABLE!                       │
│   }                                                                         │
│                                                                             │
│ ↓                                                                           │
│                                                                             │
│ ReferenceBasedDetectionLoss.forward(**loss_inputs)                         │
│ ├─ bbox_loss = WIoU(pred_bboxes, target_bboxes)          ✅ WORKS          │
│ ├─ cls_loss = BCE(pred_cls, target_cls)                  ✅ WORKS          │
│ ├─ dfl_loss = DFL(pred_dfl, target_dfl)                  ✅ WORKS          │
│ ├─ supcon_loss = SupCon(query_feat, support_proto)       ❌ NO DATA        │
│ ├─ cpe_loss = CPE(proposal_feat, labels)                 ❌ NO DATA        │
│ └─ triplet_loss = Triplet(anchor, pos, neg)              ❌ NO DATA        │
│                                                                             │
│ OUTPUT: total_loss = weighted sum (only bbox+cls+dfl active!)              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 4: GRADIENT BACKPROPAGATION (Current)                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ total_loss.backward()                                                       │
│   ↓                                                                         │
│ Gradients flow through:                                                     │
│   ├─ Detection Head ← from bbox_loss, cls_loss, dfl_loss                   │
│   ├─ CHEAF Fusion    ← from detection head                                   │
│   ├─ YOLOv8 Backbone ← from CHEAF fusion                                     │
│   └─ DINOv2 Encoder ← from CHEAF fusion (if not frozen)                      │
│                                                                             │
│ ❌ NO gradients from contrastive/triplet losses!                            │
│ ❌ Support encoder only gets gradients through detection path               │
└─────────────────────────────────────────────────────────────────────────────┘
