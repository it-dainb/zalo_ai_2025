"""
Contrastive Proposal Encoding (CPE) Loss
Paper: "FSCE: Few-Shot Object Detection via Contrastive Proposal Encoding" (CVPR 2021)
https://arxiv.org/abs/2103.05950

Key Features:
- Uses IoU-based augmentation for contrastive learning
- Natural positive/negative pairs from RPN proposals
- No extra augmentation needed
- +8.8% mAP improvement on PASCAL VOC 1-shot
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CPELoss(nn.Module):
    """
    Contrastive Proposal Encoding Loss (FSCE)
    
    Uses IoU scores to determine positive/negative proposal pairs:
    - Positive: same object, different IoU proposals (IoU > threshold)
    - Negative: different objects
    
    Formula:
        L_CPE = -1/|N_pos| * sum_{i in N_pos} log(
            sum_{j in P_i} exp(s_ij/tau) / sum_{k in N_i} exp(s_ik/tau)
        )
    
    where:
        P_i: positive proposals (same object, IoU > 0.5)
        N_i: negative proposals (different objects)
        s_ij: cosine similarity between proposals i and j
    
    Args:
        temperature (float): Temperature for similarity scaling (default: 0.1)
        pos_iou_threshold (float): IoU threshold for positive pairs (default: 0.5)
        neg_iou_threshold (float): IoU threshold for negative pairs (default: 0.3)
    """
    
    def __init__(
        self, 
        temperature=0.1, 
        pos_iou_threshold=0.5,
        neg_iou_threshold=0.3
    ):
        super().__init__()
        self.temperature = temperature
        self.pos_iou_threshold = pos_iou_threshold
        self.neg_iou_threshold = neg_iou_threshold
    
    def forward(self, proposal_features, proposal_ious, proposal_labels):
        """
        Forward pass
        
        Args:
            proposal_features (torch.Tensor): Proposal features [N, D]
            proposal_ious (torch.Tensor): IoU with ground truth [N]
            proposal_labels (torch.Tensor): GT labels for each proposal [N]
                -1 for background, >=0 for object classes
                
        Returns:
            torch.Tensor: CPE loss value
        """
        device = proposal_features.device
        
        # Filter out background proposals
        fg_mask = proposal_labels >= 0
        if fg_mask.sum() == 0:
            return torch.tensor(0.0, device=device)
        
        fg_features = proposal_features[fg_mask]
        fg_ious = proposal_ious[fg_mask]
        fg_labels = proposal_labels[fg_mask]
        
        num_fg = fg_features.shape[0]
        if num_fg < 2:
            return torch.tensor(0.0, device=device)
        
        # Normalize features
        fg_features = F.normalize(fg_features, p=2, dim=1)
        
        # Compute pairwise similarity: [N_fg, N_fg]
        similarity = torch.matmul(fg_features, fg_features.T) / self.temperature
        
        # Create positive mask: same class and high IoU
        label_mask = fg_labels.unsqueeze(0) == fg_labels.unsqueeze(1)  # [N_fg, N_fg]
        
        # IoU-based positive mask
        # Two proposals are positive if they have similar high IoU (within 0.2)
        iou_diff = torch.abs(fg_ious.unsqueeze(0) - fg_ious.unsqueeze(1))
        iou_mask = (fg_ious.unsqueeze(0) > self.pos_iou_threshold) & \
                   (fg_ious.unsqueeze(1) > self.pos_iou_threshold) & \
                   (iou_diff < 0.2)
        
        pos_mask = label_mask & iou_mask
        
        # Create negative mask: different class OR low IoU
        neg_mask = ~label_mask | (fg_ious.unsqueeze(1) < self.neg_iou_threshold)
        
        # Remove self-contrast
        self_mask = torch.eye(num_fg, dtype=torch.bool, device=device)
        pos_mask = pos_mask & ~self_mask
        neg_mask = neg_mask & ~self_mask
        
        # Compute loss for each anchor
        losses = []
        for i in range(num_fg):
            pos_indices = pos_mask[i]
            neg_indices = neg_mask[i]
            
            if pos_indices.sum() == 0 or neg_indices.sum() == 0:
                continue
            
            # Positive similarities
            pos_sim = similarity[i, pos_indices]
            
            # All similarities (positive + negative)
            all_indices = pos_indices | neg_indices
            all_sim = similarity[i, all_indices]
            
            # Log-sum-exp for numerical stability
            max_sim = all_sim.max()
            # Clamp max_sim to prevent overflow
            max_sim = torch.clamp(max_sim, min=-20.0, max=20.0)
            exp_all = torch.exp(all_sim - max_sim)
            
            # Loss: -log(sum(exp(pos)) / sum(exp(all)))
            # Clamp for numerical stability
            exp_pos_sum = torch.clamp(torch.exp(pos_sim - max_sim).sum(), min=1e-6, max=1e6)
            exp_all_sum = torch.clamp(exp_all.sum(), min=1e-6, max=1e6)
            loss_i = -torch.log(exp_pos_sum / exp_all_sum)
            # Clamp individual loss to prevent extreme values
            loss_i = torch.clamp(loss_i, max=10.0)
            losses.append(loss_i)
        
        if len(losses) == 0:
            return torch.tensor(0.0, device=device)
        
        return torch.stack(losses).mean()


class SimplifiedCPELoss(nn.Module):
    """
    Simplified CPE Loss for easier integration
    
    Uses only class labels for positive/negative pairs,
    without explicit IoU threshold checking.
    
    Args:
        temperature (float): Temperature scaling
    """
    
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, features, labels):
        """
        Forward pass
        
        Args:
            features (torch.Tensor): Features [N, D]
            labels (torch.Tensor): Labels [N], -1 for background
            
        Returns:
            torch.Tensor: Simplified CPE loss
        """
        # Filter foreground
        fg_mask = labels >= 0
        if fg_mask.sum() < 2:
            return torch.tensor(0.0, device=features.device)
        
        fg_features = features[fg_mask]
        fg_labels = labels[fg_mask]
        
        # Normalize
        fg_features = F.normalize(fg_features, p=2, dim=1)
        
        # Compute similarity
        similarity = torch.matmul(fg_features, fg_features.T) / self.temperature
        
        # Create masks
        label_eq = fg_labels.unsqueeze(0) == fg_labels.unsqueeze(1)
        self_mask = torch.eye(
            fg_features.shape[0], 
            dtype=torch.bool, 
            device=features.device
        )
        
        pos_mask = label_eq & ~self_mask
        neg_mask = ~label_eq
        
        # Compute loss
        losses = []
        for i in range(fg_features.shape[0]):
            if pos_mask[i].sum() == 0:
                continue
            
            pos_sim = similarity[i, pos_mask[i]]
            all_sim = similarity[i, pos_mask[i] | neg_mask[i]]
            
            # Numerical stability
            max_sim = all_sim.max()
            # Clamp max_sim to prevent overflow
            max_sim = torch.clamp(max_sim, min=-20.0, max=20.0)
            exp_pos_sum = torch.clamp(torch.exp(pos_sim - max_sim).sum(), min=1e-6, max=1e6)
            exp_all_sum = torch.clamp(torch.exp(all_sim - max_sim).sum(), min=1e-6, max=1e6)
            loss_i = -torch.log(exp_pos_sum / exp_all_sum)
            # Clamp individual loss to prevent extreme values
            loss_i = torch.clamp(loss_i, max=10.0)
            losses.append(loss_i)
        
        if len(losses) == 0:
            return torch.tensor(0.0, device=features.device)
        
        return torch.stack(losses).mean()
