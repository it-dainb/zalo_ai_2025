"""
Triplet Loss for Base-Novel Balance
Paper: "YOLOv5-based Few-Shot Object Detection" (Remote Sensing 2024)

Key Features:
- Prevents catastrophic forgetting of base classes
- Maintains separation between base and novel classes
- Used in Stage 2-3 training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """
    Triplet Loss for maintaining base-novel class balance
    
    Prevents catastrophic forgetting by ensuring novel class features
    remain distinct from similar base class features.
    
    Formula:
        L_triplet = max(d(a, p) - d(a, n) + margin, 0)
    
    where:
        a: anchor (novel class feature)
        p: positive (same novel class)
        n: negative (confusable base class)
        d: distance function (euclidean or cosine)
    
    Args:
        margin (float): Margin for triplet loss (default: 0.3)
            0.2: Easy negatives (base and novel very different)
            0.3: Moderate similarity (default)
            0.5: Hard negatives (confusable classes)
        distance (str): Distance metric ('euclidean' or 'cosine')
        reduction (str): 'mean', 'sum', or 'none'
    """
    
    def __init__(self, margin=0.3, distance='euclidean', reduction='mean'):
        super().__init__()
        self.margin = margin
        self.distance = distance
        self.reduction = reduction
    
    def forward(self, anchor, positive, negative):
        """
        Forward pass
        
        Args:
            anchor (torch.Tensor): Anchor features [N, D] (novel class)
            positive (torch.Tensor): Positive features [N, D] (same novel class)
            negative (torch.Tensor): Negative features [N, D] (base class)
            
        Returns:
            torch.Tensor: Triplet loss value
        """
        if self.distance == 'euclidean':
            # Euclidean distance: L2 norm
            pos_dist = F.pairwise_distance(anchor, positive, p=2)
            neg_dist = F.pairwise_distance(anchor, negative, p=2)
        elif self.distance == 'cosine':
            # Cosine distance: 1 - cosine_similarity
            # Normalize features first
            anchor_norm = F.normalize(anchor, p=2, dim=1)
            positive_norm = F.normalize(positive, p=2, dim=1)
            negative_norm = F.normalize(negative, p=2, dim=1)
            
            # Cosine similarity
            pos_sim = (anchor_norm * positive_norm).sum(dim=1)
            neg_sim = (anchor_norm * negative_norm).sum(dim=1)
            
            # Convert to distance (0 = same, 2 = opposite)
            pos_dist = 1 - pos_sim
            neg_dist = 1 - neg_sim
        else:
            raise ValueError(f"Unknown distance metric: {self.distance}")
        
        # Triplet loss with margin
        loss = F.relu(pos_dist - neg_dist + self.margin)
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class BatchHardTripletLoss(nn.Module):
    """
    Batch Hard Triplet Loss (more efficient variant)
    
    Automatically mines hard triplets from a batch:
    - Hardest positive: max distance among same-class pairs
    - Hardest negative: min distance among different-class pairs
    
    Args:
        margin (float): Margin for triplet loss
        distance (str): Distance metric ('euclidean' or 'cosine')
    """
    
    def __init__(self, margin=0.3, distance='euclidean'):
        super().__init__()
        self.margin = margin
        self.distance = distance
    
    def forward(self, embeddings, labels):
        """
        Forward pass with automatic hard triplet mining
        
        Args:
            embeddings (torch.Tensor): Feature embeddings [N, D]
            labels (torch.Tensor): Class labels [N]
            
        Returns:
            torch.Tensor: Batch hard triplet loss
        """
        if self.distance == 'cosine':
            # Normalize for cosine distance
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute pairwise distances
        if self.distance == 'euclidean':
            # [N, N] pairwise L2 distance matrix
            pairwise_dist = torch.cdist(embeddings, embeddings, p=2)
        else:  # cosine
            # Cosine distance: 1 - cosine_similarity
            similarity = torch.matmul(embeddings, embeddings.T)
            pairwise_dist = 1 - similarity
        
        # Create masks
        labels = labels.unsqueeze(0)
        mask_pos = (labels == labels.T).float()  # Same class
        mask_neg = (labels != labels.T).float()  # Different class
        
        # Remove diagonal (self-comparison)
        mask_pos = mask_pos - torch.eye(mask_pos.size(0), device=embeddings.device)
        
        # For each anchor, find hardest positive and hardest negative
        losses = []
        for i in range(embeddings.size(0)):
            # Hardest positive: maximum distance among same class
            pos_dists = pairwise_dist[i] * mask_pos[i]
            if mask_pos[i].sum() > 0:
                hardest_pos_dist = pos_dists.max()
            else:
                continue  # Skip if no positives
            
            # Hardest negative: minimum distance among different class
            neg_dists = pairwise_dist[i] * mask_neg[i]
            # Set non-negative entries to large value
            neg_dists = neg_dists + (1 - mask_neg[i]) * 1e6
            if mask_neg[i].sum() > 0:
                hardest_neg_dist = neg_dists.min()
            else:
                continue  # Skip if no negatives
            
            # Triplet loss
            loss = F.relu(hardest_pos_dist - hardest_neg_dist + self.margin)
            losses.append(loss)
        
        if len(losses) == 0:
            return torch.tensor(0.0, device=embeddings.device)
        
        return torch.stack(losses).mean()


class AdaptiveTripletLoss(nn.Module):
    """
    Adaptive Triplet Loss with learnable margin
    
    The margin is learned during training, allowing the model to
    automatically adjust the separation requirement.
    
    Args:
        initial_margin (float): Initial margin value
        distance (str): Distance metric
    """
    
    def __init__(self, initial_margin=0.3, distance='euclidean'):
        super().__init__()
        self.margin = nn.Parameter(torch.tensor(initial_margin))
        self.distance = distance
    
    def forward(self, anchor, positive, negative):
        """
        Forward pass with learnable margin
        
        Args:
            anchor (torch.Tensor): Anchor features [N, D]
            positive (torch.Tensor): Positive features [N, D]
            negative (torch.Tensor): Negative features [N, D]
            
        Returns:
            torch.Tensor: Adaptive triplet loss
        """
        if self.distance == 'euclidean':
            pos_dist = F.pairwise_distance(anchor, positive, p=2)
            neg_dist = F.pairwise_distance(anchor, negative, p=2)
        else:  # cosine
            anchor_norm = F.normalize(anchor, p=2, dim=1)
            positive_norm = F.normalize(positive, p=2, dim=1)
            negative_norm = F.normalize(negative, p=2, dim=1)
            
            pos_sim = (anchor_norm * positive_norm).sum(dim=1)
            neg_sim = (anchor_norm * negative_norm).sum(dim=1)
            
            pos_dist = 1 - pos_sim
            neg_dist = 1 - neg_sim
        
        # Use learnable margin (ensure it's positive)
        margin = F.relu(self.margin)
        loss = F.relu(pos_dist - neg_dist + margin)
        
        return loss.mean()
