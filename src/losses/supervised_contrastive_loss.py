"""
Supervised Contrastive Loss (SupCon)
Paper: "Supervised Contrastive Learning" (NeurIPS 2020)
https://arxiv.org/abs/2004.11362

Key Features:
- Enhances intra-class compactness
- Increases inter-class separability
- Better than cross-entropy for few-shot learning
- Used for DINOv2 prototype fine-tuning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss for prototype matching
    
    Formula:
        L_SupCon = sum_{i in I} [-1/|P(i)| * sum_{p in P(i)} log(
            exp(z_i · z_p / tau) / sum_{a in A(i)} exp(z_i · z_a / tau)
        )]
    
    where:
        z_i: normalized embedding of sample i
        P(i): set of positive samples (same class as i)
        A(i): set of all samples except i
        tau: temperature parameter
    
    Args:
        temperature (float): Temperature scaling parameter (default: 0.07)
            Lower values (0.01-0.05) → sharper, more confident
            Higher values (0.1-0.2) → softer, more exploratory
        base_temperature (float): Base temperature for normalization
    """
    
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
    
    def forward(self, features, labels, mask=None):
        """
        Forward pass
        
        Args:
            features (torch.Tensor): Feature embeddings [N, D]
                Should be L2-normalized
            labels (torch.Tensor): Class labels [N]
            mask (torch.Tensor, optional): Contrastive mask [N, N]
                If provided, overrides automatic mask from labels
                
        Returns:
            torch.Tensor: Supervised contrastive loss value
        """
        device = features.device
        batch_size = features.shape[0]
        
        # Normalize features
        features = F.normalize(features, p=2, dim=1)
        
        # Compute similarity matrix: [N, N]
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        
        # Create mask for positive pairs (same class)
        if mask is None:
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.T).float().to(device)
        
        # Mask out self-contrast cases (diagonal)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask
        
        # For numerical stability
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()
        
        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-7)
        
        # Compute mean of log-likelihood over positive pairs
        # Handle case where a sample has no positives
        mask_sum = mask.sum(1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum
        
        # Loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()
        
        return loss


class PrototypeContrastiveLoss(nn.Module):
    """
    Variant of SupCon for prototype-query matching
    
    Compares query features against class prototypes instead of
    all samples in batch. More efficient for few-shot detection.
    
    Args:
        temperature (float): Temperature scaling parameter
    """
    
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, query_features, prototypes, labels):
        """
        Forward pass
        
        Args:
            query_features (torch.Tensor): Query embeddings [N, D]
            prototypes (torch.Tensor): Class prototypes [K, D]
                where K is number of classes
            labels (torch.Tensor): Query labels [N] (indices into prototypes)
            
        Returns:
            torch.Tensor: Prototype contrastive loss
        """
        # Normalize
        query_features = F.normalize(query_features, p=2, dim=1)
        prototypes = F.normalize(prototypes, p=2, dim=1)
        
        # Compute similarity: [N, K]
        similarity = torch.matmul(query_features, prototypes.T) / self.temperature
        
        # Cross-entropy loss
        loss = F.cross_entropy(similarity, labels)
        
        return loss
