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
        
        # Check inputs for NaN/Inf
        if torch.isnan(features).any() or torch.isinf(features).any():
            print(f"⚠️ SupCon INPUT: features has NaN/Inf!")
            print(f"  features range: [{features.min():.6f}, {features.max():.6f}]")
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Normalize features with numerical stability
        # Use manual normalization with epsilon to avoid division by zero and gradient explosions
        feature_norms = torch.norm(features, p=2, dim=1, keepdim=True)
        # Clamp norms to avoid both zero and very small values that cause gradient explosions
        feature_norms = torch.clamp(feature_norms, min=1e-4)
        features = features / feature_norms
        
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
        
        # For numerical stability, use log_softmax which is more stable than log(sum(exp()))
        # First, apply mask to set non-contrastive elements to very negative value
        # so they don't contribute to softmax
        masked_similarity = torch.where(
            logits_mask.bool(),
            similarity_matrix,
            torch.full_like(similarity_matrix, -1e9)
        )
        
        # Compute log probabilities using numerically stable log_softmax
        log_prob = F.log_softmax(masked_similarity, dim=1)
        
        # Clamp to prevent extreme values
        log_prob = torch.clamp(log_prob, min=-50.0, max=50.0)
        
        print(f"🔍 SupCon Debug:")
        print(f"  Similarity matrix range: [{similarity_matrix.min():.6f}, {similarity_matrix.max():.6f}]")
        print(f"  log_prob range: [{log_prob.min():.6f}, {log_prob.max():.6f}]")
        
        # Compute mean of log-likelihood over positive pairs
        # Handle case where a sample has no positives
        mask_sum = mask.sum(1)
        
        # If no positive pairs exist for a sample, skip it entirely
        # (This can happen in few-shot scenarios with small batch sizes)
        if (mask_sum == 0).all():
            # No valid positive pairs in entire batch - return zero loss
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Replace zero mask_sum with 1 to avoid division by zero
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum
        
        # Loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        
        # Only average over samples that have positive pairs
        valid_samples = mask_sum > 1  # >1 because we set no-positive cases to 1
        if valid_samples.any():
            loss = loss[valid_samples].mean()
        else:
            # All samples have no positives - return zero loss
            loss = torch.tensor(0.0, device=device, requires_grad=True)
        
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
