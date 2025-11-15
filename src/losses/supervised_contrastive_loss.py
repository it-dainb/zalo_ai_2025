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
        
        # DEBUG: Check for issues in similarity matrix
        if torch.isnan(similarity_matrix).any() or torch.isinf(similarity_matrix).any():
            print(f"⚠️ SupCon: similarity_matrix has NaN/Inf!")
            print(f"  features range: [{features.min():.6f}, {features.max():.6f}]")
            print(f"  features norm: {torch.norm(features, dim=1).mean():.6f}")
        
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
        
        # DEBUG: Check positive pairs
        num_positives = mask.sum(1)
        print(f"🔍 SupCon Debug:")
        print(f"  Batch size: {batch_size}")
        print(f"  Unique labels: {torch.unique(labels).tolist()}")
        print(f"  Positive pairs per sample: {num_positives.tolist()}")
        
        # For numerical stability
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()
        
        print(f"  Similarity matrix range: [{similarity_matrix.min():.6f}, {similarity_matrix.max():.6f}]")
        print(f"  Logits range (after max subtraction): [{logits.min():.6f}, {logits.max():.6f}]")
        
        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        
        print(f"  exp_logits range: [{exp_logits.min():.6f}, {exp_logits.max():.6f}]")
        
        # Clamp for numerical stability (avoid log(0))
        exp_sum = torch.clamp(exp_logits.sum(1, keepdim=True), min=1e-6)
        
        print(f"  exp_sum range: [{exp_sum.min():.6f}, {exp_sum.max():.6f}]")
        
        log_prob = logits - torch.log(exp_sum)
        
        print(f"  log_prob range (before clamp): [{log_prob.min():.6f}, {log_prob.max():.6f}]")
        print(f"  log_prob has inf: {torch.isinf(log_prob).any().item()}")
        print(f"  log_prob has nan: {torch.isnan(log_prob).any().item()}")
        
        # ⚠️ CRITICAL: Clamp log_prob BEFORE using it to prevent -inf values
        # that cause NaN gradients during backprop
        log_prob = torch.clamp(log_prob, min=-50.0, max=50.0)
        
        print(f"  log_prob range (after clamp): [{log_prob.min():.6f}, {log_prob.max():.6f}]")
        
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
