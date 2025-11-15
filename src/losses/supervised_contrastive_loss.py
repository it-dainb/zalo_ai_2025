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
    
    def __init__(self, temperature=0.07, base_temperature=0.07, debug_mode=False):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.debug_mode = debug_mode
    
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
            if self.debug_mode:
                print(f"⚠️ SupCon INPUT: features has NaN/Inf!")
                print(f"  features range: [{features.min():.6f}, {features.max():.6f}]")
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # CRITICAL FIX: Use F.cosine_similarity which is more numerically stable
        # than manual normalization + matmul in mixed precision training
        # PyTorch's cosine_similarity has built-in epsilon-stabilized implementation
        
        # Expand dimensions for pairwise similarity: [N, 1, D] and [1, N, D]
        features_i = features.unsqueeze(1)  # [N, 1, D]
        features_j = features.unsqueeze(0)  # [1, N, D]
        
        # Broadcast to [N, N, D]
        features_i_expanded = features_i.expand(batch_size, batch_size, -1)
        features_j_expanded = features_j.expand(batch_size, batch_size, -1)
        
        # Compute cosine similarity matrix: [N, N]
        similarity_matrix = F.cosine_similarity(features_i_expanded, features_j_expanded, dim=2, eps=1e-6)
        
        # Scale by temperature and clamp to prevent extreme logits
        similarity_matrix = torch.clamp(similarity_matrix / self.temperature, min=-10.0, max=10.0)
        
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
        
        # Count positive pairs per sample
        mask_sum = mask.sum(1)
        
        # If no positive pairs exist for any sample, return zero loss immediately
        # (This can happen in few-shot scenarios with small batch sizes)
        if (mask_sum == 0).all():
            if self.debug_mode:
                print(f"🔍 SupCon: No positive pairs found, returning zero loss")
            return torch.tensor(0.0, device=device, requires_grad=True)
        
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
        
        # Clamp to prevent extreme values that cause gradient explosion
        log_prob = torch.clamp(log_prob, min=-50.0, max=50.0)
        
        if self.debug_mode:
            print(f"🔍 SupCon Debug:")
            print(f"  Batch size: {batch_size}, Labels: {labels.squeeze().tolist()}")
            print(f"  Feature shape: {features.shape}")
            print(f"  Similarity matrix range: [{similarity_matrix.min():.6f}, {similarity_matrix.max():.6f}]")
            print(f"  log_prob range: [{log_prob.min():.6f}, {log_prob.max():.6f}]")
            print(f"  Positive pairs per sample: {mask_sum.tolist()}")
        
        # Replace zero mask_sum with 1 to avoid division by zero (these will be filtered out)
        mask_sum_safe = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        
        # Compute mean of log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum_safe
        
        # Loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        
        # Only average over samples that have positive pairs (mask_sum > 0, not > 1)
        valid_samples = mask_sum > 0
        if valid_samples.any():
            loss = loss[valid_samples].mean()
        else:
            # All samples have no positives - return zero loss
            loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        if self.debug_mode:
            print(f"  Final loss: {loss.item():.6f}")
        
        return loss


class PrototypeContrastiveLoss(nn.Module):
    """
    Variant of SupCon for prototype-query matching
    
    Compares query features against class prototypes instead of
    all samples in batch. More efficient for few-shot detection.
    
    Args:
        temperature (float): Temperature scaling parameter
    """
    
    def __init__(self, temperature=0.07, debug_mode=False):
        super().__init__()
        self.temperature = temperature
        self.debug_mode = debug_mode
    
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
        device = query_features.device
        
        # Check inputs for NaN/Inf
        if torch.isnan(query_features).any() or torch.isinf(query_features).any():
            if self.debug_mode:
                print(f"⚠️ PrototypeContrastive INPUT: query_features has NaN/Inf!")
            return torch.tensor(0.0, device=device, requires_grad=True)
        if torch.isnan(prototypes).any() or torch.isinf(prototypes).any():
            if self.debug_mode:
                print(f"⚠️ PrototypeContrastive INPUT: prototypes has NaN/Inf!")
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # CRITICAL FIX: Use cosine_similarity which is more numerically stable
        # than manual normalization + matmul in mixed precision training
        # PyTorch's cosine_similarity has built-in epsilon-stabilized implementation
        
        # Compute cosine similarity for each query-prototype pair: [N, K]
        # cosine_similarity expects [N, D] and [K, D], computes [N, K]
        N = query_features.shape[0]
        K = prototypes.shape[0]
        
        # Expand dimensions for broadcasting: [N, 1, D] and [1, K, D]
        query_expanded = query_features.unsqueeze(1)  # [N, 1, D]
        proto_expanded = prototypes.unsqueeze(0)      # [1, K, D]
        
        # Repeat to create [N, K, D] tensors
        query_repeated = query_expanded.expand(N, K, -1)  # [N, K, D]
        proto_repeated = proto_expanded.expand(N, K, -1)  # [N, K, D]
        
        # Compute cosine similarity along dimension 2 (feature dimension)
        # Result: [N, K] where each element is cosine similarity between query i and prototype j
        similarity = F.cosine_similarity(query_repeated, proto_repeated, dim=2, eps=1e-6)
        
        # Scale by temperature and clamp to prevent extreme logits
        similarity = torch.clamp(similarity / self.temperature, min=-10.0, max=10.0)
        
        if self.debug_mode:
            print(f"🔍 PrototypeContrastive Debug:")
            print(f"  Query features: {query_features.shape}")
            print(f"  Prototypes: {prototypes.shape}")
            print(f"  Labels: {labels.tolist()}")
            print(f"  Similarity range: [{similarity.min():.6f}, {similarity.max():.6f}]")
        
        # Cross-entropy loss (numerically stable)
        loss = F.cross_entropy(similarity, labels)
        
        if self.debug_mode:
            print(f"  Final loss: {loss.item():.6f}")
        
        return loss
