"""
Distribution Focal Loss (DFL)
Paper: "Generalized Focal Loss" (NeurIPS 2020)

Key Features:
- Treats bbox regression as classification over discrete bins
- Better fine-grained localization than direct regression
- Uncertainty modeling via distribution learning
- Complements IoU-based losses for small objects
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DFLoss(nn.Module):
    """
    Distribution Focal Loss for bbox regression
    
    Treats each bbox coordinate as a distribution over discrete bins,
    enabling fine-grained localization and uncertainty modeling.
    
    Formula:
        L_DFL(S) = -[(y_{i+1} - y) * log(S_i) + (y - y_i) * log(S_{i+1})]
        
    where:
        S: predicted distribution over bins
        y: continuous target value
        y_i, y_{i+1}: discrete bins surrounding y
    
    Args:
        reg_max (int): Maximum regression value (defines number of bins)
            Default: 16 (used in YOLOv8)
    """
    
    def __init__(self, reg_max=16):
        super().__init__()
        self.reg_max = reg_max
    
    def forward(self, pred_dist, target):
        """
        Forward pass
        
        Args:
            pred_dist (torch.Tensor): Predicted distribution [N, 4*reg_max]
                Each coordinate has reg_max bins (e.g., 16 bins for reg_max=16)
            target (torch.Tensor): Target bbox distances [N, 4] as [left, top, right, bottom]
                Values should be in range [0, reg_max-0.01] (distance from anchor to bbox edge)
                
        Returns:
            torch.Tensor: DFL loss value
        """
        # Early NaN/Inf check
        if torch.isnan(pred_dist).any() or torch.isinf(pred_dist).any():
            print(f"❌ DFLoss: Input pred_dist contains NaN/Inf!")
            return torch.tensor(0.0, device=pred_dist.device, requires_grad=True)
        
        if torch.isnan(target).any() or torch.isinf(target).any():
            print(f"❌ DFLoss: Input target contains NaN/Inf!")
            return torch.tensor(0.0, device=pred_dist.device, requires_grad=True)
        
        # Reshape pred_dist to [N, 4, reg_max]
        batch_size = pred_dist.shape[0]
        pred_dist = pred_dist.reshape(batch_size, 4, self.reg_max)
        
        # CRITICAL: Aggressive clamping BEFORE softmax to prevent gradient explosion
        # With AMP, even clamped values can cause gradient issues, so we use tighter bounds
        # Reduced from [-10, 10] to [-8, 8] for better AMP stability
        pred_dist = torch.clamp(pred_dist, min=-8.0, max=8.0)
        
        # Clamp targets to valid range [0, reg_max-1)
        # Using 1e-4 instead of 1e-6 for better numerical stability
        target = torch.clamp(target, min=0.0, max=self.reg_max - 1e-4)
        
        # Get integer bins surrounding target
        target_left = target.long()  # Floor
        target_right = target_left + 1  # Ceil
        
        # Get weights for linear interpolation
        weight_left = target_right.float() - target
        weight_right = target - target_left.float()
        
        # Calculate loss for each coordinate
        loss = torch.zeros(batch_size, device=pred_dist.device, dtype=pred_dist.dtype)
        for i in range(4):
            # Get distribution for coordinate i
            dist = pred_dist[:, i, :]  # [N, reg_max]
            
            # Apply softmax to get probabilities
            # Use float32 for softmax to prevent AMP precision issues
            original_dtype = dist.dtype
            dist_f32 = dist.float()
            dist_f32 = F.softmax(dist_f32, dim=-1)
            dist = dist_f32.to(original_dtype)
            
            # Check for NaN after softmax (can happen with extreme values even after clamping)
            if torch.isnan(dist).any():
                print(f"❌ DFLoss: NaN in softmax output for coordinate {i}!")
                return torch.tensor(0.0, device=pred_dist.device, requires_grad=True)
            
            # Get probabilities for target bins
            target_l = target_left[:, i]
            target_r = torch.clamp(target_right[:, i], max=self.reg_max - 1)
            
            # Cross-entropy with linear interpolation
            prob_left = dist[torch.arange(batch_size), target_l]
            prob_right = dist[torch.arange(batch_size), target_r]
            
            # Clamp probabilities to prevent log(0) - use larger epsilon for AMP stability
            # Increased from 1e-6 to 1e-5 for better mixed precision stability
            prob_left = torch.clamp(prob_left, min=1e-5, max=1.0)
            prob_right = torch.clamp(prob_right, min=1e-5, max=1.0)
            
            # DFL loss: negative log-likelihood weighted by distance to bins
            loss_left = -torch.log(prob_left) * weight_left[:, i]
            loss_right = -torch.log(prob_right) * weight_right[:, i]
            
            # Check for NaN in loss components
            if torch.isnan(loss_left).any() or torch.isnan(loss_right).any():
                print(f"❌ DFLoss: NaN in loss computation for coordinate {i}!")
                print(f"  prob_left range: [{prob_left.min():.6f}, {prob_left.max():.6f}]")
                print(f"  prob_right range: [{prob_right.min():.6f}, {prob_right.max():.6f}]")
                return torch.tensor(0.0, device=pred_dist.device, requires_grad=True)
            
            # Clamp individual losses to prevent outliers
            # Reduced from 20.0 to 15.0 to prevent gradient explosion
            loss_left = torch.clamp(loss_left, max=15.0)
            loss_right = torch.clamp(loss_right, max=15.0)
            
            loss += loss_left + loss_right
        
        # Return mean loss with aggressive clamping to prevent gradient explosion
        # Individual losses already clamped at 15.0, but we need to clamp mean too
        # to prevent mixed precision scaling from amplifying gradients
        loss_mean = loss.mean()
        
        # Final NaN check before returning
        if torch.isnan(loss_mean) or torch.isinf(loss_mean):
            print(f"❌ DFLoss: Final loss is NaN/Inf!")
            return torch.tensor(0.0, device=pred_dist.device, requires_grad=True)
        
        # Clamp at 12.0 (reduced from 15.0) for better AMP stability
        # This still allows learning from random init (~11.08) but prevents gradient explosion
        loss_mean = torch.clamp(loss_mean, max=12.0)
        
        return loss_mean
    
    def decode(self, pred_dist):
        """
        Decode distribution to continuous bbox distances
        
        Args:
            pred_dist (torch.Tensor): Predicted distribution [N, 4*reg_max]
            
        Returns:
            torch.Tensor: Decoded bbox distances [N, 4] as [left, top, right, bottom]
        """
        batch_size = pred_dist.shape[0]
        pred_dist = pred_dist.reshape(batch_size, 4, self.reg_max)
        
        # CRITICAL: Clamp pred_dist BEFORE softmax to prevent gradient explosion
        # Match forward pass clamping: [-8, 8] for AMP stability
        pred_dist = torch.clamp(pred_dist, min=-8.0, max=8.0)
        
        # Apply softmax in float32 for stability
        original_dtype = pred_dist.dtype
        pred_dist = pred_dist.float()
        pred_dist = F.softmax(pred_dist, dim=-1)
        pred_dist = pred_dist.to(original_dtype)
        
        # Expected value: sum of bin_index * probability
        bins = torch.arange(self.reg_max, dtype=torch.float32, device=pred_dist.device)
        decoded = (pred_dist * bins.view(1, 1, -1)).sum(dim=-1)
        
        return decoded
