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
        # Reshape pred_dist to [N, 4, reg_max]
        batch_size = pred_dist.shape[0]
        pred_dist = pred_dist.reshape(batch_size, 4, self.reg_max)
        
        # CRITICAL: Clamp pred_dist BEFORE softmax to prevent gradient explosion
        # Softmax gradient can explode even with moderate inputs, causing NaN backprop
        pred_dist = torch.clamp(pred_dist, min=-10.0, max=10.0)
        
        # Clamp targets to valid range
        target = torch.clamp(target, min=0, max=self.reg_max - 1e-6)
        
        # Get integer bins surrounding target
        target_left = target.long()  # Floor
        target_right = target_left + 1  # Ceil
        
        # Get weights for linear interpolation
        weight_left = target_right.float() - target
        weight_right = target - target_left.float()
        
        # Calculate loss for each coordinate
        loss = 0
        for i in range(4):
            # Get distribution for coordinate i
            dist = pred_dist[:, i, :]  # [N, reg_max]
            
            # Apply softmax to get probabilities
            dist = F.softmax(dist, dim=-1)
            
            # Get probabilities for target bins
            target_l = target_left[:, i]
            target_r = torch.clamp(target_right[:, i], max=self.reg_max - 1)
            
            # Cross-entropy with linear interpolation
            prob_left = dist[torch.arange(batch_size), target_l]
            prob_right = dist[torch.arange(batch_size), target_r]
            
            # Clamp probabilities to prevent log explosion (was 1e-7, now 1e-6)
            prob_left = torch.clamp(prob_left, min=1e-6, max=1.0)
            prob_right = torch.clamp(prob_right, min=1e-6, max=1.0)
            
            # DFL loss: negative log-likelihood weighted by distance to bins
            loss_left = -torch.log(prob_left) * weight_left[:, i]
            loss_right = -torch.log(prob_right) * weight_right[:, i]
            
            # Clamp individual losses to prevent outliers
            loss_left = torch.clamp(loss_left, max=20.0)
            loss_right = torch.clamp(loss_right, max=20.0)
            
            loss += loss_left + loss_right
        
        # Return mean loss (individual losses already clamped at 20.0)
        # DO NOT clamp the final mean - it prevents learning from random init
        # Random init gives ~2.77 per coord × 4 coords = 11.08, which would be clamped at 10.0
        loss_mean = loss.mean()
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
        pred_dist = torch.clamp(pred_dist, min=-10.0, max=10.0)
        
        # Apply softmax
        pred_dist = F.softmax(pred_dist, dim=-1)
        
        # Expected value: sum of bin_index * probability
        bins = torch.arange(self.reg_max, dtype=torch.float32, device=pred_dist.device)
        decoded = (pred_dist * bins.view(1, 1, -1)).sum(dim=-1)
        
        return decoded
