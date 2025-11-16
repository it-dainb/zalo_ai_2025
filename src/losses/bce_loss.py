"""
Binary Cross-Entropy Classification Loss
YOLOv8 default classification loss with optional class weighting

Key Features:
- Multi-label support (independent class probabilities)
- Numerical stability via BCEWithLogitsLoss
- Optional positive class weighting for imbalanced datasets
"""

import torch
import torch.nn as nn


class BCEClassificationLoss(nn.Module):
    """
    Binary Cross-Entropy Loss for classification
    
    Formula:
        L_BCE = -1/N * sum[y*log(p) + (1-y)*log(1-p)]
    
    Args:
        pos_weight (torch.Tensor, optional): Weight for positive class [num_classes]
            Use for class imbalance (e.g., pos_weight=torch.tensor([10.0]) for 1:10 ratio)
        reduction (str): 'mean', 'sum', or 'none'
    """
    
    def __init__(self, pos_weight=None, reduction='mean'):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight,
            reduction=reduction
        )
    
    def forward(self, pred_logits, targets):
        """
        Forward pass
        
        Args:
            pred_logits (torch.Tensor): Predicted logits [N, num_classes]
                (before sigmoid activation)
            targets (torch.Tensor): Target labels [N, num_classes]
                (binary, 0 or 1 for each class)
                
        Returns:
            torch.Tensor: BCE loss value
        """
        # Clamp logits to prevent numerical instability
        # BCEWithLogitsLoss uses log(1 + exp(x)), which can overflow for large x
        pred_logits = torch.clamp(pred_logits, min=-20.0, max=20.0)
        
        loss = self.bce(pred_logits, targets)
        
        # Clamp loss to prevent extreme values
        loss = torch.clamp(loss, max=10.0)
        
        return loss
