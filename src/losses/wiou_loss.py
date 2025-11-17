"""
Wise-IoU v3 Loss (WIoU)
Paper: "Wise-IoU: Bounding Box Regression Loss with Dynamic Focusing Mechanism" (2023)
https://arxiv.org/abs/2301.10051

Key Features:
- Dynamic focusing mechanism (beta parameter)
- Distance weighting (delta parameter)
- Outlier robustness
- Fastest convergence among IoU-based losses

Author's Implementation:
https://github.com/Instinct323/mod/blob/main/utils/loss.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WIoULoss(nn.Module):
    """
    Wise-IoU v3 Loss with dynamic focusing (author's implementation)
    
    Formula:
        L_WIoU = r * L_IoU
        r = beta * delta
        
    For monotonous=True (WIoU v3):
        beta = (L_IoU / iou_mean)^0.5
        
    For monotonous=False (WIoU v1):
        beta = L_IoU / iou_mean / (delta * alpha^(beta - delta))
        
    delta: exp(center_dist^2 / diagonal^2)
    
    Args:
        monotonous (bool): 
            - None: origin WIoU
            - True: monotonic FM (v3, recommended)
            - False: non-monotonic FM (v1)
        eps (float): Small value to avoid division by zero
        smooth_focusing (bool): Apply smoothing to reduce noise (recommended for few-shot)
    """
    
    momentum = 1e-2  # For moving average
    alpha = 1.7      # For non-monotonous mode
    delta = 2.7      # For non-monotonous mode
    
    def __init__(self, monotonous=True, eps=1e-7, smooth_focusing=False):
        super().__init__()
        self.monotonous = monotonous
        self.eps = eps
        self.smooth_focusing = smooth_focusing
        # Moving average of IoU loss (initialized to 1.0)
        self.register_buffer('iou_mean', torch.tensor(1.0))
    
    def forward(self, pred, target, reduction='mean'):
        """
        Forward pass
        
        Args:
            pred (torch.Tensor): Predicted boxes [N, 4] in format (x1, y1, x2, y2)
            target (torch.Tensor): Target boxes [N, 4] in format (x1, y1, x2, y2)
            reduction (str): 'mean', 'sum', or 'none'
            
        Returns:
            torch.Tensor: WIoU loss value
        """
        # Ensure valid boxes
        assert pred.shape == target.shape, "pred and target must have same shape"
        assert pred.shape[-1] == 4, "boxes must be in (x1, y1, x2, y2) format"
        
        # Calculate IoU
        iou = self.bbox_iou(pred, target)
        # Clamp IoU to prevent numerical issues
        iou = torch.clamp(iou, min=self.eps, max=1.0)
        iou_loss = 1 - iou
        
        # Update moving average of IoU loss (only during training)
        if self.training:
            with torch.no_grad():
                batch_iou_loss_mean = iou_loss.detach().mean()
                # Clamp to prevent extreme values
                batch_iou_loss_mean = torch.clamp(batch_iou_loss_mean, min=self.eps, max=1.0)
                self.iou_mean = self.iou_mean * (1 - self.momentum) + \
                               batch_iou_loss_mean * self.momentum
        
        # Calculate distance weighting (delta)
        delta = self._compute_delta(pred, target)
        # Clamp delta to prevent explosion
        delta = torch.clamp(delta, max=3.0)
        
        # Apply scaled loss with dynamic focusing
        loss = self._scaled_loss(delta * iou_loss, iou_loss)
        
        # Clamp final loss to prevent NaN propagation
        loss = torch.clamp(loss, max=10.0)
        
        # Apply reduction
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss
    
    def _compute_delta(self, pred, target):
        """
        Compute distance weighting factor
        
        delta = exp(center_distance^2 / diagonal^2)
        """
        # Get centers
        pred_cx = (pred[..., 0] + pred[..., 2]) / 2
        pred_cy = (pred[..., 1] + pred[..., 3]) / 2
        target_cx = (target[..., 0] + target[..., 2]) / 2
        target_cy = (target[..., 1] + target[..., 3]) / 2
        
        # Center distance squared
        center_dist_sq = (pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2
        
        # Smallest enclosing box (convex hull)
        c_x1 = torch.min(pred[..., 0], target[..., 0])
        c_y1 = torch.min(pred[..., 1], target[..., 1])
        c_x2 = torch.max(pred[..., 2], target[..., 2])
        c_y2 = torch.max(pred[..., 3], target[..., 3])
        
        # Diagonal squared of enclosing box
        c_w = c_x2 - c_x1
        c_h = c_y2 - c_y1
        diagonal_sq = c_w ** 2 + c_h ** 2 + self.eps
        
        # Distance weighting (detach diagonal to prevent gradient flow)
        delta = torch.exp(center_dist_sq / diagonal_sq.detach())
        
        return delta
    
    def _scaled_loss(self, loss, iou_loss):
        """
        Apply dynamic focusing mechanism
        
        Args:
            loss: Base loss (delta * iou_loss)
            iou_loss: IoU loss for computing beta
            
        Returns:
            Scaled loss with focusing
        """
        if self.monotonous is None:
            # Original WIoU (no focusing)
            return loss
        
        # Compute beta (focusing factor)
        beta = iou_loss.detach() / self.iou_mean
        
        if self.monotonous:
            # Monotonous focusing (WIoU v3): sqrt(beta)
            # Increases gradient for hard samples (high IoU loss)
            if self.smooth_focusing:
                # Apply smooth focusing: reduce variance while keeping benefits
                # Use tanh to bound the focusing factor, reducing extreme values
                # Formula: loss * (1 + 0.5 * tanh(beta - 1))
                # - When beta=1 (average sample): focusing_factor ≈ 1.0 (no change)
                # - When beta>1 (hard sample): focusing_factor ≈ 1.5 (modest increase)
                # - When beta<1 (easy sample): focusing_factor ≈ 0.5 (modest decrease)
                focusing_factor = 1.0 + 0.5 * torch.tanh(beta - 1.0)
                loss = loss * focusing_factor
            else:
                # Original WIoU v3: sqrt(beta) - more aggressive focusing
                loss = loss * beta.sqrt()
        else:
            # Non-monotonous focusing (WIoU v1): beta / divisor
            # Uses alpha and delta hyperparameters
            divisor = self.delta * torch.pow(self.alpha, beta - self.delta)
            loss = loss * beta / divisor
        
        return loss
    
    def bbox_iou(self, box1, box2):
        """
        Calculate IoU between two sets of boxes
        
        Args:
            box1 (torch.Tensor): [N, 4] boxes (x1, y1, x2, y2)
            box2 (torch.Tensor): [N, 4] boxes (x1, y1, x2, y2)
            
        Returns:
            torch.Tensor: [N] IoU values
        """
        # Intersection coordinates
        x1 = torch.max(box1[..., 0], box2[..., 0])
        y1 = torch.max(box1[..., 1], box2[..., 1])
        x2 = torch.min(box1[..., 2], box2[..., 2])
        y2 = torch.min(box1[..., 3], box2[..., 3])
        
        # Intersection area
        inter_w = torch.clamp(x2 - x1, min=0)
        inter_h = torch.clamp(y2 - y1, min=0)
        inter_area = inter_w * inter_h
        
        # Union area
        box1_area = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
        box2_area = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])
        union_area = box1_area + box2_area - inter_area + self.eps
        
        # IoU
        iou = inter_area / union_area
        
        return iou
    
    def __repr__(self):
        """String representation showing current iou_mean"""
        mode = 'Monotonous' if self.monotonous else 'Non-monotonous' if self.monotonous is False else 'Original'
        return f'WIoULoss(mode={mode}, iou_mean={self.iou_mean.item():.3f})'
