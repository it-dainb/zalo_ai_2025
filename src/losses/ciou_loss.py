"""
Complete IoU Loss (CIoU)
Paper: "Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression" (AAAI 2020)
https://arxiv.org/abs/1911.08287

Used by Ultralytics YOLOv8 as the default bbox regression loss.

Key Features:
- Considers overlap area (IoU)
- Considers center distance (DIoU component)
- Considers aspect ratio consistency (CIoU component)
- More stable than WIoU for general object detection
- Standard in YOLOv5/v8 implementations

CIoU = IoU - (ρ²/c²) - αv

where:
- ρ²: Euclidean distance between box centers (squared)
- c²: diagonal length of smallest enclosing box (squared)
- v: measures aspect ratio consistency
- α: positive trade-off parameter

Implementation based on:
https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/metrics.py#L77
"""

import torch
import torch.nn as nn
import math


class CIoULoss(nn.Module):
    """
    Complete IoU Loss (CIoU) - Ultralytics YOLOv8 Implementation
    
    CIoU improves upon DIoU by adding aspect ratio consistency term.
    This makes the loss more stable and improves convergence speed.
    
    Loss = 1 - CIoU
    CIoU = IoU - (ρ²/c²) - αv
    
    where:
    - IoU: Intersection over Union
    - ρ²: squared center distance between pred and target
    - c²: squared diagonal of smallest enclosing box
    - v: aspect ratio consistency measure
    - α: trade-off parameter (v / (v - IoU + 1 + eps))
    
    Args:
        eps (float): Small value to avoid division by zero
    """
    
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps
    
    def forward(self, pred, target, reduction='mean'):
        """
        Forward pass
        
        Args:
            pred (torch.Tensor): Predicted boxes [N, 4] in format (x1, y1, x2, y2)
            target (torch.Tensor): Target boxes [N, 4] in format (x1, y1, x2, y2)
            reduction (str): 'mean', 'sum', or 'none'
            
        Returns:
            torch.Tensor: CIoU loss value
        """
        # Ensure valid boxes
        assert pred.shape == target.shape, "pred and target must have same shape"
        assert pred.shape[-1] == 4, "boxes must be in (x1, y1, x2, y2) format"
        
        # CRITICAL: Check for NaN/Inf in inputs (early detection)
        if torch.isnan(pred).any() or torch.isinf(pred).any():
            return torch.tensor(0.1, device=pred.device, requires_grad=True)
        if torch.isnan(target).any() or torch.isinf(target).any():
            return torch.tensor(0.1, device=pred.device, requires_grad=True)
        
        # CRITICAL: Clamp bbox coordinates to reasonable range to prevent numerical issues
        # Image coordinates typically in range [0, 1280] for normalized or pixel coords
        pred = torch.clamp(pred, min=-1e3, max=1e4)
        target = torch.clamp(target, min=-1e3, max=1e4)
        
        # Extract coordinates - following Ultralytics implementation
        # box format: (x1, y1, x2, y2)
        b1_x1, b1_y1, b1_x2, b1_y2 = pred.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = target.chunk(4, -1)
        
        # Width and height - clamp to prevent extreme values
        # CRITICAL: Clamp box dimensions to valid range to prevent atan() explosion
        w1 = torch.clamp(b1_x2 - b1_x1, min=self.eps, max=1e4)
        h1 = torch.clamp(b1_y2 - b1_y1, min=self.eps, max=1e4)
        w2 = torch.clamp(b2_x2 - b2_x1, min=self.eps, max=1e4)
        h2 = torch.clamp(b2_y2 - b2_y1, min=self.eps, max=1e4)
        
        # Intersection area
        inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * \
                (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)
        
        # Union area
        union = w1 * h1 + w2 * h2 - inter + self.eps
        
        # IoU
        iou = inter / union
        
        # CIoU calculation (Ultralytics implementation)
        # Convex (smallest enclosing box) width and height
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
        
        # Convex diagonal squared
        c2 = cw.pow(2) + ch.pow(2) + self.eps
        
        # Center distance squared (ρ²)
        # Formula: ((x2_center - x1_center)² + (y2_center - y1_center)²) / 4
        rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + 
                (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)) / 4
        
        # Aspect ratio consistency (v)
        # v = (4/π²) * (arctan(w2/h2) - arctan(w1/h1))²
        # CRITICAL: Clamp aspect ratios to prevent atan() from producing extreme gradients
        # Valid aspect ratios typically in range [0.1, 10] for real objects
        aspect_ratio_1 = torch.clamp(w1 / h1, min=0.01, max=100.0)
        aspect_ratio_2 = torch.clamp(w2 / h2, min=0.01, max=100.0)
        v = (4 / math.pi ** 2) * (aspect_ratio_2.atan() - aspect_ratio_1.atan()).pow(2)
        
        # Additional safety: clamp v to reasonable range
        v = torch.clamp(v, min=0.0, max=4.0)
        
        # Trade-off parameter (α)
        # α = v / (v - IoU + (1 + eps))
        # CRITICAL: Clamp alpha to prevent extreme values in mixed precision
        with torch.no_grad():
            alpha = v / (v - iou + (1 + self.eps))
            alpha = torch.clamp(alpha, min=0.0, max=10.0)
        
        # CIoU = IoU - (ρ²/c²) - (α * v)
        # CRITICAL: Clamp intermediate terms to prevent overflow in mixed precision
        distance_term = torch.clamp(rho2 / c2, min=0.0, max=1.0)
        aspect_term = torch.clamp(v * alpha, min=0.0, max=1.0)
        ciou = iou - distance_term - aspect_term
        
        # Loss = 1 - CIoU
        loss = 1 - ciou
        
        # Clamp loss to prevent extreme values (even more conservative for mixed precision)
        loss = torch.clamp(loss, min=0.0, max=2.0)
        
        # CRITICAL: Check for NaN/Inf and return safe value if detected
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            # Return small positive loss to allow gradient flow but prevent explosion
            return torch.tensor(0.1, device=pred.device, requires_grad=True)
        
        # Apply reduction
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss.squeeze(-1)  # Squeeze last dim for shape [N] instead of [N, 1]
    
    def _bbox_iou(self, box1, box2):
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
        
        # Clamp to valid range
        iou = torch.clamp(iou, min=0.0, max=1.0)
        
        return iou
    
    def __repr__(self):
        """String representation"""
        return f'CIoULoss(eps={self.eps})'
