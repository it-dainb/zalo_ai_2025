"""
Example: Using Debug Logging in YOLOv8n-RefDet Trainer
======================================================

This example demonstrates how to use the debug logging features
for troubleshooting training issues.
"""

import torch
from pathlib import Path

from models.yolo_refdet import YOLORefDet
from src.losses.combined_loss import ReferenceBasedDetectionLoss
from src.training.trainer import RefDetTrainer

# Example 1: Basic Debug Setup
def example_basic_debug():
    """Enable debug logging with default settings."""
    
    # Create model, loss, optimizer (simplified)
    model = YOLORefDet(num_classes=1)
    loss_fn = ReferenceBasedDetectionLoss(stage=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Create trainer with debug mode enabled
    trainer = RefDetTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device='cuda',
        debug_mode=True,  # Enable debug logging
    )
    
    # Print model summary
    trainer.print_model_summary()
    
    return trainer


# Example 2: Debug Specific Issues
def example_debug_gradients():
    """Debug gradient flow issues."""
    
    model = YOLORefDet(num_classes=1)
    loss_fn = ReferenceBasedDetectionLoss(stage=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    trainer = RefDetTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device='cuda',
        debug_mode=True,
        gradient_clip_norm=0.5,  # Strong clipping to prevent explosions
    )
    
    # Log parameter statistics before training
    print("\n=== Initial Parameter State ===")
    trainer.log_parameter_statistics()
    
    return trainer


# Example 3: Get Model Summary Programmatically
def example_model_inspection():
    """Inspect model architecture and parameters."""
    
    model = YOLORefDet(num_classes=1)
    loss_fn = ReferenceBasedDetectionLoss(stage=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    trainer = RefDetTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device='cuda',
        debug_mode=False,  # Don't need full debug for inspection
    )
    
    # Get summary as dictionary
    summary = trainer.get_model_summary()
    
    print("\n=== Model Analysis ===")
    print(f"Total Parameters: {summary['total_parameters']:,}")
    print(f"Trainable: {summary['trainable_parameters']:,} "
          f"({summary['trainable_parameters']/summary['total_parameters']*100:.1f}%)")
    
    print("\n=== Trainable Modules ===")
    for module_name, params in summary['module_parameters'].items():
        if params['trainable'] > 0:
            pct = params['trainable'] / params['total'] * 100
            print(f"{module_name:20s}: {params['trainable']:>10,} ({pct:>5.1f}%)")
    
    print("\n=== Frozen Modules ===")
    for module_name, params in summary['module_parameters'].items():
        if params['frozen'] > 0:
            pct = params['frozen'] / params['total'] * 100
            print(f"{module_name:20s}: {params['frozen']:>10,} ({pct:>5.1f}%)")
    
    return summary


# Example 4: Custom Debug Logging
def example_custom_logging(trainer, batch, batch_idx):
    """Add custom debug logging during training."""
    
    if not trainer.debug_mode:
        return
    
    # Log custom information about batch
    if batch_idx == 0:
        print("\n=== Custom Batch Analysis ===")
        
        # Analyze support set diversity
        if 'support_images' in batch:
            support = batch['support_images']  # (N, K, 3, H, W)
            N, K = support.shape[:2]
            
            print(f"Episode Configuration:")
            print(f"  N-way: {N} classes")
            print(f"  K-shot: {K} support images per class")
            
            # Compute diversity metrics
            for class_idx in range(N):
                class_supports = support[class_idx]  # (K, 3, H, W)
                mean_intensity = class_supports.mean().item()
                std_intensity = class_supports.std().item()
                print(f"  Class {class_idx}: mean={mean_intensity:.3f}, std={std_intensity:.3f}")
        
        # Analyze query set
        if 'query_images' in batch:
            query = batch['query_images']  # (B, 3, H, W)
            print(f"\nQuery Set:")
            print(f"  Batch size: {query.shape[0]}")
            print(f"  Image stats: mean={query.mean().item():.3f}, std={query.std().item():.3f}")


# Example 5: Debugging Training Loop
def example_debug_training():
    """Complete example with debug logging in training loop."""
    
    # Setup (simplified - use real dataloaders in practice)
    model = YOLORefDet(num_classes=1)
    loss_fn = ReferenceBasedDetectionLoss(stage=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    trainer = RefDetTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device='cuda',
        debug_mode=True,
        checkpoint_dir='./debug_checkpoints',
    )
    
    print("\n=== Debug Training Configuration ===")
    print(f"Debug mode: {trainer.debug_mode}")
    print(f"Debug log file: {trainer.checkpoint_dir / 'training_debug.log'}")
    print(f"Mixed precision: {trainer.mixed_precision}")
    print(f"Gradient clipping: {trainer.gradient_clip_norm}")
    
    # Model summary
    trainer.print_model_summary()
    
    # Training loop would go here
    # trainer.train(train_loader, val_loader, num_epochs=10)
    
    return trainer


# Example 6: Analyzing Debug Logs
def analyze_debug_logs(log_file='./checkpoints/training_debug.log'):
    """Parse and analyze debug logs."""
    
    log_path = Path(log_file)
    if not log_path.exists():
        print(f"Log file not found: {log_file}")
        return
    
    print(f"\n=== Analyzing Debug Log: {log_file} ===\n")
    
    # Statistics
    total_lines = 0
    nan_warnings = 0
    grad_issues = 0
    loss_logs = 0
    
    with open(log_path, 'r') as f:
        for line in f:
            total_lines += 1
            
            if 'NaN' in line or 'Inf' in line:
                nan_warnings += 1
            if 'gradient' in line.lower():
                grad_issues += 1
            if 'Loss Components' in line:
                loss_logs += 1
    
    print(f"Total log lines: {total_lines:,}")
    print(f"NaN/Inf warnings: {nan_warnings}")
    print(f"Gradient mentions: {grad_issues}")
    print(f"Loss component logs: {loss_logs}")
    
    # Extract last few loss values
    print("\n=== Recent Loss Values ===")
    with open(log_path, 'r') as f:
        lines = f.readlines()
        
    for line in reversed(lines[-200:]):  # Check last 200 lines
        if 'Total Loss:' in line:
            print(line.strip())


if __name__ == '__main__':
    print("YOLOv8n-RefDet Debug Logging Examples")
    print("=" * 70)
    
    # Run examples
    print("\n### Example 1: Basic Debug Setup ###")
    trainer1 = example_basic_debug()
    
    print("\n### Example 3: Model Inspection ###")
    summary = example_model_inspection()
    
    print("\n### Example 6: Log Analysis ###")
    # analyze_debug_logs()  # Uncomment if log file exists
    
    print("\n" + "=" * 70)
    print("Examples completed! Run with actual data for full functionality.")
    print("=" * 70)
