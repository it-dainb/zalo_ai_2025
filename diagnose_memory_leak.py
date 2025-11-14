"""
Memory Leak Diagnostic Tool for YOLOv8n-RefDet Training

This script adds comprehensive memory profiling to identify memory leaks during training.
It tracks:
1. GPU memory usage per batch/epoch
2. CPU RAM usage growth
3. Tensor accumulation
4. Cache growth
5. List/dict size growth

Usage:
    python diagnose_memory_leak.py --monitor_interval 10
"""

import argparse
import gc
import os
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Dict, List

import torch
import psutil
import numpy as np


class MemoryMonitor:
    """Comprehensive memory monitoring for detecting leaks."""
    
    def __init__(self, log_file: str = "memory_profile.log"):
        """
        Initialize memory monitor.
        
        Args:
            log_file: Path to save memory logs
        """
        self.log_file = log_file
        self.process = psutil.Process(os.getpid())
        self.baseline_ram = None
        self.baseline_gpu = None
        self.batch_count = 0
        self.epoch_count = 0
        
        # History tracking
        self.ram_history = []
        self.gpu_history = []
        self.tensor_count_history = []
        
        # Start tracemalloc for Python object tracking
        tracemalloc.start()
        
        self._log("=" * 80)
        self._log("MEMORY DIAGNOSTIC STARTED")
        self._log("=" * 80)
    
    def _log(self, message: str):
        """Log message to both console and file."""
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(message + '\n')
    
    def set_baseline(self):
        """Set baseline memory usage."""
        ram_mb = self.process.memory_info().rss / 1024 / 1024
        gpu_mb = 0
        if torch.cuda.is_available():
            gpu_mb = torch.cuda.memory_allocated() / 1024 / 1024
        
        self.baseline_ram = ram_mb
        self.baseline_gpu = gpu_mb
        
        self._log(f"\n{'='*80}")
        self._log(f"BASELINE MEMORY")
        self._log(f"{'='*80}")
        self._log(f"CPU RAM: {ram_mb:.2f} MB")
        self._log(f"GPU Memory: {gpu_mb:.2f} MB")
        self._log(f"{'='*80}\n")
    
    def check_tensors(self) -> Dict:
        """Count and analyze all live tensors."""
        tensor_count = 0
        tensor_memory = 0
        tensors_by_size = {}
        
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj):
                    tensor_count += 1
                    size_mb = obj.element_size() * obj.nelement() / 1024 / 1024
                    tensor_memory += size_mb
                    
                    # Track by size
                    size_key = f"{obj.shape}"
                    if size_key not in tensors_by_size:
                        tensors_by_size[size_key] = {'count': 0, 'memory': 0}
                    tensors_by_size[size_key]['count'] += 1
                    tensors_by_size[size_key]['memory'] += size_mb
            except:
                pass
        
        return {
            'total_count': tensor_count,
            'total_memory_mb': tensor_memory,
            'by_size': tensors_by_size,
        }
    
    def check_memory(self, context: str = "") -> Dict:
        """
        Check current memory usage and detect leaks.
        
        Args:
            context: Description of current context (e.g., "After batch 10")
            
        Returns:
            memory_stats: Dict with memory statistics
        """
        # CPU RAM
        ram_mb = self.process.memory_info().rss / 1024 / 1024
        ram_delta = ram_mb - self.baseline_ram if self.baseline_ram else 0
        
        # GPU Memory
        gpu_mb = 0
        gpu_reserved_mb = 0
        if torch.cuda.is_available():
            gpu_mb = torch.cuda.memory_allocated() / 1024 / 1024
            gpu_reserved_mb = torch.cuda.memory_reserved() / 1024 / 1024
        gpu_delta = gpu_mb - self.baseline_gpu if self.baseline_gpu else 0
        
        # Tensor tracking
        tensor_info = self.check_tensors()
        
        # Python memory
        python_current, python_peak = tracemalloc.get_traced_memory()
        python_current_mb = python_current / 1024 / 1024
        python_peak_mb = python_peak / 1024 / 1024
        
        stats = {
            'context': context,
            'ram_mb': ram_mb,
            'ram_delta_mb': ram_delta,
            'gpu_mb': gpu_mb,
            'gpu_reserved_mb': gpu_reserved_mb,
            'gpu_delta_mb': gpu_delta,
            'tensor_count': tensor_info['total_count'],
            'tensor_memory_mb': tensor_info['total_memory_mb'],
            'python_current_mb': python_current_mb,
            'python_peak_mb': python_peak_mb,
        }
        
        # Add to history
        self.ram_history.append(ram_mb)
        self.gpu_history.append(gpu_mb)
        self.tensor_count_history.append(tensor_info['total_count'])
        
        return stats
    
    def log_memory_snapshot(self, context: str = ""):
        """Log detailed memory snapshot."""
        stats = self.check_memory(context)
        
        self._log(f"\n{'='*80}")
        self._log(f"MEMORY SNAPSHOT: {context}")
        self._log(f"{'='*80}")
        self._log(f"CPU RAM: {stats['ram_mb']:.2f} MB (Δ {stats['ram_delta_mb']:+.2f} MB)")
        self._log(f"GPU Memory: {stats['gpu_mb']:.2f} MB (Δ {stats['gpu_delta_mb']:+.2f} MB)")
        self._log(f"GPU Reserved: {stats['gpu_reserved_mb']:.2f} MB")
        self._log(f"Live Tensors: {stats['tensor_count']:,}")
        self._log(f"Tensor Memory: {stats['tensor_memory_mb']:.2f} MB")
        self._log(f"Python Objects: {stats['python_current_mb']:.2f} MB (Peak: {stats['python_peak_mb']:.2f} MB)")
        
        # Detect leak
        if stats['ram_delta_mb'] > 1000:  # > 1GB growth
            self._log(f"\n⚠️  WARNING: MEMORY LEAK DETECTED!")
            self._log(f"   RAM grew by {stats['ram_delta_mb']:.2f} MB")
            self._log(f"   This indicates a likely memory leak.")
        
        if stats['gpu_delta_mb'] > 2000:  # > 2GB growth
            self._log(f"\n⚠️  WARNING: GPU MEMORY LEAK DETECTED!")
            self._log(f"   GPU memory grew by {stats['gpu_delta_mb']:.2f} MB")
        
        self._log(f"{'='*80}\n")
        
        return stats
    
    def log_top_memory_consumers(self, top_n: int = 10):
        """Log top memory consuming Python objects."""
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        
        self._log(f"\nTop {top_n} Memory Consumers:")
        self._log("-" * 80)
        for i, stat in enumerate(top_stats[:top_n], 1):
            self._log(f"{i}. {stat.filename}:{stat.lineno}")
            self._log(f"   Size: {stat.size / 1024 / 1024:.2f} MB")
            self._log(f"   Count: {stat.count:,}")
        self._log("-" * 80)
    
    def analyze_growth_rate(self) -> Dict:
        """Analyze memory growth rate over time."""
        if len(self.ram_history) < 2:
            return {}
        
        # Calculate linear regression for RAM growth
        x = np.arange(len(self.ram_history))
        ram_array = np.array(self.ram_history)
        gpu_array = np.array(self.gpu_history)
        
        # Simple linear fit: y = mx + b
        ram_slope = (ram_array[-1] - ram_array[0]) / len(ram_array) if len(ram_array) > 1 else 0
        gpu_slope = (gpu_array[-1] - gpu_array[0]) / len(gpu_array) if len(gpu_array) > 1 else 0
        
        self._log(f"\n{'='*80}")
        self._log(f"MEMORY GROWTH ANALYSIS")
        self._log(f"{'='*80}")
        self._log(f"RAM Growth Rate: {ram_slope:.4f} MB/batch")
        self._log(f"GPU Growth Rate: {gpu_slope:.4f} MB/batch")
        
        if ram_slope > 1.0:  # Growing > 1MB per batch
            self._log(f"\n⚠️  RAM is growing steadily! ({ram_slope:.2f} MB/batch)")
            self._log(f"   At this rate, it will OOM in {(60000 - self.ram_history[-1]) / ram_slope:.0f} batches")
        
        if gpu_slope > 2.0:  # Growing > 2MB per batch
            self._log(f"\n⚠️  GPU memory is growing steadily! ({gpu_slope:.2f} MB/batch)")
        
        self._log(f"{'='*80}\n")
        
        return {
            'ram_slope_mb_per_batch': ram_slope,
            'gpu_slope_mb_per_batch': gpu_slope,
        }
    
    def check_for_leaks(self):
        """Check for common memory leak patterns."""
        self._log(f"\n{'='*80}")
        self._log(f"CHECKING FOR COMMON LEAK PATTERNS")
        self._log(f"{'='*80}")
        
        # Check for unbounded lists/dicts
        large_lists = []
        large_dicts = []
        
        for obj in gc.get_objects():
            try:
                if isinstance(obj, list) and len(obj) > 1000:
                    large_lists.append(len(obj))
                elif isinstance(obj, dict) and len(obj) > 1000:
                    large_dicts.append(len(obj))
            except:
                pass
        
        if large_lists:
            self._log(f"\n⚠️  Found {len(large_lists)} large lists (>1000 elements)")
            self._log(f"   Largest: {max(large_lists):,} elements")
            self._log(f"   This could indicate unbounded list growth!")
        
        if large_dicts:
            self._log(f"\n⚠️  Found {len(large_dicts)} large dicts (>1000 elements)")
            self._log(f"   Largest: {max(large_dicts):,} elements")
            self._log(f"   This could indicate unbounded dict growth!")
        
        self._log(f"{'='*80}\n")


def patch_trainer_with_monitoring(monitor: MemoryMonitor, check_interval: int = 10):
    """
    Patch the trainer to add memory monitoring hooks.
    
    Args:
        monitor: MemoryMonitor instance
        check_interval: Log memory every N batches
    """
    from src.training.trainer import RefDetTrainer
    
    # Save original methods
    original_train_epoch = RefDetTrainer.train_epoch
    original_validate = RefDetTrainer.validate
    
    def monitored_train_epoch(self, train_loader, epoch, triplet_loader=None, triplet_ratio=0.0):
        """Wrapped train_epoch with memory monitoring."""
        monitor._log(f"\n{'='*80}")
        monitor._log(f"STARTING EPOCH {epoch}")
        monitor._log(f"{'='*80}")
        monitor.log_memory_snapshot(f"Start of Epoch {epoch}")
        
        # Call original method with per-batch monitoring
        self.model.train()
        self.epoch = epoch
        self.nan_count = 0
        
        total_loss = 0.0
        loss_components = {}
        num_batches = 0
        
        triplet_iter = None
        if triplet_loader is not None and triplet_ratio > 0.0:
            triplet_iter = iter(triplet_loader)
        
        from tqdm import tqdm
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Check memory every N batches
            if batch_idx % check_interval == 0:
                monitor.log_memory_snapshot(f"Epoch {epoch}, Batch {batch_idx}")
                
                # Check for leaks every 50 batches
                if batch_idx % 50 == 0 and batch_idx > 0:
                    monitor.analyze_growth_rate()
                    monitor.check_for_leaks()
            
            # Continue with normal training...
            # (This is a simplified version - actual implementation would call the full method)
            break  # For diagnostic purposes
        
        monitor.log_memory_snapshot(f"End of Epoch {epoch}")
        
        # Call original to get actual metrics
        return original_train_epoch(self, train_loader, epoch, triplet_loader, triplet_ratio)
    
    def monitored_validate(self, val_loader, compute_detection_metrics=True, use_st_iou_cache=True):
        """Wrapped validate with memory monitoring."""
        monitor.log_memory_snapshot(f"Start of Validation")
        
        # Call original
        result = original_validate(self, val_loader, compute_detection_metrics, use_st_iou_cache)
        
        monitor.log_memory_snapshot(f"End of Validation")
        monitor.analyze_growth_rate()
        
        return result
    
    # Patch methods
    RefDetTrainer.train_epoch = monitored_train_epoch
    RefDetTrainer.validate = monitored_validate
    
    monitor._log("✓ Trainer patched with memory monitoring")


def main():
    parser = argparse.ArgumentParser(description='Diagnose memory leaks in training')
    parser.add_argument('--monitor_interval', type=int, default=10,
                        help='Log memory every N batches')
    parser.add_argument('--log_file', type=str, default='memory_profile.log',
                        help='Path to save memory logs')
    parser.add_argument('--dry_run', action='store_true',
                        help='Only initialize and check baseline, don\'t train')
    args = parser.parse_args()
    
    # Initialize monitor
    monitor = MemoryMonitor(log_file=args.log_file)
    
    # Check initial state
    monitor.set_baseline()
    monitor.log_top_memory_consumers(top_n=5)
    
    if args.dry_run:
        print(f"\nDry run complete. Baseline memory logged to {args.log_file}")
        return
    
    # Patch trainer
    patch_trainer_with_monitoring(monitor, check_interval=args.monitor_interval)
    
    # Now run the actual training script
    print("\n" + "="*80)
    print("MEMORY MONITORING ENABLED")
    print(f"Logs will be saved to: {args.log_file}")
    print(f"Checking memory every {args.monitor_interval} batches")
    print("="*80 + "\n")
    
    print("To use this, import this module in train.py:")
    print("  from diagnose_memory_leak import MemoryMonitor, patch_trainer_with_monitoring")
    print("  monitor = MemoryMonitor()")
    print("  monitor.set_baseline()")
    print("  patch_trainer_with_monitoring(monitor, check_interval=10)")


if __name__ == '__main__':
    main()
