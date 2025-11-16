"""
Centralized Logging Utility for YOLOv8n-RefDet
===============================================

Provides consistent logging configuration across training and evaluation scripts.

Key Features:
- Always saves logs to file (with meaningful, timestamped names)
- Conditionally prints to console based on --debug flag
- Rich, informative log filenames with experiment context
- Proper handler management to avoid duplicate logs

Usage:
    from src.training.logging_utils import setup_logger
    
    # In training script
    logger = setup_logger(
        name='training',
        log_dir='./checkpoints',
        stage=2,
        debug=args.debug,
        experiment_name='stage2_2way_4query'
    )
    
    # Logger automatically:
    # - Saves to: ./checkpoints/training_stage2_2way_4query_20250116_143022.log
    # - Prints to console only if debug=True
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str,
    log_dir: str | Path,
    stage: Optional[int] = None,
    debug: bool = False,
    experiment_name: Optional[str] = None,
    log_level: Optional[int] = None
) -> logging.Logger:
    """
    Setup logger with file output (always) and console output (if debug=True).
    
    Args:
        name: Logger name (e.g., 'training', 'evaluation', 'diagnostics')
        log_dir: Directory to save log files
        stage: Training stage number (1, 2, or 3) - included in filename
        debug: If True, enable DEBUG level logging to console and file. If False, use INFO level and file only.
        experiment_name: Optional experiment identifier for log filename
        log_level: Override logging level (default: DEBUG if debug=True, else INFO)
    
    Returns:
        Configured logger instance
    
    Example Log Filenames:
        - training_stage2_2way_4query_20250116_143022.log
        - evaluation_stage2_testset_20250116_150315.log
        - training_stage3_triplet_finetune_20250116_160045.log
    
    Notes:
        When debug=True, captures EVERYTHING at DEBUG level for detailed troubleshooting.
        This includes variable values, intermediate computations, and step-by-step flow.
    """
    # Create log directory if it doesn't exist
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine logging level: DEBUG if debug=True, otherwise INFO
    if log_level is None:
        log_level = logging.DEBUG if debug else logging.INFO
    
    # Generate meaningful log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_parts = [name]
    
    if stage is not None:
        filename_parts.append(f"stage{stage}")
    
    if experiment_name:
        filename_parts.append(experiment_name)
    
    filename_parts.append(timestamp)
    log_filename = "_".join(filename_parts) + ".log"
    log_path = log_dir / log_filename
    
    # Get or create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Always add file handler
    file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    file_formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(log_level)
    logger.addHandler(file_handler)
    
    # Conditionally add console handler
    if debug:
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            fmt='%(levelname)-8s | %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(log_level)
        logger.addHandler(console_handler)
        
        logger.info(f"{'='*70}")
        logger.info(f"DEBUG mode enabled - Capturing EVERYTHING (DEBUG level)")
        logger.info(f"Logs will be printed to console AND saved to file")
        logger.info(f"Log file: {log_path}")
        logger.info(f"Logging level: {logging.getLevelName(log_level)}")
        logger.info(f"{'='*70}")
    else:
        # Print to console ONCE to inform user where logs are saved
        print(f"{'='*70}")
        print(f"Logging to file: {log_path}")
        print(f"Use --debug flag to also print logs to console")
        print(f"{'='*70}")
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def get_experiment_name(args) -> str:
    """
    Generate meaningful experiment name from training arguments.
    
    Args:
        args: Parsed command-line arguments
    
    Returns:
        Experiment name string (e.g., '2way_4query', 'triplet_finetune')
    
    Example:
        args with n_way=2, n_query=4, stage=2 → '2way_4query'
        args with stage=3, use_triplet=True → 'triplet_finetune'
    """
    parts = []
    
    if hasattr(args, 'n_way') and hasattr(args, 'n_query'):
        parts.append(f"{args.n_way}way_{args.n_query}query")
    
    if hasattr(args, 'use_triplet') and args.use_triplet:
        parts.append('triplet')
    
    if hasattr(args, 'stage') and args.stage == 3:
        parts.append('finetune')
    
    if hasattr(args, 'batch_size'):
        parts.append(f"bs{args.batch_size}")
    
    return "_".join(parts) if parts else "default"


def log_experiment_config(logger: logging.Logger, args):
    """
    Log experiment configuration in a readable format.
    
    Args:
        logger: Logger instance
        args: Parsed command-line arguments
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"EXPERIMENT CONFIGURATION")
    logger.info(f"{'='*70}")
    
    # Group related arguments
    data_args = ['data_root', 'annotations', 'test_data_root', 'test_annotations']
    training_args = ['stage', 'epochs', 'batch_size', 'n_way', 'n_query', 'learning_rate']
    model_args = ['use_triplet', 'triplet_ratio', 'resume']
    cache_args = ['frame_cache_size', 'support_cache_size_mb', 'disable_cache']
    
    def log_group(title: str, arg_names: list):
        logger.info(f"\n{title}:")
        for arg_name in arg_names:
            if hasattr(args, arg_name):
                value = getattr(args, arg_name)
                logger.info(f"  {arg_name:30s}: {value}")
    
    log_group("Data Configuration", data_args)
    log_group("Training Configuration", training_args)
    log_group("Model Configuration", model_args)
    log_group("Caching Configuration", cache_args)
    
    logger.info(f"\n{'='*70}\n")


# Example usage for quick testing
if __name__ == "__main__":
    import argparse
    
    # Simulate command-line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--stage', type=int, default=2)
    parser.add_argument('--n_way', type=int, default=2)
    parser.add_argument('--n_query', type=int, default=4)
    args = parser.parse_args()
    
    # Setup logger
    experiment_name = get_experiment_name(args)
    logger = setup_logger(
        name='test',
        log_dir='./test_logs',
        stage=args.stage,
        debug=args.debug,
        experiment_name=experiment_name
    )
    
    # Test logging at all levels
    logger.debug("DEBUG: Variable x=42, y=3.14, performing calculation...")
    logger.info("INFO: Processing step 1/3")
    logger.warning("WARNING: Memory usage at 80%")
    logger.error("ERROR: Failed to load checkpoint")
    
    # Simulate detailed debugging
    logger.debug(f"DEBUG: Input shape: [32, 3, 640, 640]")
    logger.debug(f"DEBUG: Forward pass through encoder...")
    logger.debug(f"DEBUG: Feature maps: P2(80x80), P3(40x40), P4(20x20), P5(10x10)")
    
    print("\nTest complete! Check ./test_logs/ for the log file.")
