"""
Episode Calculator for Automatic n_episodes Computation

Automatically calculates optimal number of episodes per epoch based on:
- Dataset size (number of classes, frames per class)
- Episodic learning parameters (n_way, n_query)
- Augmentation multiplier (num_aug)
- Training efficiency considerations
"""

from typing import Dict, Optional
import math


def calculate_n_episodes(
    dataset_info: Dict,
    n_way: int,
    n_query: int,
    num_aug: int = 1,
    coverage_factor: float = 1.0,
    min_episodes: int = 10,
    max_episodes: int = 1000,
    verbose: bool = True,
) -> int:
    """
    Calculate optimal number of episodes per epoch.
    
    Strategy:
    1. Calculate total available samples (frames * num_aug)
    2. Calculate samples per episode (n_way * n_query)
    3. Compute episodes needed for full dataset coverage
    4. Apply coverage factor to control epoch length
    5. Clamp to reasonable bounds
    
    Args:
        dataset_info: Dict with keys:
            - 'num_classes': Total number of classes in dataset
            - 'total_frames': Total annotated frames across all classes
            - 'frames_per_class': List of frame counts per class (optional)
            - 'avg_frames_per_class': Average frames per class (optional)
        n_way: Number of classes per episode
        n_query: Number of query samples per class
        num_aug: Augmentation multiplier (each frame seen num_aug times)
        coverage_factor: Dataset coverage per epoch (1.0 = full coverage)
            - 1.0: Every sample seen once per epoch (full coverage)
            - 2.0: Every sample seen twice per epoch (2x coverage)
            - 0.5: Half of dataset seen per epoch (0.5x coverage)
        min_episodes: Minimum episodes (safety bound)
        max_episodes: Maximum episodes (computational bound)
        verbose: Print calculation details
        
    Returns:
        n_episodes: Optimal number of episodes per epoch
        
    Examples:
        >>> # Dataset: 10 classes, 1000 total frames, 5x augmentation
        >>> # Episode config: 2-way 4-query
        >>> dataset_info = {'num_classes': 10, 'total_frames': 1000}
        >>> n_episodes = calculate_n_episodes(dataset_info, n_way=2, n_query=4, num_aug=5)
        >>> # Result: ~625 episodes (ensures 1x coverage with augmentation)
        
        >>> # Faster training with 50% coverage
        >>> n_episodes = calculate_n_episodes(dataset_info, n_way=2, n_query=4, 
        ...                                    num_aug=5, coverage_factor=0.5)
        >>> # Result: ~313 episodes (0.5x coverage)
    """
    num_classes = dataset_info['num_classes']
    total_frames = dataset_info['total_frames']
    
    # Validate inputs
    if n_way > num_classes:
        raise ValueError(
            f"n_way ({n_way}) cannot exceed number of classes ({num_classes}). "
            f"Reduce n_way or use more classes."
        )
    
    if n_way < 1 or n_query < 1:
        raise ValueError(f"n_way and n_query must be >= 1 (got n_way={n_way}, n_query={n_query})")
    
    if num_aug < 1:
        raise ValueError(f"num_aug must be >= 1 (got {num_aug})")
    
    # Calculate average frames per class
    if 'frames_per_class' in dataset_info and dataset_info['frames_per_class']:
        avg_frames_per_class = sum(dataset_info['frames_per_class']) / len(dataset_info['frames_per_class'])
    elif 'avg_frames_per_class' in dataset_info:
        avg_frames_per_class = dataset_info['avg_frames_per_class']
    else:
        avg_frames_per_class = total_frames / num_classes
    
    # Calculate effective dataset size with augmentation
    total_samples = total_frames * num_aug
    
    # Calculate samples consumed per episode
    samples_per_episode = n_way * n_query
    
    # Calculate episodes for full coverage (with augmentation)
    # This is the number of episodes needed to see every augmented sample once
    episodes_for_full_coverage = math.ceil(total_samples / samples_per_episode)
    
    # Apply coverage factor
    n_episodes = int(episodes_for_full_coverage * coverage_factor)
    
    # Clamp to bounds
    n_episodes = max(min_episodes, min(n_episodes, max_episodes))
    
    # Print calculation details
    if verbose:
        print(f"\n{'='*70}")
        print("EPISODE CALCULATION")
        print(f"{'='*70}")
        print(f"\nDataset Information:")
        print(f"  Total classes: {num_classes}")
        print(f"  Total frames: {total_frames}")
        print(f"  Avg frames/class: {avg_frames_per_class:.1f}")
        print(f"  Augmentation multiplier: {num_aug}x")
        print(f"  Total samples (with aug): {total_samples:,}")
        
        print(f"\nEpisode Configuration:")
        print(f"  N-way: {n_way} classes/episode")
        print(f"  N-query: {n_query} samples/class")
        print(f"  Samples/episode: {samples_per_episode}")
        
        print(f"\nCalculation:")
        print(f"  Episodes for full coverage: {episodes_for_full_coverage:,}")
        print(f"  Coverage factor: {coverage_factor:.1f}x")
        print(f"  Calculated episodes: {int(episodes_for_full_coverage * coverage_factor):,}")
        print(f"  Bounded episodes: {n_episodes:,} (min={min_episodes}, max={max_episodes})")
        
        # Coverage statistics
        actual_coverage = (n_episodes * samples_per_episode) / total_samples
        print(f"\nResulting Coverage:")
        print(f"  Samples per epoch: {n_episodes * samples_per_episode:,} / {total_samples:,}")
        print(f"  Actual coverage: {actual_coverage:.2f}x")
        
        if actual_coverage < 0.5:
            print(f"  ⚠️  Low coverage! Each sample seen {actual_coverage:.2f}x per epoch")
        elif actual_coverage > 2.0:
            print(f"  ⚠️  High coverage! Each sample seen {actual_coverage:.2f}x per epoch (may overfit)")
        else:
            print(f"  ✓ Good coverage! Each sample seen ~{actual_coverage:.2f}x per epoch")
        
        # Check class sampling
        avg_times_class_sampled = (n_episodes * n_way) / num_classes
        print(f"\nClass Sampling:")
        print(f"  Each class sampled ~{avg_times_class_sampled:.1f}x per epoch")
        
        if avg_times_class_sampled < 0.5:
            print(f"  ⚠️  Warning: Some classes may not be sampled each epoch!")
        elif avg_times_class_sampled > 10:
            print(f"  ⚠️  Warning: Classes may be oversampled (risk of overfitting)")
        
        print(f"{'='*70}\n")
    
    return n_episodes


def calculate_n_episodes_from_dataset(
    dataset,
    n_way: int,
    n_query: int,
    num_aug: Optional[int] = None,
    coverage_factor: float = 1.0,
    min_episodes: int = 10,
    max_episodes: int = 1000,
    verbose: bool = True,
) -> int:
    """
    Calculate n_episodes directly from a RefDetDataset instance.
    
    Args:
        dataset: RefDetDataset instance
        n_way: Number of classes per episode
        n_query: Number of query samples per class
        num_aug: Override augmentation multiplier (if None, uses dataset.num_aug)
        coverage_factor: Dataset coverage per epoch (1.0 = full coverage)
        min_episodes: Minimum episodes
        max_episodes: Maximum episodes
        verbose: Print calculation details
        
    Returns:
        n_episodes: Optimal number of episodes
    """
    # Extract dataset info
    num_classes = len(dataset.classes)
    
    # Count frames per class
    frames_per_class = [len(data['frame_indices']) for data in dataset.class_data.values()]
    total_frames = sum(frames_per_class)
    
    # Use dataset's num_aug if not overridden
    if num_aug is None:
        num_aug = dataset.num_aug
    
    # Ensure num_aug is valid
    if not isinstance(num_aug, int) or num_aug < 1:
        raise ValueError(f"num_aug must be a positive integer, got {num_aug}")
    
    dataset_info = {
        'num_classes': num_classes,
        'total_frames': total_frames,
        'frames_per_class': frames_per_class,
        'avg_frames_per_class': total_frames / num_classes if num_classes > 0 else 0,
    }
    
    return calculate_n_episodes(
        dataset_info=dataset_info,
        n_way=n_way,
        n_query=n_query,
        num_aug=num_aug,  # Now guaranteed to be int
        coverage_factor=coverage_factor,
        min_episodes=min_episodes,
        max_episodes=max_episodes,
        verbose=verbose,
    )


def recommend_coverage_factor(
    stage: int,
    dataset_size: str = 'medium',
) -> float:
    """
    Recommend coverage factor based on training stage and dataset size.
    
    Args:
        stage: Training stage (1, 2, or 3)
        dataset_size: 'small' (<10 classes), 'medium' (10-50), 'large' (>50)
        
    Returns:
        coverage_factor: Recommended coverage factor
    """
    recommendations = {
        # Stage 1: Base training - need more coverage
        1: {'small': 2.0, 'medium': 1.5, 'large': 1.0},
        # Stage 2: Meta-learning - balanced coverage
        2: {'small': 1.5, 'medium': 1.0, 'large': 0.8},
        # Stage 3: Fine-tuning - less coverage to prevent overfitting
        3: {'small': 1.0, 'medium': 0.8, 'large': 0.5},
    }
    
    return recommendations.get(stage, {}).get(dataset_size, 1.0)


def auto_calculate_episodes(
    dataset,
    n_way: int,
    n_query: int,
    stage: int = 2,
    num_aug: Optional[int] = None,
    auto_coverage: bool = True,
    coverage_factor: Optional[float] = None,
    verbose: bool = True,
) -> int:
    """
    Fully automatic episode calculation with smart defaults.
    
    This is the recommended function for automatic configuration.
    
    Args:
        dataset: RefDetDataset instance
        n_way: Number of classes per episode
        n_query: Number of query samples per class
        stage: Training stage (1, 2, or 3) - affects coverage factor
        num_aug: Override augmentation multiplier (None = use dataset's)
        auto_coverage: Automatically determine coverage factor based on stage
        coverage_factor: Manual coverage factor (overrides auto_coverage)
        verbose: Print calculation details
        
    Returns:
        n_episodes: Optimal number of episodes
        
    Examples:
        >>> from src.datasets.refdet_dataset import RefDetDataset
        >>> from src.datasets.episode_calculator import auto_calculate_episodes
        >>> 
        >>> dataset = RefDetDataset(...)
        >>> n_episodes = auto_calculate_episodes(
        ...     dataset=dataset,
        ...     n_way=2,
        ...     n_query=4,
        ...     stage=2,  # Meta-learning stage
        ... )
    """
    # Determine dataset size
    num_classes = len(dataset.classes)
    if num_classes < 10:
        dataset_size = 'small'
    elif num_classes < 50:
        dataset_size = 'medium'
    else:
        dataset_size = 'large'
    
    # Determine coverage factor
    if coverage_factor is None and auto_coverage:
        coverage_factor = recommend_coverage_factor(stage, dataset_size)
        if verbose:
            print(f"Auto-selected coverage factor: {coverage_factor:.1f}x (stage={stage}, size={dataset_size})")
    elif coverage_factor is None:
        coverage_factor = 1.0
    
    return calculate_n_episodes_from_dataset(
        dataset=dataset,
        n_way=n_way,
        n_query=n_query,
        num_aug=num_aug,
        coverage_factor=coverage_factor,
        verbose=verbose,
    )


if __name__ == '__main__':
    # Example usage and testing
    print("Episode Calculator - Example Usage\n")
    
    # Example 1: Small dataset
    print("Example 1: Small UAV dataset")
    dataset_info = {
        'num_classes': 8,
        'total_frames': 800,  # ~100 frames/class
    }
    n_eps = calculate_n_episodes(
        dataset_info, n_way=2, n_query=4, num_aug=5, 
        coverage_factor=1.0, verbose=True
    )
    
    # Example 2: Large dataset with high augmentation
    print("\nExample 2: Large dataset with high augmentation")
    dataset_info = {
        'num_classes': 50,
        'total_frames': 10000,  # ~200 frames/class
    }
    n_eps = calculate_n_episodes(
        dataset_info, n_way=4, n_query=8, num_aug=10,
        coverage_factor=0.8, verbose=True
    )
    
    # Example 3: Stage-based recommendations
    print("\nExample 3: Stage-based automatic coverage")
    for stage in [1, 2, 3]:
        factor = recommend_coverage_factor(stage, 'medium')
        print(f"Stage {stage}: Recommended coverage factor = {factor:.1f}x")
