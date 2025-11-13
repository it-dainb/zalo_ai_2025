"""
Reference-Based Detection Dataset for Few-Shot UAV Object Detection
Handles video frame extraction, support image loading, and episodic sampling.

Optimizations:
- LRU cache for support images (all support images cached)
- LRU cache for video frames (configurable size)
- Optional disk cache for DINOv2 features
- Cache statistics tracking
"""

import json
import cv2
import numpy as np
from torch.utils.data import Dataset, Sampler
from pathlib import Path
from typing import Dict, Optional, Tuple
import random
from collections import OrderedDict
from functools import lru_cache
import pickle
import hashlib


class LRUCache:
    """Simple LRU (Least Recently Used) Cache implementation."""
    
    def __init__(self, capacity: int):
        """
        Args:
            capacity: Maximum number of items to cache
        """
        self.capacity = capacity
        self.cache = OrderedDict()
        self.hits = 0
        self.misses = 0
    
    def get(self, key):
        """Get item from cache, None if not found."""
        if key in self.cache:
            self.hits += 1
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        self.misses += 1
        return None
    
    def put(self, key, value):
        """Put item in cache, evict LRU if needed."""
        if key in self.cache:
            # Update existing
            self.cache.move_to_end(key)
        else:
            # Add new
            if len(self.cache) >= self.capacity:
                # Evict LRU (first item)
                self.cache.popitem(last=False)
            self.cache[key] = value
    
    def clear(self):
        """Clear all cached items."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        return {
            'size': len(self.cache),
            'capacity': self.capacity,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
        }


class VideoFrameExtractor:
    """Efficient video frame extraction with LRU caching and prefetching."""
    
    def __init__(self, video_path: str, cache_size: int = 500):
        """
        Args:
            video_path: Path to video file
            cache_size: Number of frames to cache (default 500 frames ~= 300MB for 640x640 RGB)
        """
        self.video_path = video_path
        self.cache = LRUCache(capacity=cache_size)
        self.cache_size = cache_size
        
        # Video metadata
        cap = cv2.VideoCapture(self.video_path)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
    def get_frame(self, frame_idx: int) -> np.ndarray:
        """
        Extract specific frame from video.
        
        Args:
            frame_idx: Frame index to extract
            
        Returns:
            Frame as RGB numpy array (H, W, 3)
        """
        # Check cache
        cached_frame = self.cache.get(frame_idx)
        if cached_frame is not None:
            return cached_frame.copy()
        
        # Read from video
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError(f"Failed to read frame {frame_idx} from {self.video_path}")
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Store in cache
        self.cache.put(frame_idx, frame.copy())
        
        return frame.copy()
    
    def get_frames_batch(self, frame_indices: list) -> list:
        """
        Efficiently extract multiple frames.
        
        Args:
            frame_indices: List of frame indices to extract
            
        Returns:
            List of frames as RGB numpy arrays
        """
        frames = []
        uncached_indices = []
        
        # Check cache first
        for idx in frame_indices:
            cached_frame = self.cache.get(idx)
            if cached_frame is not None:
                frames.append((idx, cached_frame.copy()))
            else:
                uncached_indices.append(idx)
        
        # Read uncached frames in sorted order (more efficient)
        if uncached_indices:
            cap = cv2.VideoCapture(self.video_path)
            for idx in sorted(uncached_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.cache.put(idx, frame.copy())
                    frames.append((idx, frame.copy()))
            cap.release()
        
        # Sort frames by original order
        frames.sort(key=lambda x: frame_indices.index(x[0]))
        return [f[1] for f in frames]
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        return self.cache.get_stats()


class SupportImageCache:
    """
    Global LRU cache for support images with memory management.
    
    Since support images are reused frequently across episodes,
    caching them significantly reduces disk I/O.
    """
    
    def __init__(self, max_size_mb: int = 200):
        """
        Args:
            max_size_mb: Maximum cache size in megabytes
        """
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.cache = OrderedDict()
        self.current_size = 0
        self.hits = 0
        self.misses = 0
    
    def _get_image_size(self, img: np.ndarray) -> int:
        """Get image size in bytes."""
        return img.nbytes
    
    def get(self, img_path: str) -> Optional[np.ndarray]:
        """Get image from cache."""
        if img_path in self.cache:
            self.hits += 1
            # Move to end (most recently used)
            self.cache.move_to_end(img_path)
            return self.cache[img_path].copy()
        self.misses += 1
        return None
    
    def put(self, img_path: str, img: np.ndarray):
        """Put image in cache with LRU eviction."""
        img_size = self._get_image_size(img)
        
        # Remove existing entry if present
        if img_path in self.cache:
            old_img = self.cache.pop(img_path)
            self.current_size -= self._get_image_size(old_img)
        
        # Evict LRU items until we have space
        while self.current_size + img_size > self.max_size_bytes and len(self.cache) > 0:
            # Remove oldest (first) item
            _, evicted_img = self.cache.popitem(last=False)
            self.current_size -= self._get_image_size(evicted_img)
        
        # Add new image
        self.cache[img_path] = img.copy()
        self.current_size += img_size
    
    def load_image(self, img_path: str) -> np.ndarray:
        """
        Load image from cache or disk.
        
        Args:
            img_path: Path to image file
            
        Returns:
            Image as RGB numpy array
        """
        # Check cache
        cached_img = self.get(img_path)
        if cached_img is not None:
            return cached_img
        
        # Load from disk
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Cache it
        self.put(img_path, img)
        
        return img.copy()
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        return {
            'size_mb': self.current_size / (1024 * 1024),
            'max_size_mb': self.max_size_bytes / (1024 * 1024),
            'num_images': len(self.cache),
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
        }


# Global support image cache (shared across all dataset instances)
_SUPPORT_IMAGE_CACHE = None


def get_support_image_cache(max_size_mb: int = 200) -> SupportImageCache:
    """Get or create global support image cache."""
    global _SUPPORT_IMAGE_CACHE
    if _SUPPORT_IMAGE_CACHE is None:
        _SUPPORT_IMAGE_CACHE = SupportImageCache(max_size_mb=max_size_mb)
    return _SUPPORT_IMAGE_CACHE


class RefDetDataset(Dataset):
    """
    Reference-Based Detection Dataset for Few-Shot Learning.
    
    Each sample represents one object class with:
    - Support images: K reference images (from object_images/)
    - Query frames: Frames with bounding box annotations
    
    Args:
        data_root: Root directory containing samples/
        annotations_file: Path to annotations.json
        mode: 'train' or 'val'
        cache_frames: Whether to cache extracted frames
    """
    
    def __init__(
        self,
        data_root: str,
        annotations_file: str,
        mode: str = 'train',
        cache_frames: bool = True,
        num_aug: int = 1,
        frame_cache_size: int = 500,
        support_cache_size_mb: int = 200,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.mode = mode
        self.cache_frames = cache_frames
        self.num_aug = max(1, num_aug)  # Ensure at least 1
        self.frame_cache_size = frame_cache_size
        
        # Initialize caching systems
        self.support_cache = get_support_image_cache(max_size_mb=support_cache_size_mb)
        
        # Load annotations
        print(f"Loading annotations from {annotations_file}...")
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        
        # Parse dataset structure
        self.classes = []  # List of class names (video_ids)
        self.class_to_idx = {}  # video_id -> class index
        self.class_data = {}  # video_id -> data dict
        
        self._parse_annotations()
        
        # Video frame extractors (lazy initialization)
        self.video_extractors = {}
        
        base_frames = sum(len(d['frames']) for d in self.class_data.values())
        print(f"\nDataset initialized:")
        print(f"  Mode: {mode}")
        print(f"  Classes: {len(self.classes)}")
        print(f"  Base frames: {base_frames}")
        print(f"  Augmentation multiplier: {self.num_aug}x")
        print(f"  Total samples (with augmentation): {len(self)}")
        print(f"  Frame cache size: {frame_cache_size} frames (~{frame_cache_size * 0.6:.0f}MB)")
        print(f"  Support image cache: {support_cache_size_mb}MB")
    
    def _parse_annotations(self):
        """Parse annotations.json and organize by class."""
        for entry in self.annotations:
            video_id = entry['video_id']
            
            # Check if sample directory exists
            sample_dir = self.data_root / video_id
            if not sample_dir.exists():
                print(f"Warning: Sample directory not found: {sample_dir}")
                continue
            
            # Get support images
            support_dir = sample_dir / 'object_images'
            support_images = sorted(list(support_dir.glob('*.jpg')))
            
            if len(support_images) == 0:
                print(f"Warning: No support images found for {video_id}")
                continue
            
            # Get video path
            video_path = sample_dir / 'drone_video.mp4'
            if not video_path.exists():
                print(f"Warning: Video not found: {video_path}")
                continue
            
            # Parse frame annotations
            frames = {}
            for anno in entry['annotations']:
                for bbox in anno['bboxes']:
                    frame_idx = bbox['frame']
                    bbox_data = {
                        'x1': bbox['x1'],
                        'y1': bbox['y1'],
                        'x2': bbox['x2'],
                        'y2': bbox['y2']
                    }
                    
                    if frame_idx not in frames:
                        frames[frame_idx] = []
                    frames[frame_idx].append(bbox_data)
            
            # Get total frames in video and identify background frames
            import cv2
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            annotated_frame_set = set(frames.keys())
            background_frames = [i for i in range(total_frames) if i not in annotated_frame_set]
            
            # Store class data
            class_idx = len(self.classes)
            self.classes.append(video_id)
            self.class_to_idx[video_id] = class_idx
            self.class_data[video_id] = {
                'video_path': str(video_path),
                'support_images': [str(p) for p in support_images],
                'frames': frames,  # frame_idx -> list of bboxes
                'frame_indices': sorted(frames.keys()),
                'background_frames': background_frames,  # Frame indices without objects
                'total_frames': total_frames,
            }
    
    def _get_video_extractor(self, video_path: str) -> VideoFrameExtractor:
        """Get or create video extractor for a video."""
        if video_path not in self.video_extractors:
            cache_size = self.frame_cache_size if self.cache_frames else 0
            self.video_extractors[video_path] = VideoFrameExtractor(video_path, cache_size)
        return self.video_extractors[video_path]
    
    def _load_support_images(self, img_paths: list) -> list:
        """
        Load support images using cache.
        
        Args:
            img_paths: List of image file paths
            
        Returns:
            List of RGB numpy arrays
        """
        return [self.support_cache.load_image(str(path)) for path in img_paths]
    
    def __len__(self) -> int:
        """Total number of samples (frames * num_aug across all classes)."""
        base_count = sum(len(data['frame_indices']) for data in self.class_data.values())
        return base_count * self.num_aug
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample (one frame + class info).
        With num_aug > 1, the same frame can be sampled multiple times with different augmentations.
        
        Returns:
            dict with keys:
                - query_frame: (H, W, 3) numpy array
                - bboxes: (N, 4) numpy array (x1, y1, x2, y2)
                - class_id: int (class index)
                - video_id: str (class name)
                - frame_idx: int
                - support_images: List of (H, W, 3) numpy arrays
                - aug_idx: int (augmentation index 0 to num_aug-1)
        """
        # Calculate base count (total frames without augmentation)
        base_count = sum(len(data['frame_indices']) for data in self.class_data.values())
        
        # Map augmented index to (base_idx, aug_idx)
        base_idx = idx % base_count
        aug_idx = idx // base_count
        
        # Convert base index to (class_idx, frame_idx)
        current_count = 0
        for class_name, data in self.class_data.items():
            num_frames = len(data['frame_indices'])
            if base_idx < current_count + num_frames:
                frame_offset = base_idx - current_count
                frame_idx = data['frame_indices'][frame_offset]
                break
            current_count += num_frames
        else:
            raise IndexError(f"Index {idx} out of range")
        
        # Load query frame
        video_path = data['video_path']
        extractor = self._get_video_extractor(video_path)
        query_frame = extractor.get_frame(frame_idx)
        
        # Get bboxes for this frame
        bboxes = np.array([[b['x1'], b['y1'], b['x2'], b['y2']] 
                          for b in data['frames'][frame_idx]], dtype=np.float32)
        
        # Load support images from cache
        support_images = self._load_support_images(data['support_images'])
        
        return {
            'query_frame': query_frame,
            'bboxes': bboxes,
            'class_id': self.class_to_idx[class_name],
            'video_id': class_name,
            'frame_idx': frame_idx,
            'support_images': support_images,
            'aug_idx': aug_idx,
        }
    
    def get_background_frame(self, class_name: str) -> np.ndarray:
        """
        Get a random background frame (no objects) from a video.
        
        Args:
            class_name: Video ID / class name
            
        Returns:
            Background frame as (H, W, 3) numpy array
        """
        if class_name not in self.class_data:
            raise ValueError(f"Class {class_name} not found")
        
        data = self.class_data[class_name]
        
        if len(data['background_frames']) == 0:
            raise ValueError(f"No background frames available for {class_name}")
        
        # Sample a random background frame
        frame_idx = random.choice(data['background_frames'])
        extractor = self._get_video_extractor(data['video_path'])
        return extractor.get_frame(frame_idx)
    
    def get_triplet_sample(
        self, 
        class_name: str, 
        negative_strategy: str = 'background'
    ) -> Dict:
        """
        Get a triplet sample for contrastive learning.
        
        Triplet components:
        - Anchor: Support image (reference image)
        - Positive: Query frame with object from same class
        - Negative: Background frame OR frame from different class
        
        Args:
            class_name: Video ID / class name for anchor and positive
            negative_strategy: 'background', 'cross_class', or 'mixed'
                - 'background': Use background frame from same video
                - 'cross_class': Use frame from different class
                - 'mixed': Randomly choose between background and cross_class
                
        Returns:
            dict with:
                - anchor_image: Support image (H, W, 3)
                - positive_frame: Query frame with object (H, W, 3)
                - positive_bboxes: Bounding boxes in positive frame (N, 4)
                - negative_frame: Background or cross-class frame (H, W, 3)
                - negative_bboxes: Empty array for background, or bboxes for cross_class (M, 4)
                - class_name: Anchor class name
                - negative_type: 'background' or 'cross_class'
        """
        if class_name not in self.class_data:
            raise ValueError(f"Class {class_name} not found")
        
        data = self.class_data[class_name]
        
        # 1. Sample anchor (support image) from cache
        anchor_img_path = random.choice(data['support_images'])
        anchor_image = self.support_cache.load_image(str(anchor_img_path))
        
        # 2. Sample positive (frame with object from same class)
        pos_frame_idx = random.choice(data['frame_indices'])
        extractor = self._get_video_extractor(data['video_path'])
        positive_frame = extractor.get_frame(pos_frame_idx)
        positive_bboxes = np.array([[b['x1'], b['y1'], b['x2'], b['y2']] 
                                   for b in data['frames'][pos_frame_idx]], dtype=np.float32)
        
        # 3. Sample negative based on strategy
        negative_frame = None
        negative_bboxes = None
        negative_type = None
        
        if negative_strategy == 'mixed':
            negative_strategy = random.choice(['background', 'cross_class'])
        
        if negative_strategy == 'background':
            # Use background frame from same video
            if len(data['background_frames']) == 0:
                # Fallback to cross_class if no background frames
                negative_strategy = 'cross_class'
            else:
                negative_frame = self.get_background_frame(class_name)
                negative_bboxes = np.array([], dtype=np.float32).reshape(0, 4)
                negative_type = 'background'
        
        if negative_strategy == 'cross_class':
            # Use frame from different class
            other_classes = [c for c in self.classes if c != class_name]
            if len(other_classes) == 0:
                raise ValueError("Need at least 2 classes for cross_class strategy")
            
            neg_class = random.choice(other_classes)
            neg_data = self.class_data[neg_class]
            neg_frame_idx = random.choice(neg_data['frame_indices'])
            neg_extractor = self._get_video_extractor(neg_data['video_path'])
            negative_frame = neg_extractor.get_frame(neg_frame_idx)
            negative_bboxes = np.array([[b['x1'], b['y1'], b['x2'], b['y2']] 
                                       for b in neg_data['frames'][neg_frame_idx]], dtype=np.float32)
            negative_type = 'cross_class'
        
        # Ensure all variables are assigned (should never happen, but for type safety)
        assert negative_frame is not None, "negative_frame must be assigned"
        assert negative_bboxes is not None, "negative_bboxes must be assigned"
        assert negative_type is not None, "negative_type must be assigned"
        
        return {
            'anchor_image': anchor_image,
            'positive_frame': positive_frame,
            'positive_bboxes': positive_bboxes,
            'negative_frame': negative_frame,
            'negative_bboxes': negative_bboxes,
            'class_name': class_name,
            'class_id': self.class_to_idx[class_name],
            'negative_type': negative_type,
        }
    
    def get_class_samples(self, class_name: str, num_query: int = 4) -> Dict:
        """
        Get K support images + Q query frames for a specific class.
        Used for episodic training.
        
        Args:
            class_name: Video ID / class name
            num_query: Number of query frames to sample
            
        Returns:
            dict with support_images, query_frames, query_bboxes
        """
        if class_name not in self.class_data:
            raise ValueError(f"Class {class_name} not found")
        
        data = self.class_data[class_name]
        
        # Load all support images from cache
        support_images = self._load_support_images(data['support_images'])
        
        # Sample query frames
        available_frames = data['frame_indices']
        num_query = min(num_query, len(available_frames))
        sampled_frame_indices = random.sample(available_frames, num_query)
        
        query_frames = []
        query_bboxes = []
        extractor = self._get_video_extractor(data['video_path'])
        
        for frame_idx in sampled_frame_indices:
            frame = extractor.get_frame(frame_idx)
            bboxes = np.array([[b['x1'], b['y1'], b['x2'], b['y2']] 
                              for b in data['frames'][frame_idx]], dtype=np.float32)
            query_frames.append(frame)
            query_bboxes.append(bboxes)
        
        return {
            'support_images': support_images,
            'query_frames': query_frames,
            'query_bboxes': query_bboxes,
            'class_name': class_name,
        }
    
    def get_cache_stats(self) -> Dict:
        """
        Get caching statistics for performance monitoring.
        
        Returns:
            dict with cache stats for support images and video frames
        """
        stats = {
            'support_images': self.support_cache.get_stats(),
            'video_frames': {},
        }
        
        # Aggregate video frame cache stats
        total_hits = 0
        total_misses = 0
        total_cached_frames = 0
        
        for video_path, extractor in self.video_extractors.items():
            frame_stats = extractor.get_cache_stats()
            total_hits += frame_stats['hits']
            total_misses += frame_stats['misses']
            total_cached_frames += frame_stats['size']
        
        if len(self.video_extractors) > 0:
            total_requests = total_hits + total_misses
            stats['video_frames'] = {
                'total_cached_frames': total_cached_frames,
                'total_videos': len(self.video_extractors),
                'total_hits': total_hits,
                'total_misses': total_misses,
                'hit_rate': total_hits / total_requests if total_requests > 0 else 0.0,
            }
        
        return stats
    
    def print_cache_stats(self):
        """Print cache statistics in human-readable format."""
        stats = self.get_cache_stats()
        
        print("\n" + "="*60)
        print("CACHE STATISTICS")
        print("="*60)
        
        # Support images
        si_stats = stats['support_images']
        print(f"\nSupport Images:")
        print(f"  Cached images: {si_stats['num_images']}")
        print(f"  Memory usage: {si_stats['size_mb']:.2f} MB / {si_stats['max_size_mb']:.0f} MB")
        print(f"  Hits: {si_stats['hits']}, Misses: {si_stats['misses']}")
        print(f"  Hit rate: {si_stats['hit_rate']*100:.1f}%")
        
        # Video frames
        if stats['video_frames']:
            vf_stats = stats['video_frames']
            print(f"\nVideo Frames:")
            print(f"  Active videos: {vf_stats['total_videos']}")
            print(f"  Cached frames: {vf_stats['total_cached_frames']}")
            print(f"  Hits: {vf_stats['total_hits']}, Misses: {vf_stats['total_misses']}")
            print(f"  Hit rate: {vf_stats['hit_rate']*100:.1f}%")
        
        print("="*60 + "\n")


class EpisodicBatchSampler(Sampler):
    """
    Episodic sampler for N-way K-shot Q-query few-shot learning.
    
    Each episode samples:
    - N classes (ways)
    - K support images per class (shots) - fixed by dataset
    - Q query frames per class (queries)
    
    Args:
        dataset: RefDetDataset
        n_way: Number of classes per episode
        n_query: Number of query samples per class
        n_episodes: Number of episodes per epoch
    """
    
    def __init__(
        self,
        dataset: RefDetDataset,
        n_way: int = 2,
        n_query: int = 4,
        n_episodes: int = 100,
    ):
        self.dataset = dataset
        self.n_way = n_way
        self.n_query = n_query
        self.n_episodes = n_episodes
        
        self.classes = dataset.classes
        
        if n_way > len(self.classes):
            raise ValueError(f"n_way ({n_way}) cannot exceed number of classes ({len(self.classes)})")
    
    def __iter__(self):
        """Generate episode indices."""
        for _ in range(self.n_episodes):
            # Sample N classes
            episode_classes = random.sample(self.classes, self.n_way)
            
            # For each class, sample Q query frames
            episode_indices = []
            for class_name in episode_classes:
                data = self.dataset.class_data[class_name]
                frame_indices = data['frame_indices']
                
                # Sample query frames
                num_available = len(frame_indices)
                num_sample = min(self.n_query, num_available)
                sampled_frames = random.sample(frame_indices, num_sample)
                
                # Convert to dataset indices
                # Find the starting index for this class
                start_idx = 0
                for prev_class in self.dataset.classes:
                    if prev_class == class_name:
                        break
                    start_idx += len(self.dataset.class_data[prev_class]['frame_indices'])
                
                # Map frame indices to dataset indices
                frame_to_idx_map = {frame: start_idx + i 
                                   for i, frame in enumerate(data['frame_indices'])}
                
                for frame_idx in sampled_frames:
                    episode_indices.append(frame_to_idx_map[frame_idx])
            
            # Yield all indices for this episode as a batch
            yield episode_indices
    
    def __len__(self) -> int:
        return self.n_episodes
