"""
Reference-Based Detection Dataset for Few-Shot UAV Object Detection
Handles video frame extraction, support image loading, and episodic sampling.
"""

import json
import cv2
import numpy as np
from torch.utils.data import Dataset, Sampler
from pathlib import Path
from typing import Dict
import random


class VideoFrameExtractor:
    """Efficient video frame extraction with caching."""
    
    def __init__(self, video_path: str, cache_size: int = 100):
        """
        Args:
            video_path: Path to video file
            cache_size: Number of frames to cache
        """
        self.video_path = video_path
        self.cache = {}
        self.cache_size = cache_size
        self.cache_keys = []
        
    def get_frame(self, frame_idx: int) -> np.ndarray:
        """Extract specific frame from video."""
        # Check cache
        if frame_idx in self.cache:
            return self.cache[frame_idx].copy()
        
        # Read from video
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError(f"Failed to read frame {frame_idx} from {self.video_path}")
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Update cache
        self.cache[frame_idx] = frame
        self.cache_keys.append(frame_idx)
        
        # Limit cache size
        if len(self.cache_keys) > self.cache_size:
            old_key = self.cache_keys.pop(0)
            del self.cache[old_key]
        
        return frame.copy()


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
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.mode = mode
        self.cache_frames = cache_frames
        
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
        
        print(f"\nDataset initialized:")
        print(f"  Mode: {mode}")
        print(f"  Classes: {len(self.classes)}")
        print(f"  Total annotated frames: {sum(len(d['frames']) for d in self.class_data.values())}")
    
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
            cache_size = 100 if self.cache_frames else 0
            self.video_extractors[video_path] = VideoFrameExtractor(video_path, cache_size)
        return self.video_extractors[video_path]
    
    def __len__(self) -> int:
        """Total number of annotated frames across all classes."""
        return sum(len(data['frame_indices']) for data in self.class_data.values())
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a single sample (one frame + class info).
        
        Returns:
            dict with keys:
                - query_frame: (H, W, 3) numpy array
                - bboxes: (N, 4) numpy array (x1, y1, x2, y2)
                - class_id: int (class index)
                - video_id: str (class name)
                - frame_idx: int
                - support_images: List of (H, W, 3) numpy arrays
        """
        # Convert flat index to (class_idx, frame_idx)
        current_count = 0
        for class_name, data in self.class_data.items():
            num_frames = len(data['frame_indices'])
            if idx < current_count + num_frames:
                frame_offset = idx - current_count
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
        
        # Load support images
        support_images = []
        for img_path in data['support_images']:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            support_images.append(img)
        
        return {
            'query_frame': query_frame,
            'bboxes': bboxes,
            'class_id': self.class_to_idx[class_name],
            'video_id': class_name,
            'frame_idx': frame_idx,
            'support_images': support_images,
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
        
        # 1. Sample anchor (support image)
        anchor_img_path = random.choice(data['support_images'])
        anchor_image = cv2.imread(anchor_img_path)
        anchor_image = cv2.cvtColor(anchor_image, cv2.COLOR_BGR2RGB)
        
        # 2. Sample positive (frame with object from same class)
        pos_frame_idx = random.choice(data['frame_indices'])
        extractor = self._get_video_extractor(data['video_path'])
        positive_frame = extractor.get_frame(pos_frame_idx)
        positive_bboxes = np.array([[b['x1'], b['y1'], b['x2'], b['y2']] 
                                   for b in data['frames'][pos_frame_idx]], dtype=np.float32)
        
        # 3. Sample negative based on strategy
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
        
        # Load all support images
        support_images = []
        for img_path in data['support_images']:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            support_images.append(img)
        
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
