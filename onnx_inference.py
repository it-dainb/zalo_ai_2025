"""
ONNX Inference Script for YOLOv8n-RefDet
========================================

Performs inference using exported ONNX model with ONNXRuntime.
Supports multiple execution providers (CUDA, TensorRT, CPU).

Usage:
    # Basic inference
    python onnx_inference.py --model model.onnx --query image.jpg --support ref.jpg
    
    # Batch inference on directory
    python onnx_inference.py --model model.onnx --query images/ --support ref.jpg --output results/
    
    # Use TensorRT provider
    python onnx_inference.py --model model.onnx --query image.jpg --support ref.jpg --provider tensorrt
    
    # Benchmark mode
    python onnx_inference.py --model model.onnx --benchmark --warmup 10 --iterations 100
"""

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort


class ONNXRefDetInference:
    """
    ONNX inference wrapper for YOLOv8n-RefDet.
    
    Handles:
    - Image preprocessing
    - ONNX Runtime session management
    - Multi-provider support (CUDA, TensorRT, CPU)
    - Post-processing (NMS, score filtering)
    - Visualization
    """
    
    def __init__(
        self,
        model_path: str,
        mode: str = 'dual',
        provider: str = 'cuda',
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        img_size: int = 640,
        support_size: int = 256,
    ):
        """
        Args:
            model_path: Path to ONNX model
            mode: Model mode ('standard', 'dual')
            provider: Execution provider ('cuda', 'tensorrt', 'cpu')
            conf_thres: Confidence threshold
            iou_thres: IoU threshold for NMS
            img_size: Query image input size
            support_size: Support image input size
        """
        self.model_path = model_path
        self.mode = mode
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.img_size = img_size
        self.support_size = support_size
        
        # Setup execution provider
        self.providers = self._get_providers(provider)
        
        # Create ONNX Runtime session
        self.session = ort.InferenceSession(model_path, providers=self.providers)
        
        # Get input/output names
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        
        # Cache for support features (if model uses caching)
        self._support_cached = False
        self._cached_support = None
    
    def _get_providers(self, provider: str) -> List[str]:
        """Get ONNX Runtime execution providers."""
        available = ort.get_available_providers()
        
        if provider == 'cuda':
            if 'CUDAExecutionProvider' in available:
                return ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                print("⚠ CUDA not available, falling back to CPU")
                return ['CPUExecutionProvider']
        
        elif provider == 'tensorrt':
            if 'TensorrtExecutionProvider' in available:
                return ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                print("⚠ TensorRT not available, falling back to CUDA/CPU")
                if 'CUDAExecutionProvider' in available:
                    return ['CUDAExecutionProvider', 'CPUExecutionProvider']
                return ['CPUExecutionProvider']
        
        elif provider == 'cpu':
            return ['CPUExecutionProvider']
        
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def preprocess_image(
        self,
        image: np.ndarray,
        target_size: int,
    ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Preprocess image for inference.
        
        Args:
            image: Input image (BGR, HWC)
            target_size: Target size (square)
        
        Returns:
            (preprocessed, scale, (pad_w, pad_h))
        """
        # Get original dimensions
        h, w = image.shape[:2]
        
        # Compute scaling to fit in target_size (letterbox)
        scale = min(target_size / h, target_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Pad to square
        pad_h = (target_size - new_h) // 2
        pad_w = (target_size - new_w) // 2
        
        padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
        padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
        
        # Convert to RGB and normalize
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1] and transpose to CHW
        normalized = rgb.astype(np.float32) / 255.0
        transposed = np.transpose(normalized, (2, 0, 1))
        
        # Add batch dimension
        batched = np.expand_dims(transposed, axis=0)
        
        return batched, scale, (pad_w, pad_h)
    
    def set_reference_images(
        self,
        support_images: List[np.ndarray],
    ):
        """
        Cache support images for inference.
        
        Args:
            support_images: List of support images (BGR, HWC)
        """
        # Preprocess all support images
        support_batch = []
        for img in support_images:
            preprocessed, _, _ = self.preprocess_image(img, self.support_size)
            support_batch.append(preprocessed)
        
        # Stack into batch
        self._cached_support = np.concatenate(support_batch, axis=0)
        self._support_cached = True
        
        print(f"✓ Cached {len(support_images)} support image(s)")
    
    def inference(
        self,
        query_image: np.ndarray,
        support_images: Optional[List[np.ndarray]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Run inference on query image.
        
        Args:
            query_image: Query image (BGR, HWC)
            support_images: Support images (only for dual mode without caching)
        
        Returns:
            Dictionary of outputs (boxes, cls/sim per scale)
        """
        # Preprocess query
        query_input, scale, (pad_w, pad_h) = self.preprocess_image(
            query_image, self.img_size
        )
        
        # Prepare inputs
        ort_inputs = {'query_image': query_input}
        
        # Handle support images
        if self.mode == 'dual':
            if 'support_images' in self.input_names:
                # Model expects support images as input
                if support_images is None:
                    raise ValueError("Support images required for dual mode")
                
                support_batch = []
                for img in support_images:
                    preprocessed, _, _ = self.preprocess_image(img, self.support_size)
                    support_batch.append(preprocessed)
                
                ort_inputs['support_images'] = np.concatenate(support_batch, axis=0)
            
            elif not self._support_cached:
                # Model uses caching but no support cached
                if support_images is None:
                    raise ValueError(
                        "Support images required. Call set_reference_images() first "
                        "or provide support_images."
                    )
                self.set_reference_images(support_images)
        
        # Run inference
        ort_outputs = self.session.run(None, ort_inputs)
        
        # Parse outputs
        outputs = {}
        scales = ['p2', 'p3', 'p4', 'p5']
        
        if self.mode == 'standard':
            # 8 outputs: 4 boxes + 4 cls
            outputs['standard_boxes'] = ort_outputs[:4]
            outputs['standard_cls'] = ort_outputs[4:8]
        
        elif self.mode == 'dual':
            # 16 outputs: 4 boxes + 4 cls + 4 proto_boxes + 4 proto_sim
            outputs['standard_boxes'] = ort_outputs[:4]
            outputs['standard_cls'] = ort_outputs[4:8]
            outputs['prototype_boxes'] = ort_outputs[8:12]
            outputs['prototype_sim'] = ort_outputs[12:16]
        
        # Store preprocessing info for post-processing
        outputs['_meta'] = {
            'scale': scale,
            'pad_w': pad_w,
            'pad_h': pad_h,
            'orig_shape': query_image.shape[:2],
        }
        
        return outputs
    
    def postprocess(
        self,
        outputs: Dict[str, np.ndarray],
        head: str = 'standard',
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Post-process outputs to get final detections.
        
        Args:
            outputs: Raw model outputs
            head: Which head to use ('standard' or 'prototype')
        
        Returns:
            (boxes, scores, class_ids) in original image coordinates
            boxes: (N, 4) [x1, y1, x2, y2]
            scores: (N,)
            class_ids: (N,)
        """
        # Get metadata
        meta = outputs['_meta']
        scale = meta['scale']
        pad_w = meta['pad_w']
        pad_h = meta['pad_h']
        orig_h, orig_w = meta['orig_shape']
        
        # Select head
        if head == 'standard':
            boxes_list = outputs['standard_boxes']
            scores_list = outputs['standard_cls']
        else:  # prototype
            boxes_list = outputs['prototype_boxes']
            scores_list = outputs['prototype_sim']
        
        # Collect all predictions across scales
        all_boxes = []
        all_scores = []
        all_class_ids = []
        
        for boxes, scores in zip(boxes_list, scores_list):
            # boxes: (B, H*W, 4)  [cx, cy, w, h] normalized
            # scores: (B, H*W, num_classes)
            
            boxes = boxes[0]  # Remove batch dim
            scores = scores[0]
            
            # Get class predictions
            class_ids = np.argmax(scores, axis=-1)
            class_scores = np.max(scores, axis=-1)
            
            # Filter by confidence
            mask = class_scores > self.conf_thres
            boxes = boxes[mask]
            class_scores = class_scores[mask]
            class_ids = class_ids[mask]
            
            if len(boxes) == 0:
                continue
            
            # Convert from [cx, cy, w, h] to [x1, y1, x2, y2]
            x1 = boxes[:, 0] - boxes[:, 2] / 2
            y1 = boxes[:, 1] - boxes[:, 3] / 2
            x2 = boxes[:, 0] + boxes[:, 2] / 2
            y2 = boxes[:, 1] + boxes[:, 3] / 2
            
            boxes_xyxy = np.stack([x1, y1, x2, y2], axis=-1)
            
            # Scale back to original coordinates
            boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - pad_w) / scale
            boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - pad_h) / scale
            
            # Clip to image bounds
            boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, orig_w)
            boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, orig_h)
            
            all_boxes.append(boxes_xyxy)
            all_scores.append(class_scores)
            all_class_ids.append(class_ids)
        
        if len(all_boxes) == 0:
            return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=np.int32)
        
        # Concatenate all scales
        all_boxes = np.concatenate(all_boxes, axis=0)
        all_scores = np.concatenate(all_scores, axis=0)
        all_class_ids = np.concatenate(all_class_ids, axis=0)
        
        # Apply NMS
        keep_indices = self._nms(all_boxes, all_scores, self.iou_thres)
        
        return all_boxes[keep_indices], all_scores[keep_indices], all_class_ids[keep_indices]
    
    def _nms(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        iou_threshold: float,
    ) -> np.ndarray:
        """Non-Maximum Suppression."""
        if len(boxes) == 0:
            return np.array([], dtype=np.int32)
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return np.array(keep, dtype=np.int32)
    
    def visualize(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
        class_names: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Visualize detections on image.
        
        Args:
            image: Original image (BGR)
            boxes: (N, 4) [x1, y1, x2, y2]
            scores: (N,)
            class_ids: (N,)
            class_names: Optional class names
        
        Returns:
            Annotated image
        """
        vis = image.copy()
        
        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = box.astype(int)
            
            # Draw box
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            if class_names:
                label = f"{class_names[class_id]}: {score:.2f}"
            else:
                label = f"Class {class_id}: {score:.2f}"
            
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            cv2.rectangle(
                vis,
                (x1, y1 - label_h - 10),
                (x1 + label_w, y1),
                (0, 255, 0),
                -1,
            )
            
            cv2.putText(
                vis,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
            )
        
        return vis
    
    def benchmark(
        self,
        warmup: int = 10,
        iterations: int = 100,
    ) -> Dict[str, float]:
        """
        Benchmark inference speed.
        
        Args:
            warmup: Number of warmup iterations
            iterations: Number of timed iterations
        
        Returns:
            Dictionary with timing statistics
        """
        # Create dummy inputs
        query_dummy = np.random.randn(1, 3, self.img_size, self.img_size).astype(np.float32)
        
        ort_inputs = {'query_image': query_dummy}
        
        if self.mode == 'dual' and 'support_images' in self.input_names:
            support_dummy = np.random.randn(3, 3, self.support_size, self.support_size).astype(np.float32)
            ort_inputs['support_images'] = support_dummy
        elif self.mode == 'dual' and not self._support_cached:
            # Need to cache dummy support
            support_dummy = np.random.randn(3, 3, self.support_size, self.support_size).astype(np.float32)
            self._cached_support = support_dummy
            self._support_cached = True
        
        # Warmup
        print(f"Warming up ({warmup} iterations)...")
        for _ in range(warmup):
            self.session.run(None, ort_inputs)
        
        # Benchmark
        print(f"Benchmarking ({iterations} iterations)...")
        times = []
        
        for _ in range(iterations):
            start = time.perf_counter()
            self.session.run(None, ort_inputs)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
        
        times = np.array(times)
        
        stats = {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
            'median': np.median(times),
            'fps': 1000 / np.mean(times),
        }
        
        return stats


def main():
    parser = argparse.ArgumentParser(description='ONNX Inference for YOLOv8n-RefDet')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to ONNX model')
    parser.add_argument('--query', type=str, default=None,
                        help='Query image or directory')
    parser.add_argument('--support', type=str, default=None,
                        help='Support image(s) (comma-separated for multiple)')
    parser.add_argument('--output', type=str, default='output',
                        help='Output directory for visualizations')
    parser.add_argument('--mode', type=str, default='dual', choices=['standard', 'dual'],
                        help='Inference mode')
    parser.add_argument('--provider', type=str, default='cuda',
                        choices=['cuda', 'tensorrt', 'cpu'],
                        help='Execution provider')
    parser.add_argument('--conf-thres', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45,
                        help='IoU threshold for NMS')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run benchmark mode')
    parser.add_argument('--warmup', type=int, default=10,
                        help='Warmup iterations for benchmark')
    parser.add_argument('--iterations', type=int, default=100,
                        help='Benchmark iterations')
    
    args = parser.parse_args()
    
    # Initialize inference engine
    print("\nInitializing ONNX inference engine...")
    engine = ONNXRefDetInference(
        model_path=args.model,
        mode=args.mode,
        provider=args.provider,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
    )
    
    print(f"✓ Model loaded: {args.model}")
    print(f"  Mode: {args.mode}")
    print(f"  Provider: {engine.providers[0]}")
    print(f"  Inputs: {engine.input_names}")
    print(f"  Outputs: {len(engine.output_names)} tensors")
    
    # Benchmark mode
    if args.benchmark:
        print("\n" + "="*70)
        print("Benchmark Mode")
        print("="*70)
        
        stats = engine.benchmark(warmup=args.warmup, iterations=args.iterations)
        
        print(f"\nResults:")
        print(f"  Mean: {stats['mean']:.2f} ms")
        print(f"  Std: {stats['std']:.2f} ms")
        print(f"  Min: {stats['min']:.2f} ms")
        print(f"  Max: {stats['max']:.2f} ms")
        print(f"  Median: {stats['median']:.2f} ms")
        print(f"  FPS: {stats['fps']:.1f}")
        print("="*70 + "\n")
        return
    
    # Inference mode
    if args.query is None:
        print("Error: --query required for inference mode")
        return
    
    # Load support images
    support_images = None
    if args.mode == 'dual':
        if args.support is None:
            print("Error: --support required for dual mode")
            return
        
        support_paths = args.support.split(',')
        support_images = []
        for path in support_paths:
            img = cv2.imread(path.strip())
            if img is None:
                print(f"Error: Failed to load support image: {path}")
                return
            support_images.append(img)
        
        print(f"\n✓ Loaded {len(support_images)} support image(s)")
        engine.set_reference_images(support_images)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process query image(s)
    query_path = Path(args.query)
    
    if query_path.is_file():
        query_paths = [query_path]
    elif query_path.is_dir():
        query_paths = list(query_path.glob('*.jpg')) + list(query_path.glob('*.png'))
    else:
        print(f"Error: Query path not found: {args.query}")
        return
    
    print(f"\n Processing {len(query_paths)} image(s)...")
    
    for img_path in query_paths:
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"⚠ Failed to load: {img_path}")
            continue
        
        # Run inference
        start = time.perf_counter()
        outputs = engine.inference(image, support_images=None)  # Support already cached
        
        # Post-process
        boxes, scores, class_ids = engine.postprocess(outputs, head='standard')
        end = time.perf_counter()
        
        # Visualize
        vis = engine.visualize(image, boxes, scores, class_ids)
        
        # Save
        output_path = output_dir / img_path.name
        cv2.imwrite(str(output_path), vis)
        
        print(f"✓ {img_path.name}: {len(boxes)} detections, {(end-start)*1000:.1f}ms")
    
    print(f"\n✓ Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
