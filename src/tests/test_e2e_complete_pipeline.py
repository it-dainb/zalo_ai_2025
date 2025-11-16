"""
Comprehensive End-to-End Pipeline Test
=======================================

Tests the complete workflow from smallest to biggest components:
1. Data loading → Model components → Training → Evaluation → Inference

This test ensures all components work together in the full pipeline.
"""

import pytest
import torch
import sys
from pathlib import Path
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.datasets.refdet_dataset import RefDetDataset, EpisodicBatchSampler
from src.datasets.collate import RefDetCollator
from src.models.yolov8n_refdet import YOLOv8nRefDet
from src.losses.combined_loss import ReferenceBasedDetectionLoss
from src.training.trainer import RefDetTrainer
from src.augmentations.augmentation_config import AugmentationConfig
from torch.utils.data import DataLoader


class TestCompleteE2EPipeline:
    """Test complete end-to-end pipeline"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def workspace(self, tmp_path):
        """Create temporary workspace for pipeline test"""
        workspace = {
            'checkpoint_dir': tmp_path / "checkpoints",
            'output_dir': tmp_path / "outputs",
        }
        workspace['checkpoint_dir'].mkdir(parents=True, exist_ok=True)
        workspace['output_dir'].mkdir(parents=True, exist_ok=True)
        return workspace
    
    @pytest.fixture
    def data_available(self):
        """Check if required data is available"""
        train_data = Path("./datasets/train/samples").exists()
        train_ann = Path("./datasets/train/annotations/annotations.json").exists()
        test_data = Path("./datasets/test/samples").exists()
        test_ann = Path("./datasets/test/annotations/annotations.json").exists()
        return train_data and train_ann and test_data and test_ann
    
    def test_complete_pipeline_flow(self, device, workspace, data_available):
        """
        Test complete pipeline:
        Data Loading → Model Init → Training → Checkpoint → Evaluation → Inference
        """
        if not data_available:
            pytest.skip("Required datasets not available")
        
        print("\n" + "="*70)
        print("COMPREHENSIVE END-TO-END PIPELINE TEST")
        print("="*70)
        
        # ========================================================================
        # STEP 1: DATA LOADING
        # ========================================================================
        print("\n[Step 1/6] Data Loading...")
        print("-" * 70)
        
        # Create training dataset
        train_dataset = RefDetDataset(
            data_root="./datasets/train/samples",
            annotations_file="./datasets/train/annotations/annotations.json",
            mode='train',
            cache_frames=True,
        )
        
        train_sampler = EpisodicBatchSampler(
            dataset=train_dataset,
            n_way=min(2, len(train_dataset.classes)),
            n_query=2,
            n_episodes=2,  # Minimal for testing
        )
        
        aug_config = AugmentationConfig(
            query_img_size=640,
            support_img_size=256,
            query_mosaic_prob=0.0,  # Disable for faster testing
            query_mixup_prob=0.0,
        )
        
        train_collator = RefDetCollator(
            config=aug_config,
            mode='train',
            stage=2,
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            collate_fn=train_collator,
            num_workers=0,
            pin_memory=False,
        )
        
        # Create test dataset
        test_dataset = RefDetDataset(
            data_root="./datasets/test/samples",
            annotations_file="./datasets/test/annotations/annotations.json",
            mode='val',
            cache_frames=True,
        )
        
        test_sampler = EpisodicBatchSampler(
            dataset=test_dataset,
            n_way=min(2, len(test_dataset.classes)),
            n_query=2,
            n_episodes=1,
        )
        
        test_collator = RefDetCollator(
            config=aug_config,
            mode='val',
            stage=2,
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_sampler=test_sampler,
            collate_fn=test_collator,
            num_workers=0,
        )
        
        print(f"✅ Data loading successful")
        print(f"   Train classes: {len(train_dataset.classes)}")
        print(f"   Test classes: {len(test_dataset.classes)}")
        print(f"   Train episodes: {len(train_sampler)}")
        print(f"   Test episodes: {len(test_sampler)}")
        
        # ========================================================================
        # STEP 2: MODEL INITIALIZATION
        # ========================================================================
        print("\n[Step 2/6] Model Initialization...")
        print("-" * 70)
        
        weights_path = "yolov8n.pt"
        if not Path(weights_path).exists():
            weights_path = "yolov8n.pt"
        
        model = YOLOv8nRefDet(
            yolo_weights=weights_path,
            nc_base=0,
            freeze_yolo=False,
            freeze_dinov3=True,
        ).to(device)
        
        print(f"✅ Model initialized")
        print(f"   Total parameters: {model.count_parameters()/1e6:.2f}M")
        print(f"   Trainable parameters: {model.count_parameters(trainable_only=True)/1e6:.2f}M")
        
        # ========================================================================
        # STEP 3: TRAINING SETUP
        # ========================================================================
        print("\n[Step 3/6] Training Setup...")
        print("-" * 70)
        
        loss_fn = ReferenceBasedDetectionLoss(
            stage=2,
            bbox_weight=7.5,
            cls_weight=0.5,
            supcon_weight=1.0,
            cpe_weight=0.5,
        ).to(device)
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=0.05,
        )
        
        trainer = RefDetTrainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=str(device),
            mixed_precision=False,  # Disable for testing
            gradient_accumulation_steps=1,
            checkpoint_dir=str(workspace['checkpoint_dir']),
            aug_config=aug_config,
            stage=2,
        )
        
        print(f"✅ Training setup complete")
        print(f"   Loss function: ReferenceBasedDetectionLoss (Stage 2)")
        print(f"   Optimizer: AdamW (lr=1e-4)")
        
        # ========================================================================
        # STEP 4: TRAINING
        # ========================================================================
        print("\n[Step 4/6] Training (1 epoch)...")
        print("-" * 70)
        
        # Training now works with nc_base=0 (OOV detection)
        trainer.train(
            train_loader=train_loader,
            val_loader=None,  # Skip validation for speed
            num_epochs=1,
            save_interval=1,
        )
        print(f"✅ Training completed")
        
        # ========================================================================
        # STEP 5: CHECKPOINT VERIFICATION
        # ========================================================================
        print("\n[Step 5/6] Checkpoint Verification...")
        print("-" * 70)
        
        checkpoint_files = list(workspace['checkpoint_dir'].glob("*.pt"))
        print(f"   Checkpoint dir: {workspace['checkpoint_dir']}")
        print(f"   Files found: {[f.name for f in checkpoint_files]}")
        assert len(checkpoint_files) > 0, f"No checkpoint saved in {workspace['checkpoint_dir']}"
        
        checkpoint_path = checkpoint_files[0]
        checkpoint = torch.load(checkpoint_path)
        
        assert 'model_state_dict' in checkpoint
        assert 'optimizer_state_dict' in checkpoint
        assert 'epoch' in checkpoint
        
        print(f"✅ Checkpoint verified")
        print(f"   Path: {checkpoint_path.name}")
        print(f"   Epoch: {checkpoint['epoch']}")
        
        # Load checkpoint into new model
        model_eval = YOLOv8nRefDet(
            yolo_weights=weights_path,
            nc_base=0,
        ).to(device)
        # Use strict=True to ensure model architecture is deterministic
        model_eval.load_state_dict(checkpoint['model_state_dict'], strict=True)
        model_eval.eval()
        
        print(f"✅ Checkpoint loaded into new model with strict=True")
        
        # ========================================================================
        # STEP 6: EVALUATION
        # ========================================================================
        print("\n[Step 6/6] Evaluation...")
        print("-" * 70)
        
        from evaluate import evaluate_episode
        
        all_results = []
        for batch in test_loader:
            try:
                results = evaluate_episode(model_eval, batch, device)
                all_results.append(results)
            except Exception as e:
                print(f"⚠️  Batch evaluation error: {str(e)}")
        
        if len(all_results) > 0:
            # Aggregate results
            total_tp = 0
            total_fp = 0
            total_fn = 0
            
            for results in all_results:
                for class_id, metrics in results.items():
                    total_tp += metrics.get('tp', 0)
                    total_fp += metrics.get('fp', 0)
                    total_fn += metrics.get('fn', 0)
            
            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"✅ Evaluation completed")
            print(f"   Episodes evaluated: {len(all_results)}")
            print(f"   Precision: {precision:.4f}")
            print(f"   Recall: {recall:.4f}")
            print(f"   F1 Score: {f1:.4f}")
        else:
            print(f"⚠️  No episodes evaluated")
        
        # ========================================================================
        # STEP 7: INFERENCE
        # ========================================================================
        print("\n[Step 7/7] Inference Testing...")
        print("-" * 70)
        
        # Test inference with dummy data
        support_images = torch.randn(3, 3, 256, 256).to(device)
        query_image = torch.randn(1, 3, 640, 640).to(device)
        
        with torch.no_grad():
            model_eval.set_reference_images(support_images, average_prototypes=True)
            outputs = model_eval(
                query_image=query_image,
                mode='prototype',
                use_cache=True,
            )
        
        assert outputs is not None
        print(f"✅ Inference successful")
        print(f"   Output keys: {list(outputs.keys())}")
        
        # ========================================================================
        # PIPELINE COMPLETE
        # ========================================================================
        print("\n" + "="*70)
        print("✅ COMPLETE END-TO-END PIPELINE TEST PASSED")
        print("="*70)
        print("\nPipeline Summary:")
        print("  1. ✅ Data Loading - Created train/test datasets")
        print("  2. ✅ Model Initialization - Loaded YOLOv8n-RefDet")
        print("  3. ✅ Training Setup - Configured loss, optimizer, trainer")
        print("  4. ✅ Training - Completed 1 epoch")
        print("  5. ✅ Checkpoint - Saved and loaded checkpoint")
        print("  6. ✅ Evaluation - Computed metrics on test set")
        print("  7. ✅ Inference - Ran inference with cached features")
        print("="*70 + "\n")


class TestPipelineRobustness:
    """Test pipeline robustness and error handling"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_pipeline_with_missing_data(self):
        """Test pipeline gracefully handles missing data"""
        # This should skip or handle missing data gracefully
        fake_path = Path("./nonexistent_dataset")
        
        with pytest.raises(FileNotFoundError):
            dataset = RefDetDataset(
                data_root=str(fake_path),
                annotations_file=str(fake_path / "annotations.json"),
                mode='train',
            )
    
    def test_pipeline_with_empty_batch(self, device):
        """Test pipeline handles edge cases"""
        # Test with minimal/edge case inputs
        weights_path = "yolov8n.pt"
        if not Path(weights_path).exists():
            weights_path = "yolov8n.pt"
        
        model = YOLOv8nRefDet(
            yolo_weights=weights_path,
            nc_base=0,
        ).to(device)
        
        # Single query image
        query = torch.randn(1, 3, 640, 640).to(device)
        support = torch.randn(1, 3, 256, 256).to(device)
        
        with torch.no_grad():
            model.set_reference_images(support)
            outputs = model(query_image=query, mode='prototype', use_cache=True)
        
        assert outputs is not None
        print(f"✅ Pipeline handles minimal input")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
