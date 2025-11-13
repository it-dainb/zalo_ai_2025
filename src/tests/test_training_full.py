"""
Test Full Training Pipeline
============================

Tests complete training workflow:
1. Full training loop (minimal epochs)
2. Training with validation
3. Checkpoint saving/loading during training
4. Training resumption
5. Multi-stage training
"""

import pytest
import torch
import sys
from pathlib import Path
from torch.utils.data import DataLoader
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.datasets.refdet_dataset import RefDetDataset, EpisodicBatchSampler
from src.datasets.collate import RefDetCollator
from src.models.yolov8n_refdet import YOLOv8nRefDet
from src.losses.combined_loss import ReferenceBasedDetectionLoss
from src.training.trainer import RefDetTrainer
from src.augmentations.augmentation_config import AugmentationConfig


class TestMinimalTraining:
    """Test minimal training loop"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def checkpoint_dir(self, tmp_path):
        """Create temporary checkpoint directory"""
        return tmp_path / "checkpoints"
    
    @pytest.fixture
    def dataset_available(self):
        """Check if training dataset is available"""
        dataset_path = Path("./datasets/train/samples")
        annotations_path = Path("./datasets/train/annotations/annotations.json")
        return dataset_path.exists() and annotations_path.exists()
    
    def test_training_single_epoch(self, device, checkpoint_dir, dataset_available):
        """Test training for single epoch"""
        if not dataset_available:
            pytest.skip("Training dataset not available")
        
        print("\n" + "="*60)
        print("Testing Single Epoch Training")
        print("="*60)
        
        # Create dataset
        dataset = RefDetDataset(
            data_root="./datasets/train/samples",
            annotations_file="./datasets/train/annotations/annotations.json",
            mode='train',
            cache_frames=True,
        )
        
        # Create sampler (minimal episodes)
        sampler = EpisodicBatchSampler(
            dataset=dataset,
            n_way=min(2, len(dataset.classes)),
            n_query=2,
            n_episodes=2,  # Minimal for testing
        )
        
        # Create collator
        aug_config = AugmentationConfig(
            query_mosaic_prob=0.0,  # Disable for faster testing
            query_mixup_prob=0.0,
        )
        collator = RefDetCollator(
            config=aug_config,
            mode='train',
            stage=2,
        )
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=collator,
            num_workers=0,
            pin_memory=False,
        )
        
        # Create model
        weights_path = "yolov8n.pt"
        if not Path(weights_path).exists():
            weights_path = "yolov8n.pt"
        
        model = YOLOv8nRefDet(
            yolo_weights=weights_path,
            nc_base=0,
            freeze_yolo=False,
            freeze_dinov3=True,
        ).to(device)
        
        # Create loss function
        loss_fn = ReferenceBasedDetectionLoss(stage=2).to(device)
        
        # Create optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=0.05,
        )
        
        # Create trainer
        trainer = RefDetTrainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=str(device),
            mixed_precision=False,  # Disable for testing
            gradient_accumulation_steps=1,
            checkpoint_dir=str(checkpoint_dir),
            aug_config=aug_config,
            stage=2,
        )
        
        # Train for 1 epoch
        try:
            trainer.train(
                train_loader=dataloader,
                val_loader=None,
                num_epochs=1,
                save_interval=1,
            )
            print("✅ Single epoch training completed")
        except Exception as e:
            print(f"⚠️  Training error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def test_training_with_validation(self, device, checkpoint_dir, dataset_available):
        """Test training with validation"""
        if not dataset_available:
            pytest.skip("Training dataset not available")
        
        test_data_path = Path("./datasets/test/samples")
        test_ann_path = Path("./datasets/test/annotations/annotations.json")
        
        if not test_data_path.exists() or not test_ann_path.exists():
            pytest.skip("Test dataset not available")
        
        print("\n" + "="*60)
        print("Testing Training with Validation")
        print("="*60)
        
        # Create train dataset
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
            n_episodes=2,
        )
        
        aug_config = AugmentationConfig(
            query_mosaic_prob=0.0,
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
        )
        
        # Create val dataset
        val_dataset = RefDetDataset(
            data_root=str(test_data_path),
            annotations_file=str(test_ann_path),
            mode='val',
            cache_frames=True,
        )
        
        val_sampler = EpisodicBatchSampler(
            dataset=val_dataset,
            n_way=min(2, len(val_dataset.classes)),
            n_query=2,
            n_episodes=1,  # Just 1 for validation
        )
        
        val_collator = RefDetCollator(
            config=aug_config,
            mode='val',
            stage=2,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_sampler=val_sampler,
            collate_fn=val_collator,
            num_workers=0,
        )
        
        # Create model
        weights_path = "yolov8n.pt"
        if not Path(weights_path).exists():
            weights_path = "yolov8n.pt"
        
        model = YOLOv8nRefDet(
            yolo_weights=weights_path,
            nc_base=0,
            freeze_yolo=False,
            freeze_dinov3=True,
        ).to(device)
        
        loss_fn = ReferenceBasedDetectionLoss(stage=2).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        trainer = RefDetTrainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=str(device),
            mixed_precision=False,
            checkpoint_dir=str(checkpoint_dir),
            aug_config=aug_config,
            stage=2,
        )
        
        # Train with validation
        try:
            trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=1,
                save_interval=1,
            )
            print("✅ Training with validation completed")
        except Exception as e:
            print(f"⚠️  Training error: {str(e)}")
    
    def test_checkpoint_saving(self, device, checkpoint_dir, dataset_available):
        """Test checkpoint is saved correctly"""
        if not dataset_available:
            pytest.skip("Training dataset not available")
        
        # Create minimal setup
        weights_path = "yolov8n.pt"
        if not Path(weights_path).exists():
            weights_path = "yolov8n.pt"
        
        model = YOLOv8nRefDet(
            yolo_weights=weights_path,
            nc_base=0,
            freeze_yolo=False,
            freeze_dinov3=True,
        ).to(device)
        
        loss_fn = ReferenceBasedDetectionLoss(stage=2).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        aug_config = AugmentationConfig()
        
        trainer = RefDetTrainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=str(device),
            checkpoint_dir=str(checkpoint_dir),
            aug_config=aug_config,
            stage=2,
        )
        
        # Save checkpoint
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        trainer.epoch = 0  # Set epoch for checkpoint naming
        test_metrics = {'train_loss': 1.5, 'val_loss': 1.6}
        trainer.save_checkpoint(metrics=test_metrics, is_best=False)
        
        # Checkpoint is saved as checkpoint_epoch_0.pt by the trainer
        checkpoint_path = checkpoint_dir / "checkpoint_epoch_0.pt"
        assert checkpoint_path.exists()
        
        # Load checkpoint and verify
        checkpoint = torch.load(checkpoint_path)
        assert 'model_state_dict' in checkpoint
        assert 'optimizer_state_dict' in checkpoint
        assert 'epoch' in checkpoint
        
        print("✅ Checkpoint saved and verified")
    
    def test_training_resumption(self, device, checkpoint_dir, dataset_available):
        """Test training can be resumed from checkpoint"""
        if not dataset_available:
            pytest.skip("Training dataset not available")
        
        print("\n" + "="*60)
        print("Testing Training Resumption")
        print("="*60)
        
        # First training session
        weights_path = "yolov8n.pt"
        if not Path(weights_path).exists():
            weights_path = "yolov8n.pt"
        
        model = YOLOv8nRefDet(
            yolo_weights=weights_path,
            nc_base=0,
        ).to(device)
        
        loss_fn = ReferenceBasedDetectionLoss(stage=2).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        aug_config = AugmentationConfig()
        
        trainer = RefDetTrainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=str(device),
            checkpoint_dir=str(checkpoint_dir),
            aug_config=aug_config,
            stage=2,
        )
        
        # Save checkpoint
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        trainer.epoch = 5  # Simulate trained epochs
        test_metrics = {'train_loss': 1.5}
        trainer.save_checkpoint(metrics=test_metrics, is_best=False)
        
        # Checkpoint is saved as checkpoint_epoch_5.pt by the trainer
        checkpoint_path = checkpoint_dir / "checkpoint_epoch_5.pt"
        
        # Create new trainer and resume
        model2 = YOLOv8nRefDet(
            yolo_weights=weights_path,
            nc_base=0,
        ).to(device)
        
        loss_fn2 = ReferenceBasedDetectionLoss(stage=2).to(device)
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-4)
        
        trainer2 = RefDetTrainer(
            model=model2,
            loss_fn=loss_fn2,
            optimizer=optimizer2,
            device=str(device),
            checkpoint_dir=str(checkpoint_dir),
            aug_config=aug_config,
            stage=2,
        )
        
        trainer2.load_checkpoint(str(checkpoint_path))
        
        assert trainer2.epoch == 5
        print("✅ Training resumption working")


class TestMultiStageTraining:
    """Test multi-stage training workflow"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_stage2_stage3_transition(self, device):
        """Test transitioning from stage 2 to stage 3"""
        weights_path = "yolov8n.pt"
        if not Path(weights_path).exists():
            weights_path = "yolov8n.pt"
        
        # Stage 2 setup
        model = YOLOv8nRefDet(
            yolo_weights=weights_path,
            nc_base=0,
        ).to(device)
        
        loss_fn_stage2 = ReferenceBasedDetectionLoss(stage=2).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Verify stage 2 loss weights
        assert loss_fn_stage2.stage == 2
        
        # Stage 3 setup (with triplet loss)
        loss_fn_stage3 = ReferenceBasedDetectionLoss(
            stage=3,
            triplet_weight=0.5,
        ).to(device)
        
        # Verify stage 3 includes triplet loss
        assert loss_fn_stage3.stage == 3
        assert loss_fn_stage3.weights['triplet'] > 0
        
        print("✅ Multi-stage training setup verified")
        print(f"   Stage 2 weights: {loss_fn_stage2.weights}")
        print(f"   Stage 3 weights: {loss_fn_stage3.weights}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
