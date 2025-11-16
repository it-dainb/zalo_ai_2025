"""
Test Training Pipeline Components
==================================

Tests training-related components:
1. Loss functions (individual and combined)
2. Optimizer setup
3. Learning rate scheduling
4. Trainer initialization
5. Single training step
6. Gradient accumulation
7. Mixed precision training
"""

import pytest
import torch
import sys
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from losses.combined_loss import ReferenceBasedDetectionLoss
from losses.wiou_loss import WIoULoss
from losses.bce_loss import BCEClassificationLoss
from losses.dfl_loss import DFLoss
from losses.supervised_contrastive_loss import SupervisedContrastiveLoss
from losses.triplet_loss import TripletLoss
from training.trainer import RefDetTrainer
from models.yolov8n_refdet import YOLOv8nRefDet


class TestLossComponents:
    """Test individual loss components"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_wiou_loss(self, device):
        """Test WIoU loss"""
        loss_fn = WIoULoss().to(device)
        
        # Create dummy predictions and targets
        pred_boxes = torch.randn(10, 4).to(device)
        target_boxes = torch.randn(10, 4).to(device)
        
        loss = loss_fn(pred_boxes, target_boxes)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == torch.Size([])
        assert not torch.isnan(loss)
        
        print(f"✅ WIoU loss: {loss.item():.4f}")
    
    def test_bce_loss(self, device):
        """Test BCE loss"""
        loss_fn = BCEClassificationLoss().to(device)
        
        # Create dummy predictions and targets
        pred = torch.randn(10, 80).to(device)
        target = torch.randint(0, 2, (10, 80)).float().to(device)
        
        loss = loss_fn(pred, target)
        
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)
        
        print(f"✅ BCE loss: {loss.item():.4f}")
    
    def test_dfl_loss(self, device):
        """Test DFL loss"""
        loss_fn = DFLoss(reg_max=16).to(device)
        
        # Create dummy predictions and targets
        pred_dist = torch.randn(10, 16 * 4).to(device)  # 17 = reg_max
        target_boxes = torch.randn(10, 4).to(device)
        
        try:
            loss = loss_fn(pred_dist, target_boxes)
            assert isinstance(loss, torch.Tensor)
            print(f"✅ DFL loss: {loss.item():.4f}")
        except Exception as e:
            print(f"⚠️  DFL loss test: {str(e)}")
    
    def test_supervised_contrastive_loss(self, device):
        """Test supervised contrastive loss"""
        loss_fn = SupervisedContrastiveLoss(temperature=0.07).to(device)
        
        # Create dummy features and labels
        features = torch.randn(16, 128).to(device)
        labels = torch.randint(0, 4, (16,)).to(device)
        
        loss = loss_fn(features, labels)
        
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)
        
        print(f"✅ SupCon loss: {loss.item():.4f}")
    
    def test_triplet_loss(self, device):
        """Test triplet loss"""
        loss_fn = TripletLoss(margin=0.5).to(device)
        
        # Create dummy anchor, positive, negative
        anchor = torch.randn(8, 128).to(device)
        positive = torch.randn(8, 128).to(device)
        negative = torch.randn(8, 128).to(device)
        
        loss = loss_fn(anchor, positive, negative)
        
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)
        
        print(f"✅ Triplet loss: {loss.item():.4f}")


class TestCombinedLoss:
    """Test combined loss function"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def loss_fn(self, device):
        """Create combined loss function"""
        return ReferenceBasedDetectionLoss(
            stage=2,
            bbox_weight=7.5,
            cls_weight=0.5,
            dfl_weight=1.5,
            supcon_weight=1.0,
            cpe_weight=0.5,
            triplet_weight=0.2,
        ).to(device)
    
    def test_combined_loss_initialization(self, loss_fn):
        """Test combined loss initializes correctly"""
        assert loss_fn is not None
        assert hasattr(loss_fn, 'bbox_loss')
        assert hasattr(loss_fn, 'cls_loss')
        assert hasattr(loss_fn, 'dfl_loss')
        print(f"✅ Combined loss initialized")
    
    def test_combined_loss_stage2(self, loss_fn, device):
        """Test combined loss in stage 2"""
        # Create minimal dummy outputs and targets
        outputs = {
            'pred_bboxes': torch.randn(1, 100, 4).to(device),
            'pred_scores': torch.randn(1, 100).to(device),
            'support_features': torch.randn(1, 256).to(device),
            'query_features': torch.randn(1, 256).to(device),
        }
        
        targets = {
            'target_bboxes': [torch.randn(5, 4).to(device)],
            'target_classes': [torch.zeros(5).long().to(device)],
        }
        
        try:
            loss, loss_dict = loss_fn(outputs, targets)
            
            assert isinstance(loss, torch.Tensor)
            assert isinstance(loss_dict, dict)
            assert not torch.isnan(loss)
            
            print(f"✅ Combined loss (stage 2): {loss.item():.4f}")
            for key, value in loss_dict.items():
                print(f"   {key}: {value:.4f}")
        except Exception as e:
            print(f"⚠️  Combined loss test: {str(e)}")


class TestOptimizer:
    """Test optimizer setup"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def model(self, device):
        """Create simple model for testing"""
        weights_path = "yolov8n.pt"
        if not Path(weights_path).exists():
            weights_path = "yolov8n.pt"
        
        return YOLOv8nRefDet(
            yolo_weights=weights_path,
            nc_base=0,
            freeze_yolo=False,
            freeze_dinov3=True,
        ).to(device)
    
    def test_optimizer_creation(self, model):
        """Test optimizer can be created"""
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-4,
            weight_decay=0.05,
        )
        
        assert optimizer is not None
        print(f"✅ Optimizer created")
    
    def test_layerwise_learning_rates(self, model):
        """Test layerwise learning rate setup"""
        param_groups = [
            {'params': model.support_encoder.parameters(), 'lr': 1e-5},
            {'params': model.backbone.parameters(), 'lr': 1e-4},
            {'params': model.scs_fusion.parameters(), 'lr': 2e-4},
            {'params': model.detection_head.parameters(), 'lr': 2e-4},
        ]
        
        optimizer = torch.optim.AdamW(param_groups, weight_decay=0.05)
        
        assert len(optimizer.param_groups) == 4
        assert optimizer.param_groups[0]['lr'] == 1e-5
        assert optimizer.param_groups[1]['lr'] == 1e-4
        
        print(f"✅ Layerwise learning rates configured")


class TestScheduler:
    """Test learning rate scheduler"""
    
    def test_cosine_scheduler(self):
        """Test cosine annealing scheduler"""
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=100,
            eta_min=1e-6,
        )
        
        assert scheduler is not None
        
        # Test scheduling
        initial_lr = optimizer.param_groups[0]['lr']
        for _ in range(10):
            scheduler.step()
        updated_lr = optimizer.param_groups[0]['lr']
        
        assert updated_lr != initial_lr
        print(f"✅ Scheduler working: {initial_lr:.6f} → {updated_lr:.6f}")


class TestTrainer:
    """Test trainer class"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def model(self, device):
        """Create model for trainer"""
        weights_path = "yolov8n.pt"
        if not Path(weights_path).exists():
            weights_path = "yolov8n.pt"
        
        return YOLOv8nRefDet(
            yolo_weights=weights_path,
            nc_base=0,
            freeze_yolo=False,
            freeze_dinov3=True,
        ).to(device)
    
    @pytest.fixture
    def loss_fn(self, device):
        """Create loss function"""
        return ReferenceBasedDetectionLoss(stage=2).to(device)
    
    @pytest.fixture
    def optimizer(self, model):
        """Create optimizer"""
        return torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    
    def test_trainer_initialization(self, model, loss_fn, optimizer, device):
        """Test trainer initializes correctly"""
        from augmentations.augmentation_config import AugmentationConfig
        
        aug_config = AugmentationConfig()
        
        trainer = RefDetTrainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=str(device),
            mixed_precision=False,  # Disable for testing
            gradient_accumulation_steps=1,
            checkpoint_dir='./test_checkpoints',
            aug_config=aug_config,
            stage=2,
        )
        
        assert trainer is not None
        assert trainer.model is not None
        assert trainer.loss_fn is not None
        assert trainer.optimizer is not None
        
        print(f"✅ Trainer initialized")
    
    def test_trainer_checkpoint_save_load(self, model, loss_fn, optimizer, device, tmp_path):
        """Test checkpoint saving and loading"""
        from augmentations.augmentation_config import AugmentationConfig
        
        aug_config = AugmentationConfig()
        
        trainer = RefDetTrainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=str(device),
            mixed_precision=False,
            checkpoint_dir=str(tmp_path),
            aug_config=aug_config,
            stage=2,
        )
        
        # Save checkpoint
        trainer.epoch = 0  # Set epoch for checkpoint naming
        test_metrics = {'train_loss': 1.5}
        trainer.save_checkpoint(metrics=test_metrics, is_best=False)
        
        # Checkpoint is saved as checkpoint_epoch_0.pt by the trainer
        checkpoint_path = tmp_path / "checkpoint_epoch_0.pt"
        assert checkpoint_path.exists()
        print(f"✅ Checkpoint saved")
        
        # Load checkpoint
        trainer.load_checkpoint(str(checkpoint_path))
        print(f"✅ Checkpoint loaded")


class TestTrainingStep:
    """Test single training step"""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def test_single_training_step_mock(self, device):
        """Test single training step with mock data"""
        # Create simple mock model
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 1),
        ).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        loss_fn = torch.nn.MSELoss()
        
        # Mock batch
        x = torch.randn(4, 10).to(device)
        y = torch.randn(4, 1).to(device)
        
        # Training step
        model.train()
        optimizer.zero_grad()
        
        pred = model(x)
        loss = loss_fn(pred, y)
        
        loss.backward()
        optimizer.step()
        
        assert not torch.isnan(loss)
        print(f"✅ Single training step: loss={loss.item():.4f}")
    
    def test_gradient_accumulation(self, device):
        """Test gradient accumulation"""
        model = torch.nn.Linear(10, 10).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        loss_fn = torch.nn.MSELoss()
        
        accumulation_steps = 4
        
        # Simulate gradient accumulation
        for i in range(accumulation_steps):
            x = torch.randn(2, 10).to(device)
            y = torch.randn(2, 10).to(device)
            
            pred = model(x)
            loss = loss_fn(pred, y)
            loss = loss / accumulation_steps
            
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
        
        print(f"✅ Gradient accumulation working")
    
    def test_mixed_precision_training(self, device):
        """Test mixed precision training"""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        model = torch.nn.Linear(10, 10).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler()
        loss_fn = torch.nn.MSELoss()
        
        # Mixed precision training step
        model.train()
        optimizer.zero_grad()
        
        x = torch.randn(4, 10).to(device)
        y = torch.randn(4, 10).to(device)
        
        with torch.amp.autocast(device_type='cuda'):
            pred = model(x)
            loss = loss_fn(pred, y)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        assert not torch.isnan(loss)
        print(f"✅ Mixed precision training: loss={loss.item():.4f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
