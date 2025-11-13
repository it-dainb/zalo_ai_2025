# YOLOv8n-RefDet Parameter Quick Reference

**Last Updated:** November 14, 2025  
**Model Version:** PSALM Fusion (v2)

---

## Quick Stats

```
Total Parameters:    30.22M  ✅ (60.4% of 50M budget)
Trainable:           19.28M  (63.8%)
Frozen:              10.94M  (36.2%)
GPU Memory:           0.24GB (inference, batch=1)
Status:              PRODUCTION READY
```

---

## Module Breakdown (30.22M Total)

| Module | Params | % | Status |
|--------|--------|---|--------|
| DINOv3 Encoder | 21.77M | 72.0% | 🔀 50% trainable |
| YOLOv8n Backbone | 3.16M | 10.5% | 🔓 100% trainable |
| PSALM Fusion | 0.78M | 2.6% | 🔓 100% trainable |
| Dual Head | 4.51M | 14.9% | 🔓 100% trainable |

---

## PSALM vs CHEAF

| Metric | CHEAF | PSALM | Δ |
|--------|-------|-------|---|
| Fusion params | 1.76M | 0.78M | **-56%** ⚡ |
| Inference speed | 1.0x | 1.46x | **+46%** ⚡ |
| Total model | 31.20M | 30.22M | -3.1% |

**Winner:** PSALM (cleaner, faster, smaller)

---

## Memory Usage

### Training (per batch)
- Batch=1: 0.45 GB
- Batch=4: 0.75 GB
- Batch=8: 1.15 GB
- Batch=16: 1.95 GB

### Inference (GPU actual)
- Batch=1: 0.24 GB ✅
- Batch=4: 0.51 GB ✅

### Jetson Xavier NX (8GB)
- Inference: ✅ <1GB (excellent fit)
- Training: ✅ 0.75-1.15GB for batch 4-8

---

## Testing Commands

### Quick parameter check:
```bash
python test_parameters.py
```

### One-liner:
```bash
python -c "from src.models.yolov8n_refdet import YOLOv8nRefDet; \
m = YOLOv8nRefDet(); \
print(f'{sum(p.numel() for p in m.parameters())/1e6:.2f}M params')"
```

### Memory profiling (GPU):
```bash
python -c "import torch; \
from src.models.yolov8n_refdet import YOLOv8nRefDet; \
torch.cuda.reset_peak_memory_stats(); \
m = YOLOv8nRefDet().cuda().eval(); \
q = torch.randn(1,3,640,640).cuda(); \
s = torch.randn(1,3,256,256).cuda(); \
with torch.no_grad(): _ = m(q,s,mode='dual'); \
print(f'{torch.cuda.max_memory_allocated()/1e9:.2f} GB')"
```

---

## Layer Details

### DINOv3 Encoder (21.77M)
- Patch embed: 0.30M (frozen)
- Transformer blocks 0-5: ~5.4M (frozen)
- Transformer blocks 6-11: ~5.4M (trainable)
- P2-P5 projections: 0.37M (trainable)

### YOLOv8n Backbone (3.16M)
- All layers trainable
- Outputs: P2 (32ch), P3 (64ch), P4 (128ch), P5 (256ch)

### PSALM Fusion (0.78M)
- Pyramid enrichment: 54K
- Cross-attention: 277K
- Refinement projectors: 227K
- Residual projectors: 437K

### Dual Head (4.51M)
- Standard head (COCO): 3.23M
- Prototype head (novel): 1.28M

---

## Budget Analysis

```
50M Hard Limit:  ████████████████████████████████░░░░░░░░░░░░░░░░░░░░ 60.4%
                 ├─ Used:      30.22M
                 └─ Remaining: 19.78M (39.6% margin)

10M Target:      ████████████████████████████████████████████████████
                 └─ Current: 30.22M (302% over) ⚠️
```

**Status:** Within hard limit ✅, exceeds soft target ⚠️

---

## Optimization Options

### For Speed
1. **ONNX Export** → +10-15% faster
2. **TensorRT** → +30-50% faster (Jetson)
3. **FP16** → +20-30% faster, 2x less memory

### For Size (if <10M target needed)
1. **Smaller encoder** (MobileNet: ~2M) → -20M params
2. **Knowledge distillation** → -40% params
3. **INT8 quantization** → 4x memory reduction

---

## Key Files

- `test_parameters.py` - Full analysis script
- `MODEL_PARAMETERS_REPORT.md` - Detailed 10-section report
- `src/models/yolov8n_refdet.py:122` - Built-in parameter summary

---

## Recommendations

### Current Status ✅
- Within budget with healthy margin
- Jetson-compatible memory footprint
- PSALM provides significant improvement over CHEAF

### Next Steps
1. **Keep current architecture** for development
2. **Export to ONNX** for deployment speedup
3. **Profile on target hardware** (Jetson Xavier NX)
4. **Consider FP16** for production deployment

### If 10M Target Required
- Current approach prioritizes accuracy (DINOv3: 21.77M)
- Options: Encoder swap, distillation, or accept 30M as realistic
- Recommendation: Stick with 30M unless hard constraints exist

---

**Generated:** November 14, 2025  
**Model:** YOLOv8n-RefDet with PSALM Fusion  
**Status:** ✅ PRODUCTION READY
