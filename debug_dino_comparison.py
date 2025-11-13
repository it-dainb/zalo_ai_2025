"""
Debug script to compare DINOv2 and DINOv3 architectures
"""
import torch
import timm

def print_separator(title):
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80)

def analyze_model(model_name: str, img_size: int = 224):
    """Load and analyze a DINO model"""
    print(f"\nLoading model: {model_name}")
    print(f"Image size: {img_size}×{img_size}")
    
    try:
        # Create model
        model = timm.create_model(
            model_name,
            pretrained=False,  # Don't download weights for quick debug
            num_classes=0,
            img_size=img_size,
        )
        
        # Get model info
        print(f"\n✓ Model loaded successfully!")
        print(f"  Embedding dimension: {model.embed_dim}")
        print(f"  Number of heads: {model.num_heads if hasattr(model, 'num_heads') else 'N/A'}")
        print(f"  Depth (blocks): {len(model.blocks) if hasattr(model, 'blocks') else 'N/A'}")
        print(f"  Patch size: {model.patch_embed.patch_size if hasattr(model, 'patch_embed') else 'N/A'}")
        
        # Test forward pass
        print(f"\nTesting forward pass...")
        dummy_input = torch.randn(2, 3, img_size, img_size)
        
        with torch.no_grad():
            features = model.forward_features(dummy_input)
        
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {features.shape}")
        print(f"  CLS token shape: {features[:, 0].shape}")
        
        # Show architecture
        print(f"\nModel Architecture:")
        print(f"  Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
        
        # Check for register tokens
        if hasattr(model, 'num_prefix_tokens'):
            print(f"  Prefix tokens: {model.num_prefix_tokens}")
        
        # Print first few layers
        print(f"\nFirst few components:")
        for name, module in list(model.named_children())[:5]:
            print(f"  - {name}: {type(module).__name__}")
        
        # Get data config for transforms
        data_config = timm.data.resolve_model_data_config(model)
        print(f"\nData Configuration:")
        print(f"  Mean: {data_config['mean']}")
        print(f"  Std: {data_config['std']}")
        print(f"  Input size: {data_config['input_size']}")
        
        return model, data_config
        
    except Exception as e:
        print(f"\n✗ Error loading model: {e}")
        return None, None

def main():
    print_separator("DINOv2 vs DINOv3 Architecture Comparison")
    
    # Test DINOv2 models
    print_separator("DINOv2 Models")
    
    print("\n[1] DINOv2 ViT-Small (reg4, lvd142m) - Current Default")
    dinov2_reg4, dinov2_reg4_config = analyze_model(
        "vit_small_patch14_reg4_dinov2.lvd142m",
        img_size=518
    )
    
    print("\n[2] DINOv2 ViT-Small (base, lvd142m)")
    dinov2_base, dinov2_base_config = analyze_model(
        "vit_small_patch14_dinov2.lvd142m",
        img_size=224
    )
    
    # Test DINOv3 models
    print_separator("DINOv3 Models")
    
    print("\n[3] DINOv3 ViT-Small (lvd1689m) - New Default")
    # Try different possible names for DINOv3
    possible_names = [
        "vit_small_patch16_dinov3.lvd1689m",
        "vit_small_patch14_dinov3.lvd1689m",
        "dinov3_vit_small_patch16_lvd1689m",
    ]
    
    dinov3_model = None
    dinov3_config = None
    
    for name in possible_names:
        print(f"\nTrying model name: {name}")
        model, config = analyze_model(name, img_size=224)
        if model is not None:
            dinov3_model = model
            dinov3_config = config
            print(f"✓ Successfully loaded: {name}")
            break
    
    if dinov3_model is None:
        print("\n⚠️  Could not load DINOv3 model with any attempted name.")
        print("Checking available DINO models in timm...")
        all_models = timm.list_models('*dino*')
        print(f"\nAvailable DINO models ({len(all_models)}):")
        for i, model_name in enumerate(all_models[:20], 1):  # Show first 20
            print(f"  {i}. {model_name}")
        if len(all_models) > 20:
            print(f"  ... and {len(all_models) - 20} more")
    
    # Summary comparison
    print_separator("Summary Comparison")
    
    if dinov2_reg4 and dinov3_model:
        print("\nDINOv2 (reg4) vs DINOv3:")
        print(f"  Embedding dim: {dinov2_reg4.embed_dim} vs {dinov3_model.embed_dim}")
        print(f"  Num blocks: {len(dinov2_reg4.blocks)} vs {len(dinov3_model.blocks)}")
        print(f"  Params: {sum(p.numel() for p in dinov2_reg4.parameters())/1e6:.2f}M vs "
              f"{sum(p.numel() for p in dinov3_model.parameters())/1e6:.2f}M")
        print(f"  Input size: {dinov2_reg4_config['input_size']} vs {dinov3_config['input_size']}")
    
    print_separator("End of Comparison")

if __name__ == "__main__":
    main()
