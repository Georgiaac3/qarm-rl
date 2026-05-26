#!/usr/bin/env python3
"""Debug script to inspect model checkpoint."""

import sys
from pathlib import Path
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.pixel_policy import PixelWiseGraspingPolicy
from config import get_config

model_path = Path(__file__).parent.parent / "training" / "checkpoints" / "policy.pt"

print(f"\n{'='*60}")
print(f"Inspecting model at: {model_path}")
print(f"{'='*60}")

if not model_path.exists():
    print(f"✗ Model not found at {model_path}")
    exit(1)

checkpoint = torch.load(model_path, map_location='cpu')

print(f"\nCheckpoint type: {type(checkpoint)}")
print(f"Checkpoint keys: {checkpoint.keys() if isinstance(checkpoint, dict) else 'N/A (not dict)'}")

if isinstance(checkpoint, dict):
    for key in checkpoint.keys():
        val = checkpoint[key]
        if isinstance(val, dict):
            print(f"\n{key}:")
            for k in list(val.keys())[:5]:  # First 5
                v = val[k]
                if isinstance(v, torch.Tensor):
                    print(f"  {k}: {v.shape}")
                else:
                    print(f"  {k}: {type(v)}")
        elif isinstance(val, torch.Tensor):
            print(f"\n{key}: Tensor {val.shape}")
        else:
            print(f"\n{key}: {type(val)}")

# Try to load as PixelWiseGraspingPolicy
print(f"\n{'='*60}")
print("Testing PixelWiseGraspingPolicy load...")
print(f"{'='*60}")

config = get_config()
device = config["model"]["device"]

model = PixelWiseGraspingPolicy(
    in_channels=5,
    hidden_channels=config["model"]["hidden_channels"],
)

try:
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    print("✓ Successfully loaded as PixelWiseGraspingPolicy")
    
    # Test forward pass
    print("\nTesting forward pass...")
    rgb = torch.randn(1, 3, 64, 64).to(device)
    depth = torch.randn(1, 1, 64, 64).to(device)
    heatmap = torch.randn(1, 1, 64, 64).to(device)
    
    with torch.no_grad():
        grasp_logits, velocity, confidence = model.forward(rgb, depth, heatmap)
    
    print(f"  grasp_logits: {grasp_logits.shape}, min={grasp_logits.min():.2f}, max={grasp_logits.max():.2f}")
    print(f"  velocity: {velocity.shape}, min={velocity.min():.2f}, max={velocity.max():.2f}")
    print(f"  confidence: {confidence.shape}, min={confidence.min():.2f}, max={confidence.max():.2f}")
    
    # Find best pixel
    score = torch.sigmoid(grasp_logits) * confidence
    score_flat = score[0, 0, :, :]
    pixel_idx = torch.argmax(score_flat)
    pixel_y = pixel_idx // 64
    pixel_x = pixel_idx % 64
    print(f"\n  Best pixel: ({pixel_x.item()}, {pixel_y.item()})")
    print(f"  Score map: min={score_flat.min():.4f}, max={score_flat.max():.4f}")
    print(f"  Score map mean: {score_flat.mean():.4f}")
    
    # Print score distribution
    print(f"\n  Score distribution:")
    for th in [0.1, 0.5, 0.7, 0.9]:
        count = (score_flat > th).sum().item()
        print(f"    Pixels with score > {th}: {count}")
    
except Exception as e:
    print(f"✗ Failed to load: {e}")
    import traceback
    traceback.print_exc()
