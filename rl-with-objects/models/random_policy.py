#!/usr/bin/env python3
"""Baseline random policy for testing environment."""

import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.pixel_policy import PixelWiseGraspingPolicy

class RandomGraspPolicy(torch.nn.Module):
    """Simple baseline: random grasps near image center."""
    
    def __init__(self):
        super().__init__()
        self.image_width = 64
        self.image_height = 64
    
    def predict_action(self, rgb, depth, heatmap):
        """Predict random grasp near center."""
        # Random pixel near center with gaussian distribution
        center_x = self.image_width / 2
        center_y = self.image_height / 2
        
        # Gaussian around center (sigma = 15 pixels)
        pixel_x = int(np.clip(np.random.normal(center_x, 15), 0, self.image_width - 1))
        pixel_y = int(np.clip(np.random.normal(center_y, 15), 0, self.image_height - 1))
        
        # Random velocity correction between -0.5 and 0.5
        velocity = np.random.uniform(-0.5, 0.5, size=2)
        
        # Confidence always 0.7
        confidence = 0.7
        
        return (pixel_x, pixel_y), velocity, confidence


def save_random_policy(save_path="training/checkpoints/policy_random.pt"):
    """Save a random policy model."""
    model = RandomGraspPolicy()
   
    # Create dummy weights for PixelWiseGraspingPolicy so we can still use load_state_dict
    real_model = PixelWiseGraspingPolicy(in_channels=5, hidden_channels=32)
    
    # Save just this random model's state
    torch.save(model.state_dict(), Path(__file__).parent.parent / save_path)
    print(f"✓ Saved random policy to {save_path}")


if __name__ == "__main__":
    save_random_policy()
