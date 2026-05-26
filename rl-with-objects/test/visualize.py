#!/usr/bin/env python3
"""
Visualization script for bin picking predictions.

Visualizes:
- Input multi-modal data (RGB, depth, heatmap)
- Model predictions (grasp probability, velocity field)
- Predicted grasp point and velocity vector
"""

import sys
from pathlib import Path
import numpy as np
import torch
import cv2

# Add parent dirs to path
sys.path.insert(0, str(Path(__file__).parent))

from environment.bin_picking_env import BinPickingEnv
from models.pixel_policy import PixelWiseGraspingPolicy
from config import get_config


def visualize_predictions(
    model_path: str = "training/checkpoints/policy.pt",
    save_dir: str = "visualization_outputs",
):
    """
    Visualize model predictions on a single environment observation.
    
    Args:
        model_path: Path to saved model checkpoint
        save_dir: Directory to save visualization images
    """
    print(f"\n{'='*60}")
    print("Visualizing Model Predictions")
    print(f"{'='*60}")
    
    # Create output directory
    Path(save_dir).mkdir(exist_ok=True)
    
    # Load config and model
    config = get_config()
    device = config["model"]["device"]
    
    model = PixelWiseGraspingPolicy(
        in_channels=5,
        hidden_channels=config["model"]["hidden_channels"],
    )
    
    # Load checkpoint
    if not Path(model_path).exists():
        print(f"✗ Model not found at {model_path}")
        return
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    print(f"✓ Loaded model from {model_path}")
    
    # Create environment
    env = BinPickingEnv(
        image_height=config["env"]["image_height"],
        image_width=config["env"]["image_width"],
        max_steps=config["env"]["max_steps"],
        n_objects=config["env"]["n_objects"],
    )
    
    obs, info = env.reset()
    print(f"✓ Environment reset with {len(env.objects)} objects")
    
    # Prepare input
    rgb = obs['rgb'].copy()
    depth = obs['depth'].copy()
    heatmap = obs['heatmap'].copy()
    
    # Convert to tensors
    rgb_tensor = torch.from_numpy(rgb/255.0).float().to(device).permute(2, 0, 1)
    depth_tensor = torch.from_numpy(depth).float().to(device).permute(2, 0, 1)
    heatmap_tensor = torch.from_numpy(heatmap).float().to(device).permute(2, 0, 1)
    
    # Get predictions
    with torch.no_grad():
        grasp_logits, velocity_pred, confidence_pred = model(
            rgb_tensor.unsqueeze(0),
            depth_tensor.unsqueeze(0),
            heatmap_tensor.unsqueeze(0)
        )
        pixel, velocity, confidence = model.predict_action(rgb_tensor, depth_tensor, heatmap_tensor)
    
    # Convert predictions to numpy
    grasp_prob = torch.sigmoid(grasp_logits[0, 0]).cpu().numpy()
    velocity_field = velocity_pred[0].permute(1, 2, 0).cpu().numpy()
    conf_map = torch.sigmoid(confidence_pred[0, 0]).cpu().numpy()
    
    H, W = env.image_height, env.image_width
    
    # ===== Visualization 1: Input Data
    fig_input = np.zeros((H, W*3, 3), dtype=np.uint8)
    
    # RGB channel
    fig_input[:, :W] = rgb
    
    # Depth normalized to RGB
    depth_normalized = ((depth - depth.min()) / (depth.max() - depth.min() + 1e-6) * 255).astype(np.uint8)
    depth_rgb = cv2.cvtColor(depth_normalized[:, :, 0], cv2.COLOR_GRAY2BGR)
    fig_input[:, W:2*W] = depth_rgb
    
    # Heatmap normalized to RGB
    heatmap_normalized = ((heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6) * 255).astype(np.uint8)
    heatmap_rgb = cv2.cvtColor(heatmap_normalized[:, :, 0], cv2.COLOR_GRAY2BGR)
    heatmap_rgb = cv2.applyColorMap(heatmap_rgb, cv2.COLORMAP_JET)
    fig_input[:, 2*W:3*W] = heatmap_rgb
    
    # Add labels
    cv2.putText(fig_input, "RGB", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(fig_input, "Depth", (W+10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(fig_input, "Heatmap", (2*W+10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    output_path = Path(save_dir) / "01_input_data.png"
    cv2.imwrite(str(output_path), cv2.cvtColor(fig_input, cv2.COLOR_RGB2BGR))
    print(f"✓ Saved: {output_path}")
    
    # ===== Visualization 2: Model Predictions
    fig_pred = np.zeros((H, W*3, 3), dtype=np.uint8)
    
    # Grasp probability
    grasp_rgb = cv2.applyColorMap((grasp_prob * 255).astype(np.uint8), cv2.COLORMAP_JET)
    fig_pred[:, :W] = grasp_rgb
    
    # Confidence map
    conf_rgb = cv2.applyColorMap((conf_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
    fig_pred[:, W:2*W] = conf_rgb
    
    # Velocity magnitude
    vel_magnitude = np.sqrt(velocity_field[:, :, 0]**2 + velocity_field[:, :, 1]**2)
    vel_mag_normalized = ((vel_magnitude - vel_magnitude.min()) / (vel_magnitude.max() - vel_magnitude.min() + 1e-6) * 255).astype(np.uint8)
    vel_rgb = cv2.applyColorMap(vel_mag_normalized, cv2.COLORMAP_VIRIDIS)
    fig_pred[:, 2*W:3*W] = vel_rgb
    
    # Add labels
    cv2.putText(fig_pred, "Grasp Prob", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(fig_pred, "Confidence", (W+10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(fig_pred, "Velocity Mag", (2*W+10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    output_path = Path(save_dir) / "02_model_predictions.png"
    cv2.imwrite(str(output_path), cv2.cvtColor(fig_pred, cv2.COLOR_RGB2BGR))
    print(f"✓ Saved: {output_path}")
    
    # ===== Visualization 3: Best Grasp + Velocity Field
    fig_action = rgb.copy()
    
    # Draw velocity field (simplified, every Nth pixel)
    step = 8
    for y in range(0, H, step):
        for x in range(0, W, step):
            vx = velocity_field[y, x, 0] * 20  # Scale for visibility
            vy = velocity_field[y, x, 1] * 20
            
            if np.sqrt(vx**2 + vy**2) > 0.5:
                cv2.arrowedLine(fig_action, (x, y), 
                               (int(x + vx), int(y + vy)),
                               (0, 255, 0), 1, tipLength=0.3)
    
    # Draw best grasp point
    px, py = pixel
    cv2.circle(fig_action, (py, px), 8, (255, 0, 0), 2)  # Blue circle
    
    # Draw velocity vector from best grasp
    vel_x = velocity[0] * 30
    vel_y = velocity[1] * 30
    cv2.arrowedLine(fig_action, (py, px),
                   (int(py + vel_y), int(px + vel_x)),
                   (0, 0, 255), 2, tipLength=0.2)
    
    # Add text info
    cv2.putText(fig_action, f"Best Grasp: ({py}, {px})", (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(fig_action, f"Velocity: [{velocity[0]:.2f}, {velocity[1]:.2f}]", (10, 45),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(fig_action, f"Confidence: {confidence:.3f}", (10, 65),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    output_path = Path(save_dir) / "03_grasp_and_velocity.png"
    cv2.imwrite(str(output_path), cv2.cvtColor(fig_action, cv2.COLOR_RGB2BGR))
    print(f"✓ Saved: {output_path}")
    
    print(f"\n{'='*60}")
    print(f"Visualization complete! Images saved to {save_dir}/")
    print(f"{'='*60}")
    
    # Print prediction details
    print(f"\nPrediction Summary:")
    print(f"  Best grasp pixel:    ({pixel[1]}, {pixel[0]})")
    print(f"  Grasp probability:   {grasp_prob[pixel[0], pixel[1]]:.3f}")
    print(f"  Velocity:            [{velocity[0]:.3f}, {velocity[1]:.3f}]")
    print(f"  Confidence:          {confidence:.3f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize model predictions")
    parser.add_argument("--model", type=str, default="training/checkpoints/policy.pt",
                        help="Path to saved model checkpoint")
    parser.add_argument("--output", type=str, default="visualization_outputs",
                        help="Output directory for visualization images")
    
    args = parser.parse_args()
    visualize_predictions(args.model, args.output)
