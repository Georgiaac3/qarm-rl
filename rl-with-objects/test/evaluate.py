#!/usr/bin/env python3

import sys
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

# Add parent dirs to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from environment.bin_picking_env import BinPickingEnv
from models.pixel_policy import PixelWiseGraspingPolicy
from config import get_config

# Try to import SAC for loading trained RL models
try:
    from stable_baselines3 import SAC
    HAS_SAC = True
except ImportError:
    HAS_SAC = False


def _resolve_path(path_str: str) -> Path:
    """Resolve relative paths from project root."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


class RandomGraspPolicy:
    """Baseline policy: random grasps near image center."""
    
    def __init__(self, image_width=64, image_height=64):
        self.image_width = image_width
        self.image_height = image_height
    
    def predict_action(self, rgb, depth, heatmap):
        """Return random grasp near center, random velocity."""
        # Gaussian distribution around center
        center_x = self.image_width / 2
        center_y = self.image_height / 2
        sigma = 10  # pixels
        
        pixel_x = int(np.clip(np.random.normal(center_x, sigma), 0, self.image_width - 1))
        pixel_y = int(np.clip(np.random.normal(center_y, sigma), 0, self.image_height - 1))
        
        # Random velocity
        velocity = np.random.uniform(-0.5, 0.5, size=2)
        confidence = 0.5
        
        return (pixel_x, pixel_y), velocity, confidence


class SACPolicyWrapper:
    """Wrapper to make SAC model compatible with evaluate_policy interface."""
    
    def __init__(self, sac_model, image_width=64, image_height=64):
        self.model = sac_model
        self.image_width = image_width
        self.image_height = image_height
    
    def predict_action(self, rgb, depth, heatmap):
        """
        Predict action using SAC model.
        
        SAC outputs: [grasp_x, grasp_y, vel_x, vel_y] in [-1, 1]
        Convert to: (pixel_x, pixel_y), velocity, confidence
        
        Args:
            rgb: numpy array or torch tensor
            depth: numpy array or torch tensor
            heatmap: numpy array or torch tensor
        """
        # Convert torch tensors to numpy if needed
        if torch.is_tensor(rgb):
            rgb = rgb.cpu().numpy()
        if torch.is_tensor(depth):
            depth = depth.cpu().numpy()
        if torch.is_tensor(heatmap):
            heatmap = heatmap.cpu().numpy()
        
        # Handle different tensor shapes from preprocessing
        # evaluate.py premultiplies RGB by 1/255.0 and permutes to (C, H, W)
        # SAC expects dict with (H, W, C) format after normalization
        
        # Convert from (C, H, W) to (H, W, C) if needed
        if rgb.ndim == 3 and rgb.shape[0] == 3:
            rgb = np.transpose(rgb, (1, 2, 0))
        if depth.ndim == 3 and depth.shape[0] == 1:
            depth = depth[0]
        if heatmap.ndim == 3 and heatmap.shape[0] == 1:
            heatmap = heatmap[0]
        
        # Stack observation dict for SAC
        obs = {
            'rgb': rgb,
            'depth': depth[..., np.newaxis] if depth.ndim == 2 else depth,
            'heatmap': heatmap[..., np.newaxis] if heatmap.ndim == 2 else heatmap,
        }
        
        # Get action from SAC
        action, _ = self.model.predict(obs, deterministic=True)
        
        # Parse action: [grasp_x, grasp_y, vel_x, vel_y]
        grasp_x_norm = float(action[0])  # [-1, 1]
        grasp_y_norm = float(action[1])  # [-1, 1]
        vel_x = float(action[2])
        vel_y = float(action[3])
        
        # Convert normalized coordinates to pixels
        pixel_x = int(np.clip((grasp_x_norm + 1) * self.image_width / 2, 0, self.image_width - 1))
        pixel_y = int(np.clip((grasp_y_norm + 1) * self.image_height / 2, 0, self.image_height - 1))
        
        # Confidence: use 1.0 (SAC is confident in its prediction)
        confidence = 1.0
        velocity = np.array([vel_x, vel_y])
        
        return (pixel_x, pixel_y), velocity, confidence


def evaluate_policy(
    model_path: str = "training/checkpoints/policy.pt",
    n_eval_episodes: int = 10,
    use_random: bool = False,
) -> dict:
    """
    Evaluate trained policy on multiple episodes.
    
    Args:
        model_path: Path to saved model checkpoint
        n_eval_episodes: Number of evaluation episodes
        use_random: Use random policy baseline instead of trained model
    
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\n{'='*60}")
    print(f"Evaluating {'Random' if use_random else 'Trained'} Policy")
    print(f"{'='*60}")
    
    # Load config and model
    config = get_config()
    device = config["model"]["device"]
    
    if use_random:
        model = RandomGraspPolicy(
            image_width=config["env"]["image_width"],
            image_height=config["env"]["image_height"],
        )
        print("✓ Using random policy baseline")
    else:
        # Resolve model path from project root
        model_path = str(_resolve_path(model_path))
        
        # Load checkpoint - detect format by extension
        if not Path(model_path).exists():
            print(f"✗ Model not found at {model_path}")
            print(f"   Available: .pt files or .zip files from SAC training")
            return None
        
        if model_path.endswith('.zip'):
            # Load SAC model from stable-baselines3
            if not HAS_SAC:
                print(f"✗ Cannot load SAC model: stable-baselines3 not available")
                print(f"   Install: pip install stable-baselines3")
                return None
            
            print(f"Loading SAC model from {model_path}...")
            sac_model = SAC.load(model_path, device=device)
            model = SACPolicyWrapper(
                sac_model,
                image_width=config["env"]["image_width"],
                image_height=config["env"]["image_height"],
            )
            print(f"✓ Loaded SAC model from {model_path}")
        else:
            # Load PixelWiseGraspingPolicy from .pt file
            model = PixelWiseGraspingPolicy(
                in_channels=5,
                hidden_channels=config["model"]["hidden_channels"],
            )
            
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint)
            model.to(device)
            model.eval()
            print(f"✓ Loaded PixelWiseGraspingPolicy from {model_path}")
    
    # Create environment
    env = BinPickingEnv(
        image_height=config["env"]["image_height"],
        image_width=config["env"]["image_width"],
        max_steps=config["env"]["max_steps"],
        n_objects=config["env"]["n_objects"],
    )
    
    # Run evaluation episodes
    episode_rewards = []
    episode_lengths = []
    grasp_successes = []
    final_object_counts = []
    
    for _ in tqdm(range(n_eval_episodes), desc="Evaluating"):
        obs, info = env.reset()
        episode_reward = 0.0
        done = False
        
        episode_grasps = 0
        episode_grasp_successes = 0
        
        while not done:
            # Prepare input
            rgb = torch.from_numpy(obs['rgb']/255.0).float().to(device).permute(2, 0, 1)
            depth = torch.from_numpy(obs['depth']).float().to(device).permute(2, 0, 1)
            heatmap = torch.from_numpy(obs['heatmap']).float().to(device).permute(2, 0, 1)
            
            # Get action from policy
            with torch.no_grad():
                pixel, velocity, confidence = model.predict_action(rgb, depth, heatmap)
            
            # Normalize pixel coordinates to [-1, 1]
            grasp_x = (pixel[1] / env.image_width) * 2 - 1   # pixel_y
            grasp_y = (pixel[0] / env.image_height) * 2 - 1  # pixel_x
            
            action = np.array([grasp_x, grasp_y, velocity[0], velocity[1]])
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            # Track grasp attempts
            episode_grasps += 1
            if info.get('grasp_success', False):
                episode_grasp_successes += 1
            
            done = terminated or truncated
        
        # Record episode statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(env.step_count)
        grasp_successes.append(episode_grasp_successes / max(1, episode_grasps))
        final_object_counts.append(len(env.objects))
    
    # Compute metrics
    metrics = {
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "max_reward": np.max(episode_rewards),
        "min_reward": np.min(episode_rewards),
        "mean_episode_length": np.mean(episode_lengths),
        "mean_grasp_success_rate": np.mean(grasp_successes),
        "mean_final_object_count": np.mean(final_object_counts),
    }
    
    # Print results
    print(f"\n{'='*60}")
    print("Evaluation Results")
    print(f"{'='*60}")
    print(f"Episodes evaluated:        {n_eval_episodes}")
    print(f"\nRewards:")
    print(f"  Mean:                    {metrics['mean_reward']:.2f}")
    print(f"  Std Dev:                 {metrics['std_reward']:.2f}")
    print(f"  Min/Max:                 {metrics['min_reward']:.2f} / {metrics['max_reward']:.2f}")
    print(f"\nEpisode Length:")
    print(f"  Mean steps:              {metrics['mean_episode_length']:.1f}")
    print(f"\nGrasping:")
    print(f"  Success rate:            {metrics['mean_grasp_success_rate']*100:.1f}%")
    print(f"  Final objects in bin:    {metrics['mean_final_object_count']:.1f}")
    print(f"\n{'='*60}")
    
    return metrics


def interactive_test(
    model_path: str = "training/checkpoints/policy.pt",
    n_steps: int = 50,
):
    """
    Interactive test - run one episode and show predictions at each step.
    
    Args:
        model_path: Path to saved model checkpoint
        n_steps: Maximum steps in episode
    """
    print(f"\n{'='*60}")
    print("Interactive Policy Test")
    print(f"{'='*60}")
    
    # Load config and model
    config = get_config()
    device = config["model"]["device"]
    
    # Resolve model path from project root
    model_path = str(_resolve_path(model_path))
    
    # Load checkpoint - detect format by extension
    if not Path(model_path).exists():
        print(f"✗ Model not found at {model_path}")
        return
    
    if model_path.endswith('.zip'):
        # Load SAC model from stable-baselines3
        if not HAS_SAC:
            print(f"✗ Cannot load SAC model: stable-baselines3 not available")
            return
        
        print(f"Loading SAC model from {model_path}...")
        sac_model = SAC.load(model_path, device=device)
        model = SACPolicyWrapper(
            sac_model,
            image_width=config["env"]["image_width"],
            image_height=config["env"]["image_height"],
        )
        print(f"✓ Loaded SAC model from {model_path}")
    else:
        # Load PixelWiseGraspingPolicy from .pt file
        model = PixelWiseGraspingPolicy(
            in_channels=5,
            hidden_channels=config["model"]["hidden_channels"],
        )
        
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        print(f"✓ Loaded PixelWiseGraspingPolicy from {model_path}")
    
    # Create environment
    env = BinPickingEnv(
        image_height=config["env"]["image_height"],
        image_width=config["env"]["image_width"],
        max_steps=n_steps,
        n_objects=config["env"]["n_objects"],
    )
    
    obs, info = env.reset()
    print(f"\n🎬 Episode started with {len(env.objects)} objects")
    
    step = 0
    episode_reward = 0.0
    done = False
    
    while not done and step < n_steps:
        # Prepare input
        rgb = torch.from_numpy(obs['rgb']/255.0).float().to(device).permute(2, 0, 1)
        depth = torch.from_numpy(obs['depth']).float().to(device).permute(2, 0, 1)
        heatmap = torch.from_numpy(obs['heatmap']).float().to(device).permute(2, 0, 1)
        
        # Get action from policy
        with torch.no_grad():
            pixel, velocity, confidence = model.predict_action(rgb, depth, heatmap)
        
        # Normalize pixel coordinates
        grasp_x = (pixel[1] / env.image_width) * 2 - 1
        grasp_y = (pixel[0] / env.image_height) * 2 - 1
        
        action = np.array([grasp_x, grasp_y, velocity[0], velocity[1]])
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        
        # Print step info
        print(f"\nStep {step+1}:")
        print(f"  Grasp pixel:             ({pixel[1]}, {pixel[0]})")
        print(f"  Velocity:                [{velocity[0]:.3f}, {velocity[1]:.3f}]")
        print(f"  Confidence:              {confidence:.3f}")
        print(f"  Reward:                  {reward:.2f}")
        if info.get('grasped'):
            print(f"  ✓ Grasped object!")
        
        done = terminated or truncated
        step += 1
    
    print(f"\n{'='*60}")
    print(f"Episode finished!")
    print(f"  Total steps:             {step}")
    print(f"  Total reward:            {episode_reward:.2f}")
    print(f"  Objects remaining:       {len(env.objects)}")
    print(f"{'='*60}")


