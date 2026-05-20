# ============================================================================
# ENVIRONMENT CONFIG
# ============================================================================
ENV_CONFIG = {
    "max_steps": 200,
    "gravity": 9.81,
    "air_resistance": 0.1,
    "projectile_radius": 0.025,
    "dt": 0.01,  # Physics timestep
}

# Curriculum learning: start easy, progress to harder
CURRICULUM_CONFIG = {
    "use_curriculum": False,  # ⬅️ Set to False for accurate eval logging (SB3 callback bug with multiple learn() calls)
    "initial_range": (0.5, 1.5),  # Easy: close targets
    "final_range": (1.0, 5.0),  # Hard: distant targets
}

# ============================================================================
# REWARD SHAPING CONFIG
# ============================================================================
REWARD_CONFIG = {
    # Distance penalty scaling
    "distance_penalty_scale": 5.0,  # Max penalty magnitude
    "max_distance": 7.0,  # Reference distance for normalization
    # Progress reward for getting closer
    "progress_scale": 0.5,  # Reward per meter of improvement
    # Hit bonuses at different distance thresholds
    "hit_bonuses": {
        "direct_hit": 100.0,  # distance < projectile_radius * 2
        "very_close": 50.0,  # distance < projectile_radius * 5
        "close": 20.0,  # distance < 0.2m
        "medium_close": 5.0,  # distance < 0.5m
    },
    # Time penalty (discourage long episodes)
    "time_penalty_scale": 0.0005,
}

# ============================================================================
# TRAINING CONFIG - Tune this for your hardware!
# ============================================================================
TRAINING_CONFIG = {
    # Training duration
    "total_timesteps": 50_000,  # ⬅️ GPU: 100K-200K, CPU: 20K-50K
    # Evaluation
    "eval_freq": 5_000,  # How often to evaluate (not too frequent to avoid slowdown)
    "n_eval_episodes": 10,  # Episodes per evaluation
    # Checkpoint saving
    "checkpoint_freq": 5_000,  # Save model every N steps
}

# ============================================================================
# SAC HYPERPARAMETERS - Algorithm settings
# ============================================================================
SAC_CONFIG = {
    # Learning rate - CRITICAL for stability
    "learning_rate": 5e-5,  # ⬅️ GPU: 1e-4-3e-4, CPU: 3e-5-5e-5
    # Replay buffer
    "buffer_size": 2_000,  # Smaller = fresher experience
    "learning_starts": 1_000,  # Start learning after random exploration
    "batch_size": 128,  # ⬅️ GPU: 256, CPU: 128
    # Network updates
    "train_freq": 1,  # Update every step
    "gradient_steps": 1,  # One gradient step per update
    "tau": 0.01,  # Soft update coefficient (smaller = slower)
    # Entropy regularization (exploration)
    "ent_coef": 0.02,  # Fixed value, NO auto (auto was chaotic)
    "target_entropy": -2.0,  # Target for entropy
    # Discount factor
    "gamma": 0.99,
    # Neural network architecture
    "net_arch": [128, 128],  # ⬅️ GPU: [256, 256], CPU: [128, 128]
    # Seed for reproducibility
    "seed": 42,
    "device": "auto",  # cuda if available, else cpu
}

# ============================================================================
# HARDWARE PRESETS
# ============================================================================
# Use these to quickly switch between CPU and GPU configs:
#
# CPU PRESET:
#   - total_timesteps: 50_000
#   - learning_rate: 5e-5
#   - batch_size: 128
#   - net_arch: [128, 128]
#
# GPU PRESET:
#   - total_timesteps: 150_000
#   - learning_rate: 2e-4
#   - batch_size: 256
#   - net_arch: [256, 256]
#


def get_config(device: str = "auto"):
    """
    Get configuration adjusted for device type.

    Args:
        device: "auto" (detect), "cuda" (force GPU), "cpu" (force CPU)

    Returns:
        Tuple of (ENV_CONFIG, CURRICULUM_CONFIG, REWARD_CONFIG, TRAINING_CONFIG, SAC_CONFIG)
    """
    config = {
        "env": ENV_CONFIG.copy(),
        "curriculum": CURRICULUM_CONFIG.copy(),
        "reward": REWARD_CONFIG.copy(),
        "training": TRAINING_CONFIG.copy(),
        "sac": SAC_CONFIG.copy(),
    }

    # Detect device if needed
    if device == "auto":
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except:
            device = "cpu"

    # Adjust for GPU
    if device == "cuda":
        config["training"]["total_timesteps"] = 150_000
        config["sac"]["learning_rate"] = 2e-4
        config["sac"]["batch_size"] = 256
        config["sac"]["net_arch"] = [256, 256]
        print("⚡ GPU detected - using optimized hyperparameters")
    else:
        print("💻 CPU mode - using conservative hyperparameters")

    config["sac"]["device"] = device

    return config
