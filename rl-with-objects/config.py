ENV_CONFIG = {
    "image_height": 64,
    "image_width": 64,
    "max_steps": 200,
    "n_objects": 5,
    "camera_fov": 60.0,
}

TRAINING_CONFIG = {
    "total_timesteps": 100_000,
    "eval_freq": 5_000,
    "n_eval_episodes": 5,
    "checkpoint_freq": 10_000,
    "learning_rate": 1e-3,
    "batch_size": 32,
    "warmup_steps": 1000,
}

MODEL_CONFIG = {
    "in_channels": 5,  # RGB(3) + Depth(1) + Heatmap(1)
    "hidden_channels": 32,
    "device": "auto",  # cuda if available, else cpu
}

def get_config(device="auto"):
    """Get configuration adjusted for device type."""
    config = {
        "env": ENV_CONFIG.copy(),
        "training": TRAINING_CONFIG.copy(),
        "model": MODEL_CONFIG.copy(),
    }
    
    # Detect device
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except:
            device = "cpu"
    
    # Adjust for hardware
    if device == "cuda":
        config["training"]["batch_size"] = 64
        config["training"]["learning_rate"] = 1e-3
        print("⚡ GPU detected - using optimized hyperparameters")
    else:
        print("💻 CPU mode - using conservative hyperparameters")
    
    config["model"]["device"] = device
    
    return config

