# RL with Variable Objects - Bin Picking Task

Vision-based robotic bin picking with **TossingBot**-style grasping and throwing. The agent learns to pick objects of variable sizes and masses, then throw them out of the bin using predicted velocity corrections.

## Project Structure

```
rl-with-objects/
├── environment/
│   ├── objects.py           # Object definitions (spheres, cubes, cylinders)
│   └── bin_picking_env.py   # Gymnasium environment with multi-modal observations
├── models/
│   └── pixel_policy.py      # Pixel-wise FCN-based grasping policy
├── training/
│   └── train.py             # Training script (supervised learning baseline)
├── config.py                # Training configuration
└── README.md
```

## Features

### Multi-Modal Input
The environment provides:
- **RGB** (64×64×3): Color image of bin contents
- **Depth** (64×64×1): Normalized depth map from table surface
- **Heatmap** (64×64×1): Object density visualization (Gaussian splatted)

### Pixel-Wise Output
The policy predicts for each pixel:
- **Grasp probability** (1 channel): Binary logits for successful grasp
- **Velocity correction** (2 channels): [vx, vy] normalized throwing velocity
- **Confidence** (1 channel): Model's confidence in the prediction

### Variable Objects
Five object types with different physical properties:
- Small plastic · Medium plastic · Large plastic
- Small metal · Medium metal · Large metal  
- Heavy objects

Each object has:
- Realistic mass (10-200g)
- Variable size (5-25mm radius)
- Distinct shapes (sphere, cube, cylinder)
- Unique colors for identification

### Physics Simulation
- Gravity-driven object dynamics
- Air resistance/damping
- Projectile motion with velocity corrections
- Simple collision detection (table boundaries)

## Environment Details

### Gymnasium Interface
```python
env = BinPickingEnv(
    image_height=64,        # Input image resolution
    image_width=64,
    max_steps=200,          # Steps per episode
    n_objects=5,            # Objects in bin
    camera_fov=60.0,        # Camera field of view
)

obs, info = env.reset()
# obs = {'rgb': (64,64,3), 'depth': (64,64,1), 'heatmap': (64,64,1)}

action = np.array([grasp_x, grasp_y, vel_x, vel_y])  # Normalized [-1, 1]
obs, reward, terminated, truncated, info = env.step(action)
```

### Reward Structure
- **+10**: Successful grasp (object within reach)
- **+50**: Object leaves bin (z > 0.3m OR distance > 0.7m)
- **+100**: All objects cleared from bin
- **-1**: Time penalty per step
- **-5**: Collision with structure

## Policy Network

### Architecture
Fully Convolutional Network (FCN) with residual connections:
1. **Input layer**: Concatenate RGB + Depth + Heatmap (5 channels)
2. **Encoder**: 2× 2D convolution with stride 2 (feature extraction)
3. **Bottleneck**: Residual blocks at lowest resolution
4. **Decoder**: 2× transposed convolution with stride 2 (upsampling)
5. **Output heads**: 
   - Grasp head → (H, W, 1) logits
   - Velocity head → (H, W, 2) normalized to [-1, 1]
   - Confidence head → (H, W, 1) sigmoid

### Model Code
```python
from rl_with_objects.models.pixel_policy import PixelWiseGraspingPolicy

model = PixelWiseGraspingPolicy(
    in_channels=5,
    hidden_channels=32,
)

rgb = torch.randn(1, 3, 64, 64)
depth = torch.randn(1, 1, 64, 64)
heatmap = torch.randn(1, 1, 64, 64)

grasp_logits, velocity, confidence = model(rgb, depth, heatmap)
# grasp_logits:  (1, 1, 64, 64)
# velocity:      (1, 2, 64, 64)
# confidence:    (1, 1, 64, 64)

# Predict best action
pixel, velocity_correction, conf = model.predict_action(rgb[0], depth[0], heatmap[0])
```

## Training

### Supervised Learning Baseline
```bash
cd rl-with-objects
python training/train.py
```

This trains the model on generated trajectories using:
- Binary cross-entropy loss for grasp classification
- Mean squared error loss for velocity prediction
- Adam optimizer with learning rate 1e-3

### Integration with Stable Baselines3 (Future)
For full RL training, use PPO or SAC with the multi-modal observations:
```python
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

env = BinPickingEnv(...)
model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)
```

## Configuration

Edit `config.py` to adjust:
- **Environment**: Image resolution, number of objects, simulation params
- **Training**: Learning rate, batch size, training duration
- **Model**: Network depth, channel sizes, device (CPU/GPU)

GPU detection is automatic:
```bash
# Uses GPU if available
python training/train.py

# Force CPU
CUDA_VISIBLE_DEVICES="" python training/train.py
```

## References

### Papers
- **TossingBot** ([Zeng et al., 2020](https://tossingbot.cs.princeton.edu/)): Learning to Throw and Catch with 6-DoF Robotic Manipulation
- **Grasp Prediction Networks**: Pixel-wise predictions for manipulation
- **Vision-based Control**: End-to-end learning from images

### Implementation Notes
- Objects are represented as 3D spheres/cubes/cylinders
- Camera model is simplified (orthographic projection from above)
- Physics uses Euler integration with gravity and damping
- Collision detection is approximate (bounding boxes)

## Next Steps

1. **Collect Real Data**: Replace simulation with teleoperated demonstrations
2. **End-to-End RL**: Train with PPO/SAC on actual rewards
3. **Domain Randomization**: Vary lighting, object textures, camera parameters
4. **Real Robot Sim2Real**: Deploy trained policy on physical robot arm

## Development Setup

```bash
# Copy environment
cd ..
mkdir rl-with-objects
cd rl-with-objects

# Install dependencies
pip install gymnasium torch torchvision numpy opencv-python tqdm

# Test environment
python -c "from environment.bin_picking_env import BinPickingEnv; env = BinPickingEnv(); obs, _ = env.reset(); print('✓ Environment OK')"

# Test model
python -c "from models.pixel_policy import PixelWiseGraspingPolicy; model = PixelWiseGraspingPolicy(); print('✓ Model OK')"

# Run training
python training/train.py
```