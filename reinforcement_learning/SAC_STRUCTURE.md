# SAC Algorithm Structure

The SAC implementation has been split into 3 modules for better organization:

## 📦 Module Structure

### 1. `sac_components.py` - Core Components
Contains the SAC algorithm components:
- **ReplayBuffer**: Circular buffer for storing transitions
- **PolicyNetwork**: Stochastic policy network (Actor)
- **QNet**: Q-value network (Critic)

### 2. `sac_utils.py` - Utility Functions
Contains utility functions for training and testing:
- **calculate_target()**: Compute TD targets for Critic training
- **train_qarm_sac()**: Main training loop
- **test_qarm_sac()**: Testing loop
- **save_model()**: Save trained models
- **load_model()**: Load trained models

### 3. `main.py` - Main Entry Point
Main script for running training and testing:
- Environment setup
- ROS initialization (if simulation)
- Training execution
- Model saving
- Testing
- Cleanup

## 🚀 Usage

### Run the complete training pipeline:
```bash
python -m reinforcement_learning.main
```

### Import and use components:
```python
from reinforcement_learning.sac_components import PolicyNetwork, QNet, ReplayBuffer
from reinforcement_learning.sac_utils import train_qarm_sac, test_qarm_sac

# Create environment
env = QArmEnvironment(...)

# Train
policy, q1, q2 = train_qarm_sac(env, num_episodes=500)

# Test
test_qarm_sac(env, policy)
```

## 🔧 Configuration

Hyperparameters can be adjusted in `config.py`:
```python
lr_pi = 0.0005          # Policy learning rate
lr_q = 0.001            # Q-network learning rate
init_alpha = 0.01       # Initial temperature
gamma = 0.98            # Discount factor
batch_size = 32         # Mini-batch size
buffer_limit = 50000    # Replay buffer size
tau = 0.01             # Soft update coefficient
target_entropy = -4.0   # Target entropy
```
