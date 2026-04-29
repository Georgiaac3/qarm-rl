"""
2D Arm Gymnasium Environment for RL Testing
Validates: Analytical IK + Residual RL correction
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from kinematics_2d import Arm2D

#faire une simu comme gazebo (pas .step) faire en sorte que la simu ait son propre temps(horloge)  

class Arm2DEnv(gym.Env):
    """
    Gymnasium environment for 2D arm reaching task.
    
    The arm starts with a target position. The RL agent learns to correct
    the inverse kinematics solution by applying residual joint velocities.
    
    Observation: [target_x, target_y, q1, q2, ik_q1_dot, ik_q2_dot]
    Action: [residual_q1_dot, residual_q2_dot] (corrections to IK velocities)
    Reward: -distance_to_target (continuous)
    """
    
    metadata = {"render_modes": []}
    
    def __init__(
        self,
        l1: float = 0.5,
        l2: float = 0.5,
        max_steps: int = 100,
        target_radius: float = 1.0,
        dt: float = 0.05,
        ik_scale: float = 1.0,
    ):
        """
        Initialize 2D arm environment.
        
        Args:
            l1, l2: Arm segment lengths
            max_steps: Maximum steps per episode
            target_radius: Radius of target sampling region
            dt: Time step for simulation
            ik_scale: Scale factor for IK velocities
        """
        super().__init__()
        
        self.arm = Arm2D(l1=l1, l2=l2)
        self.max_steps = max_steps
        self.target_radius = target_radius
        self.dt = dt
        self.ik_scale = ik_scale
        
        # Observation space: [target_x, target_y, q1, q2, ik_q1_dot, ik_q2_dot]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )
        
        # Action space: corrections to joint velocities (in range [-1, 1])
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        
        # State variables
        self.q = np.array([0.0, 0.0], dtype=np.float32)  # Current joint angles
        self.target = np.array([0.0, 0.0], dtype=np.float32)
        self.ik_q_dot = np.array([0.0, 0.0], dtype=np.float32)  # IK-computed velocities
        self.step_count = 0
    
    def _sample_target(self) -> np.ndarray:
        """Sample random target in workspace."""
        # Sample in circle within workspace
        radius = np.random.uniform(0.3, self.target_radius)
        angle = np.random.uniform(0, 2 * np.pi)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        return np.array([x, y], dtype=np.float32)
    
    def _get_ik_velocities(self, target: np.ndarray) -> np.ndarray:
        """
        Compute ideal velocities using inverse kinematics.
        
        This simulates the "analytical prior" in your hybrid approach.
        DEGRADED VERSION: Weak IK signal to force RL to learn corrections.
        """
        # Compute target joint angles
        q_target, success = self.arm.inverse_kinematics(target, self.q)
        
        if not success:
            return np.array([0.0, 0.0], dtype=np.float32)
        
        # Simple proportional control: move towards target
        q_error = q_target - self.q
        
        # Compute velocities (P-controller)
        # DEGRADED: Reduced from 2.0 to 0.5 to make IK less effective
        Kp = 0.5  # Weak controller → RL must learn corrections
        q_dot = Kp * q_error
        
        # DEGRADED: ik_scale reduced from 1.0 to 0.2
        return np.clip(q_dot * 0.2, -np.pi, np.pi).astype(np.float32)
    
    def _get_obs(self) -> np.ndarray:
        """Get observation vector."""
        return np.concatenate([
            self.target,
            self.q,
            self.ik_q_dot
        ]).astype(np.float32)
    
    def _compute_reward(self) -> float:
        """Compute reward based on distance to target."""
        ee_pos = self.arm.forward_kinematics(self.q)
        distance = float(np.linalg.norm(ee_pos - self.target))
        
        reward = -distance
        
        # Bonus for reaching target
        if distance < 0.05:  # Within 5cm
            reward += 10.0
        
        return reward
    
    def _is_done(self) -> bool:
        """Check if episode is done."""
        ee_pos = self.arm.forward_kinematics(self.q)
        distance = float(np.linalg.norm(ee_pos - self.target))
        
        # Done if: reached target OR max steps
        return distance < 0.02 or self.step_count >= self.max_steps
    
    def reset(self, seed=None, options=None):
        """Reset environment."""
        super().reset(seed=seed)
        
        # Random initial configuration
        self.q = np.random.uniform(self.arm.q_min, self.arm.q_max).astype(np.float32)
        
        # Sample random target
        self.target = self._sample_target()
        
        # Compute initial IK velocities
        self.ik_q_dot = self._get_ik_velocities(self.target)
        
        self.step_count = 0
        
        obs = self._get_obs()
        info = {"target": self.target.copy()}
        
        return obs, info
    
    def step(self, action: np.ndarray):
        """
        Execute one step.
        
        action: residual corrections to joint velocities
        This implements: q_final_dot = q_ik_dot + action * scale
        """
        # Residual action is scaled correction
        residual_scale = 0.5  # Limit correction magnitude
        q_dot_correction = action * residual_scale
        
        # Final joint velocities: IK + residual correction
        q_dot_final = self.ik_q_dot + q_dot_correction
        q_dot_final = np.clip(q_dot_final, -np.pi, np.pi)
        
        # Integrate: simple Euler step
        self.q = self.q + q_dot_final * self.dt
        self.q = np.clip(self.q, self.arm.q_min, self.arm.q_max)
        
        # Update IK velocities for next observation
        self.ik_q_dot = self._get_ik_velocities(self.target)
        
        # Compute reward
        reward = self._compute_reward()
        terminated = self._is_done()
        truncated = False
        
        # Observation
        obs = self._get_obs()
        
        # Info
        ee_pos = self.arm.forward_kinematics(self.q)
        info = {
            "end_effector": ee_pos.copy(),
            "target": self.target.copy(),
            "distance": np.linalg.norm(ee_pos - self.target),
        }
        
        self.step_count += 1
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        pass
    
    def close(self):
        pass
