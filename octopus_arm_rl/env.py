"""Octopus Arm environment for reinforcement learning."""

import numpy as np

import gymnasium as gym
from gymnasium import spaces
# except ImportError:
#     import gym
#     from gym import spaces


class OctopusArmEnv(gym.Env):
    """
    Octopus Arm environment where an agent learns to reach a target.
    
    The arm has multiple segments that can bend using muscle contractions.
    The goal is to reach the red dot (target) with the arm tip.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self, num_segments=3, render_mode=None):
        """
        Initialize the environment.
        
        Args:
            num_segments: Number of segments in the arm
            render_mode: Rendering mode ("human" or "rgb_array")
        """
        super().__init__()
        
        self.num_segments = num_segments
        self.render_mode = render_mode
        self.max_length = 1.0  # Total max length of the arm
        self.segment_length = self.max_length / num_segments
        
        # Action space: one angle per segment (how much to bend each segment)
        # Each action is in [-1, 1] which will be scaled to angle changes
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(num_segments,),
            dtype=np.float32
        )
        
        # Observation space: tip position + target position + distance to target
        # Tip position: (x, y)
        # Target position: (x, y)
        # Distance: scalar
        obs_size = 5
        self.observation_space = spaces.Box(
            low=-3.0, high=3.0,
            shape=(obs_size,),
            dtype=np.float32
        )
        
        self.seed()
        self.reset()
        
    def seed(self, seed=None):
        """Set random seed."""
        self.np_random = np.random.RandomState(seed)
        return [seed]
    
    def reset(self, *, seed=None, options=None):
        """Reset the environment and return initial observation."""

        # Handle seeding for Gymnasium compatibility
        if seed is not None:
            self.seed(seed)
        
        # Initialize arm segments in a vertical line
        self.segment_angles = np.zeros(self.num_segments)
        
        # Place target randomly in reachable area
        angle = self.np_random.uniform(0, 2 * np.pi)
        distance = self.np_random.uniform(self.max_length * 0.5, self.max_length * 0.9)
        self.target = np.array([
            distance * np.cos(angle),
            distance * np.sin(angle)
        ])
        
        self.step_count = 0
        # timer (30 steps at ~20 steps/sec, or ~50ms per step)
        self.max_steps = 30
        
        obs = self._get_observation()
        info = {} # gymnasium reset() returns (obs, info)
        
        return obs, info
    
    def _get_observation(self):
        """Get the current observation."""
        # Calculate segment positions
        segments = self._calculate_segments()
        
        # Only use the tip position (last segment), not all segments
        tip = segments[-1]
        
        # Calculate distance to target
        distance = np.linalg.norm(self.target - tip)
        
        # Combine observations: tip position + target position + distance
        obs = np.concatenate([tip, self.target, [distance]])
        
        # Clip to be within observation space bounds
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        return obs.astype(np.float32)
    
    def _calculate_segments(self):
        """Calculate positions of all arm segments."""
        segments = np.zeros((self.num_segments + 1, 2))
        segments[0] = [0, 0]  # origin
        
        current_angle = np.pi / 2  # Start pointing up
        
        for i in range(self.num_segments):
            # Update angle based on muscle contraction
            current_angle += self.segment_angles[i] * 0.5
            
            # Calculate new segment position
            dx = self.segment_length * np.cos(current_angle)
            dy = self.segment_length * np.sin(current_angle)
            
            segments[i + 1] = segments[i] + np.array([dx, dy])
        
        return segments
    
    def step(self, action):
        """
        Execute one step of the environment.
        
        Args:
            action: Array of angle changes for each segment [-1, 1]
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Apply angle changes directly to segments
        # action is in [-1, 1], we scale it to angle change
        max_angle_change = 0.5  # Max angle change per step (in radians)
        
        for i in range(self.num_segments):
            # Directly update angle based on action
            self.segment_angles[i] += action[i] * max_angle_change
            
            # Keep angles in [-π, π]
            self.segment_angles[i] = np.clip(self.segment_angles[i], -np.pi, np.pi)
        
        # Calculate new tip position
        segments = self._calculate_segments()
        tip = segments[-1]
        
        # Create reward based on distance to target
        distance = np.linalg.norm(self.target - tip)
        
        # Check if goal is reached
        target_reached = distance < 0.1
        
        # Reward structure:
        if target_reached:
            # Success! Big reward for reaching target
            reward = 10.0
        else:
            # Reward is inverse of distance (closer is better)
            reward = -distance
        
        # Small penalty for each step to encourage efficiency
        reward -= 0.01
        
        # Penalty for out of bounds
        if np.linalg.norm(tip) > self.max_length * 1.5:
            reward -= 1.0
        
        self.step_count += 1
        
        # Check if goal is reached (terminated)
        terminated = target_reached
        
        # Check if max steps reached (truncated/timeout)
        truncated = self.step_count >= self.max_steps
        
        # Big penalty if time runs out without reaching target
        if truncated and not terminated:
            reward -= 5.0  # Penalty for failing to reach target in time
        
        obs = self._get_observation()
        info = {
            "distance": distance, 
            "target_reached": target_reached,
            "timeout": truncated and not terminated,
            "steps_taken": self.step_count
        }
        
        # Return Gymnasium format: (obs, reward, terminated, truncated, info)
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "rgb_array":
            return self._render_rgb_array()
        elif self.render_mode == "human":
            import matplotlib.pyplot as plt
            img = self._render_rgb_array()
            plt.imshow(img)
            plt.axis("off")
            plt.pause(0.01)
            return img
        return None
    
    def _render_rgb_array(self):
        """Render as RGB array."""
        try:
            import cv2
        except ImportError:
            # Fallback if cv2 not available
            return np.zeros((500, 500, 3), dtype=np.uint8)
        
        img = np.ones((500, 500, 3), dtype=np.uint8) * 255
        
        # Calculate segments
        segments = self._calculate_segments()
        
        # Scale to image coordinates
        center = np.array([250, 250])
        scale = 150
        
        # Draw arm segments
        for i in range(len(segments) - 1):
            pt1 = (center + segments[i] * scale).astype(int)
            pt2 = (center + segments[i+1] * scale).astype(int)
            cv2.line(img, tuple(pt1), tuple(pt2), (0, 0, 255), 3)  # Red arm
        
        # Draw arm tip
        tip = (center + segments[-1] * scale).astype(int)
        cv2.circle(img, tuple(tip), 5, (255, 0, 0), -1)  # Blue tip
        
        # Draw target
        target_pos = (center + self.target * scale).astype(int)
        cv2.circle(img, tuple(target_pos), 8, (0, 255, 0), -1)  # Green target
        
        # Draw base
        cv2.circle(img, tuple(center.astype(int)), 5, (0, 0, 0), -1)  # Black base
        
        return img
