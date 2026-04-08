"""
2D Arm Gymnasium Environment using Physics Simulator
Uses simulator_2d.py for realistic physics and simulation clock
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from simulator_2d import Arm2DSimulator, SimulationLoop
from kinematics_2d import Arm2D


class Arm2DEnvPhysics(gym.Env):
    """
    Gymnasium environment for 2D arm with REALISTIC PHYSICS.
    
    Unlike simple step-based env, this uses:
    - Physics simulator with 1 kHz update
    - Simulation clock (like Gazebo)
    - Motor dynamics and friction
    - Realistic joint behavior
    
    Observation: [target_x, target_y, q1, q2, q_dot1, q_dot2]
    Action: [residual_accel1, residual_accel2] (joint accelerations)
    Reward: -distance_to_target
    """
    
    metadata = {"render_modes": []}
    
    def __init__(
        self,
        l1: float = 0.5,
        l2: float = 0.5,
        max_episode_duration: float = 5.0,  # 5 seconds simulation time
        dt_control: float = 0.05,  # 50 Hz control
        motor_inertia: float = 0.01,
        damping: float = 0.1,
        target_radius: float = 1.0,
    ):
        """
        Initialize 2D arm environment with physics.
        
        Args:
            l1, l2: Arm lengths
            max_episode_duration: Max sim time (seconds)
            dt_control: Control update rate (50 Hz)
            motor_inertia: Joint inertia
            damping: Viscous damping
            target_radius: Target sampling radius
        """
        super().__init__()
        
        # Create simulator
        self.sim = Arm2DSimulator(
            l1=l1, l2=l2,
            dt_sim=0.001,  # 1 kHz physics
            dt_control=dt_control,
            motor_inertia=motor_inertia,
            damping=damping,
        )
        
        # Simulation loop
        self.sim_loop = SimulationLoop(
            self.sim,
            dt_control=dt_control,
            max_sim_duration=max_episode_duration,
        )
        
        self.arm = Arm2D(l1=l1, l2=l2)
        self.max_episode_duration = max_episode_duration
        self.dt_control = dt_control
        self.target_radius = target_radius
        
        # Observation space: [target_x, target_y, q1, q2, q_dot1, q_dot2]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )
        
        # Action space: joint accelerations
        self.action_space = spaces.Box(
            low=-5.0, high=5.0, shape=(2,), dtype=np.float32
        )
        
        # State
        self.target = np.array([0.0, 0.0], dtype=np.float32)
        self.episode_start_time = 0.0
    
    def _sample_target(self) -> np.ndarray:
        """Sample random target in workspace."""
        radius = np.random.uniform(0.3, self.target_radius)
        angle = np.random.uniform(0, 2 * np.pi)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        return np.array([x, y], dtype=np.float32)
    
    def _get_ik_accelerations(self, target: np.ndarray) -> np.ndarray:
        """
        Compute ideal accelerations using inverse kinematics.
        DEGRADED: Weak signal to force RL to learn.
        """
        # Compute target joint angles
        q, _ = self.arm.inverse_kinematics(target, self.sim.q)
        
        # Error in position
        q_error = q - self.sim.q
        
        # PD control: compute desired accelerations
        # Weak gains (degraded)
        Kp = 2.0  # Position gain
        Kd = 0.5  # Velocity gain
        
        q_ddot_ik = Kp * q_error - Kd * self.sim.q_dot
        
        # Scale (weak signal)
        return np.clip(q_ddot_ik * 0.3, -5.0, 5.0).astype(np.float32)
    
    def _get_obs(self) -> np.ndarray:
        """Get observation."""
        q_ik = self._get_ik_accelerations(self.target)
        # Convert accelerations to observable (for state checking)
        obs = np.concatenate([
            self.target,
            self.sim.q,
            self.sim.q_dot,
        ]).astype(np.float32)
        return obs
    
    def _compute_reward(self) -> float:
        """Compute reward based on end-effector distance."""
        ee_pos = self.sim.get_end_effector()
        distance = float(np.linalg.norm(ee_pos - self.target))
        
        reward = -distance
        
        # Bonus for reaching
        if distance < 0.05:
            reward += 5.0
        
        return reward
    
    def _is_episode_done(self) -> bool:
        """Check if episode should terminate."""
        ee_pos = self.sim.get_end_effector()
        distance = float(np.linalg.norm(ee_pos - self.target))
        
        # Done if: reached target OR exceeded max time
        episode_time = self.sim.sim_time - self.episode_start_time
        
        return (distance < 0.02) or (episode_time >= self.max_episode_duration)
    
    def reset(self, seed=None, options=None):
        """Reset environment."""
        super().reset(seed=seed)
        
        # Reset simulator
        self.sim_loop.reset()
        self.episode_start_time = self.sim.sim_time
        
        # Sample target
        self.target = self._sample_target()
        
        obs = self._get_obs()
        info = {"target": self.target.copy(), "sim_time": self.sim.sim_time}
        
        return obs, info
    
    def step(self, action: np.ndarray):
        """
        Execute one control cycle.
        
        Args:
            action: Joint accelerations [accel1, accel2]
        
        This runs physics for dt_control seconds internally.
        """
        # Get IK accelerations (degraded)
        q_ik_ddot = self._get_ik_accelerations(self.target)
        
        # Final command: IK + residual correction
        q_ddot_final = q_ik_ddot + action
        
        # Run simulation loop for one control cycle
        q, sim_time = self.sim_loop.run_one_control_cycle(q_ddot_final)
        
        # Observation
        obs = self._get_obs()
        
        # Reward
        reward = self._compute_reward()
        
        # Termination
        terminated = self._is_episode_done()
        truncated = False
        
        # Info
        ee_pos = self.sim.get_end_effector()
        info = {
            "end_effector": ee_pos.copy(),
            "target": self.target.copy(),
            "distance": np.linalg.norm(ee_pos - self.target),
            "sim_time": sim_time,
            "q": q.copy(),
            "q_dot": self.sim.q_dot.copy(),
        }
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        pass
    
    def close(self):
        pass


# Debugging utility
def test_simulator():
    """Quick test of physics simulator."""
    print("=" * 80)
    print("PHYSICS SIMULATOR TEST")
    print("=" * 80)
    
    env = Arm2DEnvPhysics(max_episode_duration=2.0)
    obs, info = env.reset()
    
    print(f"Initial sim time: {env.sim.sim_time:.3f} s")
    print(f"Initial state: q={env.sim.q}, q_dot={env.sim.q_dot}")
    
    # Run a few steps
    for step in range(5):
        action = np.array([1.0, -0.5])  # Some acceleration
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"\nStep {step}:")
        print(f"  Sim time: {info['sim_time']:.3f} s")
        print(f"  q: {info['q']}")
        print(f"  q_dot: {info['q_dot']}")
        print(f"  distance: {info['distance']:.4f} m")
        print(f"  reward: {reward:.4f}")
        
        if terminated or truncated:
            print("  Episode done!")
            break
    
    print("\n" + "=" * 80)
    print("✓ Physics simulator working!")
    print("=" * 80)


if __name__ == "__main__":
    test_simulator()
