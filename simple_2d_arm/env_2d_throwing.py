"""
2D Throwing Environment for RL

Agent learns to throw a projectile to hit targets at different distances.
The agent controls:
  - Initial angle (θ) of the throw
  - Initial velocity (v) of the projectile
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class Arm2DThrowingEnv(gym.Env):
    """
    2D Throwing Environment using projectile physics.

    The agent learns to throw objects to hit targets.

    Observation: [target_distance_x, target_distance_y, throw_angle, projectile_vel, time_since_throw, projectile_x, projectile_y]
    Action: [launch_angle, launch_velocity] - radians and m/s
    Reward: -distance_to_target + 100 for hitting target - time penalty

    Physics Model:
    - Gravity: 9.81 m/s²
    - Air resistance: proportional to velocity squared
    - Projectile diameter: 0.025 m
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        max_steps: int = 200,
        target_distance_range: tuple = (1.0, 5.0),  # (min, max) meters
        gravity: float = 9.81,
        air_resistance: float = 0.1,
        projectile_radius: float = 0.025,  # meters
        dt: float = 0.01,  # Physics simulation timestep
    ):
        """
        Initialize 2D throwing environment.

        Args:
            max_steps: Maximum steps per episode (time to hit target)
            target_distance_range: (min, max) distance to target
            gravity: Gravity acceleration (m/s²)
            air_resistance: Drag coefficient
            projectile_radius: Radius of projectile (for collision)
            dt: Physics simulation timestep
        """
        super().__init__()

        self.max_steps = max_steps
        self.target_distance_range = target_distance_range
        self.gravity = gravity
        self.air_resistance = air_resistance
        self.projectile_radius = projectile_radius
        self.dt = dt

        # Observation space: [target_x, target_y, last_angle, last_vel, time_in_flight, proj_x, proj_y]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)

        # Action space: [launch_angle (rad), launch_velocity (m/s)]
        # Angle: -π/2 to π/2 (downward to upward)
        # Velocity: 0 to 20 m/s
        self.action_space = spaces.Box(
            low=np.array([-np.pi / 2, 0.0]), high=np.array([np.pi / 2, 20.0]), dtype=np.float32
        )

        # State
        self.target = np.array([0.0, 0.0], dtype=np.float32)  # Target position
        self.projectile = np.array([0.0, 0.0], dtype=np.float32)  # Projectile position
        self.projectile_vel = np.array([0.0, 0.0], dtype=np.float32)  # Velocity
        self.time_in_flight = 0.0
        self.step_count = 0
        self.last_action = np.array([0.0, 5.0], dtype=np.float32)
        self.has_thrown = False
        self.best_distance = float("inf")  # Track best distance during trajectory

    def _sample_target(self) -> np.ndarray:
        """Sample random target position."""
        distance = np.random.uniform(self.target_distance_range[0], self.target_distance_range[1])
        # Target is in front of the throwing position
        # Add some randomness in y (height)
        angle = np.random.uniform(-np.pi / 6, np.pi / 6)  # ±30°
        x = distance * np.cos(angle)
        y = distance * np.sin(angle)
        return np.array([x, y], dtype=np.float32)

    def _compute_drag_force(self, velocity: np.ndarray) -> np.ndarray:
        """
        Compute drag force using F_drag = -0.5 * ρ * v² * Cd * A
        Simplified: F_drag = -k * v * |v|
        """
        speed = np.linalg.norm(velocity)
        if speed < 1e-6:
            return np.zeros_like(velocity)
        drag = -self.air_resistance * velocity * speed
        return drag

    def _simulate_projectile_step(self):
        """Simulate one physics step of the projectile."""
        if not self.has_thrown:
            return

        # Forces
        gravity_force = np.array([0.0, -self.gravity], dtype=np.float32)
        drag_force = self._compute_drag_force(self.projectile_vel)

        # Acceleration (F = ma, m=1 simplified)
        acceleration = gravity_force + drag_force

        # Integration (Euler)
        self.projectile_vel = self.projectile_vel + acceleration * self.dt
        self.projectile = self.projectile + self.projectile_vel * self.dt

        self.time_in_flight += self.dt

        # Stop simulation if projectile hits ground or goes too far
        if self.projectile[1] < -0.5 or np.linalg.norm(self.projectile) > 10.0:
            self.has_thrown = False

    def _get_obs(self) -> np.ndarray:
        """Get observation vector."""
        return np.array(
            [
                self.target[0],
                self.target[1],
                self.last_action[0],  # Last throw angle
                self.last_action[1],  # Last throw velocity
                self.time_in_flight,
                self.projectile[0],
                self.projectile[1],
            ],
            dtype=np.float32,
        )

    def _compute_reward(self) -> float:
        """
        Progressive reward during simulation (not used for RL, only for tracking).
        For RL training, use _compute_final_reward() instead.
        """
        if not self.has_thrown:
            return 0.0

        distance = float(np.linalg.norm(self.projectile - self.target))

        # Track progress: reward for getting closer
        progress_reward = 0.0
        if self.best_distance != float("inf") and distance < self.best_distance:
            progress_reward = (self.best_distance - distance) * 0.1

        # Update best distance for next step
        if distance < self.best_distance:
            self.best_distance = distance

        # Distance penalty: quadratic for larger errors
        max_distance = 7.0
        normalized_distance = min(distance / max_distance, 1.0)
        distance_penalty = -(normalized_distance**2) * 2.0  # Small per-step penalty

        # Tiered bonus for getting close
        hit_bonus = 0.0
        if distance < 0.5:
            hit_bonus = 1.0

        reward = distance_penalty + progress_reward + hit_bonus
        return reward

    def _compute_final_reward(self, distance: float) -> float:
        """
        Compute final reward for the entire throw based on outcome.
        This is what RL agent sees - a single reward evaluating the throw.

        Args:
            distance: Final distance from projectile to target

        Returns:
            Single reward value for this throw
        """
        # Distance penalty: quadratic, normalized to max distance
        max_distance = 7.0
        normalized_distance = min(distance / max_distance, 1.0)
        distance_reward = -(normalized_distance**2) * 10.0  # Range: -10 to 0

        # Major bonus for hitting
        hit_bonus = 0.0
        if distance < self.projectile_radius * 2:
            # Direct hit!
            hit_bonus = 100.0
        elif distance < self.projectile_radius * 5:
            # Very close
            hit_bonus = 50.0
        elif distance < 0.2:
            # Close (10cm)
            hit_bonus = 20.0
        elif distance < 0.5:
            # Medium close (50cm)
            hit_bonus = 5.0

        # Time penalty (small)
        time_penalty = -0.01 * self.time_in_flight

        final_reward = distance_reward + hit_bonus + time_penalty

        return final_reward

    def _is_done(self) -> bool:
        """
        Episode ends when:
        - Projectile hits ground (y < 0)
        - Projectile goes out of bounds
        - Max steps reached
        - Hit target
        """
        if not self.has_thrown:
            return False

        distance = float(np.linalg.norm(self.projectile - self.target))
        hit = distance < self.projectile_radius * 2
        out_of_bounds = self.projectile[1] < -0.5 or np.linalg.norm(self.projectile) > 10.0
        max_reached = self.step_count >= self.max_steps

        return hit or out_of_bounds or max_reached

    def reset(self, seed=None, options=None):
        """Reset environment."""
        super().reset(seed=seed)

        # Sample random target
        self.target = self._sample_target()

        # Reset projectile state
        self.projectile = np.array([0.0, 0.0], dtype=np.float32)  # Start at origin
        self.projectile_vel = np.array([0.0, 0.0], dtype=np.float32)
        self.time_in_flight = 0.0
        self.step_count = 0
        self.last_action = np.array([0.0, 5.0], dtype=np.float32)
        self.has_thrown = False

        obs = self._get_obs()
        info = {"target": self.target.copy()}

        return obs, info

    def step(self, action: np.ndarray):
        """
        Execute one step.

        Action: [launch_angle, launch_velocity]
        - Throw the projectile with given angle and velocity
        - Simulates physics until projectile stops or hits ground
        - Returns SINGLE reward for the entire throw
        """
        # Take the action as the throw parameters
        launch_angle = float(action[0])
        launch_velocity = float(action[1])

        # Clip to valid range
        launch_velocity = np.clip(launch_velocity, 0.0, 20.0)

        # Convert angle to velocity components
        self.projectile_vel = np.array(
            [
                launch_velocity * np.cos(launch_angle),
                launch_velocity * np.sin(launch_angle),
            ],
            dtype=np.float32,
        )

        self.projectile = np.array([0.0, 0.0], dtype=np.float32)  # Reset position
        self.time_in_flight = 0.0
        self.has_thrown = True
        self.last_action = np.array([launch_angle, launch_velocity], dtype=np.float32)
        self.best_distance = float("inf")  # Reset best distance for this throw

        # Simulate physics until projectile stops or max steps
        # NOTE: We simulate multiple physics steps but return a SINGLE reward
        for _ in range(100):  # Max 100 physics simulation steps per action
            self._simulate_projectile_step()

            if self._is_done():
                break

        # Calculate reward ONCE based on final state
        # This is the key: we don't accumulate rewards, we evaluate the outcome
        distance = float(np.linalg.norm(self.projectile - self.target))
        episode_reward = self._compute_final_reward(distance)

        # Get observation
        obs = self._get_obs()
        terminated = self._is_done()
        truncated = False

        # Info
        distance = float(np.linalg.norm(self.projectile - self.target))
        info = {
            "projectile": self.projectile.copy(),
            "target": self.target.copy(),
            "distance": distance,
            "time_in_flight": self.time_in_flight,
        }

        self.step_count += 1

        return obs, episode_reward, terminated, truncated, info

    def render(self):
        """Rendering placeholder."""
        pass

    def close(self):
        """Close environment."""
        pass


if __name__ == "__main__":
    # Test the environment
    print("Testing Arm2DThrowingEnv...")

    env = Arm2DThrowingEnv()
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Target: {info['target']}")

    for i in range(3):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(
            f"Step {i}: reward={reward:.2f}, distance={info['distance']:.2f}, "
            f"projectile={info['projectile']}"
        )
        if terminated:
            break

    print("\n✓ Environment test passed!")
