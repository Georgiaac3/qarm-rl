"""
2D Arm Simulator with Physics and Simulation Clock
Similar to Gazebo: continuous time, dynamics integration, internal clock
"""

import numpy as np
from typing import Tuple
from kinematics_2d import Arm2D


class Arm2DSimulator:
    """
    Physics-based simulator for 2D arm with continuous time.
    
    Features:
    - Internal simulation clock (like Gazebo)
    - Dynamics integration (acceleration → velocity → position)
    - Motor dynamics simulation
    - Joint limits and friction
    """
    
    def __init__(
        self,
        l1: float = 0.5,
        l2: float = 0.5,
        dt_sim: float = 0.001,  # Simulation timestep (1 kHz)
        dt_control: float = 0.05,  # Control update rate (50 Hz)
        motor_inertia: float = 0.01,  # Joint inertia
        damping: float = 0.01,  # Viscous damping (REDUCED from 0.1)
    ):
        """
        Initialize 2D arm simulator.
        
        Args:
            l1, l2: Arm lengths
            dt_sim: Physics simulation timestep (seconds)
            dt_control: Control command update rate (seconds)
            motor_inertia: Rotor inertia for each joint
            damping: Viscous damping coefficient
        """
        self.arm = Arm2D(l1=l1, l2=l2)
        
        # Simulation timing
        self.dt_sim = dt_sim  # Physics timestep (1ms)
        self.dt_control = dt_control  # Control update (50ms)
        self.sim_time = 0.0  # Simulation clock
        self.last_control_time = 0.0
        
        # State: joint angles [q1, q2] and velocities [q1_dot, q2_dot]
        self.q = np.array([0.0, 0.0], dtype=np.float32)
        self.q_dot = np.array([0.0, 0.0], dtype=np.float32)
        
        # Physical parameters
        self.motor_inertia = motor_inertia
        self.damping = damping
        self.friction_coeff = 0.01  # Coulomb friction (REDUCED from 0.05)
        
        # Current control command (buffered until next control update)
        self.q_ddot_cmd = np.array([0.0, 0.0], dtype=np.float32)
        
        # Joint limits
        self.q_min = self.arm.q_min
        self.q_max = self.arm.q_max
        self.q_dot_max = np.pi  # Max joint velocity (rad/s)
    
    def set_control_command(self, q_ddot: np.ndarray):
        """
        Set acceleration command (holds until next update).
        This simulates motor commands.
        
        Args:
            q_ddot: Desired joint accelerations [2,]
        """
        self.q_ddot_cmd = np.clip(q_ddot, -10.0, 10.0).astype(np.float32)
    
    def _compute_dynamics(self) -> np.ndarray:
        """
        Compute joint accelerations from motor torque + friction.
        
        Simplified dynamics:
            I * q_ddot = τ_motor - b*q_dot - f_friction
        """
        # Motor torque (simplified: proportional to acceleration command)
        tau_motor = self.motor_inertia * self.q_ddot_cmd
        
        # Damping: b * q_dot
        tau_damping = self.damping * self.q_dot
        
        # Coulomb friction: f * sign(q_dot)
        tau_friction = self.friction_coeff * np.sign(self.q_dot)
        
        # Net torque -> acceleration
        q_ddot = (tau_motor - tau_damping - tau_friction) / self.motor_inertia
        
        return np.clip(q_ddot, -50.0, 50.0).astype(np.float32)
    
    def step(self):
        """
        Advance simulation by dt_sim (1 physics step).
        This is called internally many times per control cycle.
        """
        # Compute current accelerations
        q_ddot = self._compute_dynamics()
        
        # Integration: Euler method (simple but works for testing)
        self.q_dot = self.q_dot + q_ddot * self.dt_sim
        self.q = self.q + self.q_dot * self.dt_sim
        
        # Apply joint limits (hard stops)
        self.q = np.clip(self.q, self.q_min, self.q_max)
        
        # Velocity damping at limits
        for i in range(2):
            if self.q[i] <= self.q_min[i] or self.q[i] >= self.q_max[i]:
                self.q_dot[i] = 0.0  # Stop at limit
        
        # Velocity limits
        self.q_dot = np.clip(self.q_dot, -self.q_dot_max, self.q_dot_max)
        
        # Advance clock
        self.sim_time += self.dt_sim
    
    def get_simulation_step(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Get current simulator state.
        
        Returns:
            q: Joint angles [q1, q2]
            q_dot: Joint velocities [q1_dot, q2_dot]
            sim_time: Simulation time (seconds)
        """
        return self.q.copy(), self.q_dot.copy(), self.sim_time
    
    def get_end_effector(self) -> np.ndarray:
        """Get end-effector position in workspace."""
        return self.arm.forward_kinematics(self.q)
    
    def reset(self, q_init: np.ndarray = None):
        """Reset simulator state."""
        if q_init is None:
            self.q = np.random.uniform(self.q_min, self.q_max).astype(np.float32)
        else:
            self.q = np.clip(q_init, self.q_min, self.q_max).astype(np.float32)
        
        self.q_dot = np.zeros(2, dtype=np.float32)
        self.q_ddot_cmd = np.zeros(2, dtype=np.float32)
        self.sim_time = 0.0
        self.last_control_time = 0.0


class SimulationLoop:
    """
    Main simulation loop (like Gazebo physics server).
    
    Runs physics at high frequency (1 kHz) while accepting
    control commands at lower frequency (50 Hz).
    """
    
    def __init__(
        self,
        simulator: Arm2DSimulator,
        dt_control: float = 0.05,
        max_sim_duration: float = 10.0,
    ):
        """
        Initialize simulation loop.
        
        Args:
            simulator: Arm2DSimulator instance
            dt_control: Control update rate (50 Hz)
            max_sim_duration: Max simulation time (seconds)
        """
        self.sim = simulator
        self.dt_control = dt_control
        self.max_sim_duration = max_sim_duration
        
        self.is_running = False
        self.debug_info = {}
    
    def run_one_control_cycle(self, q_ddot_cmd: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Run one full control cycle:
        1. Set control command
        2. Integrate physics for dt_control seconds
        3. Return updated state
        
        Args:
            q_ddot_cmd: Desired accelerations for this cycle
        
        Returns:
            q: Current joint angles
            sim_time: Simulation time
        """
        # Set command
        self.sim.set_control_command(q_ddot_cmd)
        
        # Run physics for dt_control seconds
        num_physics_steps = int(self.dt_control / self.sim.dt_sim)
        for _ in range(num_physics_steps):
            self.sim.step()
            
            # Emergency stop if exceeded max time
            if self.sim.sim_time >= self.max_sim_duration:
                self.is_running = False
                break
        
        # Return state
        q, q_dot, sim_time = self.sim.get_simulation_step()
        
        self.debug_info = {
            "q": q.copy(),
            "q_dot": q_dot.copy(),
            "sim_time": sim_time,
            "physics_steps": num_physics_steps,
        }
        
        return q, sim_time
    
    def get_debug_info(self) -> dict:
        """Get debugging information about last step."""
        return self.debug_info
    
    def reset(self, q_init: np.ndarray = None):
        """Reset simulation."""
        self.sim.reset(q_init=q_init)

