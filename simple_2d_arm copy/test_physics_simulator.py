"""
Test script for physics-based simulator
Compares: Simple env vs Physics env
"""

import numpy as np
import matplotlib.pyplot as plt
from env_2d import Arm2DEnv
from env_2d_physics import Arm2DEnvPhysics


def test_simple_vs_physics():
    """Compare simple env vs physics-based env."""
    print("=" * 80)
    print("COMPARING SIMPLE vs PHYSICS SIMULATOR")
    print("=" * 80)
    
    # Create both environments
    env_simple = Arm2DEnv(max_steps=100, dt=0.05)
    env_physics = Arm2DEnvPhysics(max_episode_duration=5.0)
    
    # Reset both
    obs_simple, _ = env_simple.reset()
    obs_physics, _ = env_physics.reset()
    
    print("\n[SIMPLE ENV]")
    print(f"  Initial q: {env_simple.q}")
    print(f"  Initial ik_q_dot: {env_simple.ik_q_dot}")
    print(f"  Each step represents: {env_simple.dt} seconds")
    
    print("\n[PHYSICS ENV]")
    print(f"  Initial q: {env_physics.sim.q}")
    print(f"  Initial q_dot: {env_physics.sim.q_dot}")
    print(f"  Simulation clock: {env_physics.sim.sim_time:.3f} s")
    print(f"  Physics runs at: 1 kHz (dt_sim={env_physics.sim.dt_sim}s)")
    print(f"  Control updates at: {1/env_physics.dt_control:.0f} Hz")
    
    # Run both for same number of steps
    print("\n" + "=" * 80)
    print("RUNNING 5 STEPS WITH SAME ACTION")
    print("=" * 80)
    
    action = np.array([0.5, -0.3])
    
    for step in range(5):
        # Simple env
        obs_s, reward_s, term_s, trunc_s, info_s = env_simple.step(action)
        
        # Physics env
        obs_p, reward_p, term_p, trunc_p, info_p = env_physics.step(action)
        
        print(f"\nSTEP {step}:")
        print(f"  SIMPLE:")
        print(f"    q: {info_s['end_effector']}")
        print(f"    distance: {info_s['distance']:.4f} m")
        
        print(f"  PHYSICS:")
        print(f"    q: {info_p['q']}")
        print(f"    q_dot: {info_p['q_dot']}")
        print(f"    Sim time: {info_p['sim_time']:.3f} s")
        print(f"    distance: {info_p['distance']:.4f} m")
        
        if term_s or trunc_s or term_p or trunc_p:
            break
    
    print("\n" + "=" * 80)
    print("KEY DIFFERENCES")
    print("=" * 80)
    print("""
SIMPLE ENV (env_2d.py):
  ✓ Fast (no physics integr ation)
  ✗ No physics dynamics
  ✗ No simulation clock
  ✗ Abstract state updates

PHYSICS ENV (env_2d_physics.py):
  ✓ Realistic physics (motor dynamics, friction)
  ✓ Simulation clock (like Gazebo)
  ✓ Physics runs at 1 kHz internally
  ✓ Control updates at 50 Hz
  ✗ Slightly slower (but still very fast)
  ✗ More parameters to tune
    """)


def test_physics_env_detailed():
    """Detailed test of physics environment."""
    print("\n" + "=" * 80)
    print("DETAILED PHYSICS ENVIRONMENT TEST")
    print("=" * 80)
    
    env = Arm2DEnvPhysics(max_episode_duration=1.0)
    obs, info = env.reset()
    
    print(f"\nInitial state:")
    print(f"  Target: {env.target}")
    print(f"  q: {env.sim.q}")
    print(f"  q_dot: {env.sim.q_dot}")
    print(f"  Sim time: {env.sim.sim_time:.3f} s")
    
    # Run with varying accelerations
    print(f"\nRunning episode with varying accelerations...")
    
    history = {
        "sim_time": [],
        "q0": [],
        "q1": [],
        "q_dot0": [],
        "q_dot1": [],
        "distance": [],
    }
    
    for step in range(20):
        # Vary action over time
        phase = step / 20.0
        action = np.array([2.0 * np.sin(phase * 2 * np.pi),
                          1.0 * np.cos(phase * 2 * np.pi)])
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        history["sim_time"].append(info["sim_time"])
        history["q0"].append(info["q"][0])
        history["q1"].append(info["q"][1])
        history["q_dot0"].append(info["q_dot"][0])
        history["q_dot1"].append(info["q_dot"][1])
        history["distance"].append(info["distance"])
        
        if terminated or truncated:
            print(f"  Episode ended at step {step}, sim_time={info['sim_time']:.3f}s")
            break
    
    # Plot
    print(f"\nGenerated trajectory with {len(history['sim_time'])} points")
    print(f"  Sim time range: {min(history['sim_time']):.3f} - {max(history['sim_time']):.3f} s")
    print(f"  Joint 1 range: {min(history['q0']):.3f} - {max(history['q0']):.3f} rad")
    print(f"  Joint 2 range: {min(history['q1']):.3f} - {max(history['q1']):.3f} rad")
    print(f"  Distance to target: {history['distance'][-1]:.4f} m")
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    ax = axes[0, 0]
    ax.plot(history["sim_time"], history["q0"], label="q1", marker='o')
    ax.plot(history["sim_time"], history["q1"], label="q2", marker='s')
    ax.set_xlabel("Simulation Time (s)")
    ax.set_ylabel("Joint Angle (rad)")
    ax.set_title("Joint Angles over Simulation Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.plot(history["sim_time"], history["q_dot0"], label="q1_dot", marker='o')
    ax.plot(history["sim_time"], history["q_dot1"], label="q2_dot", marker='s')
    ax.set_xlabel("Simulation Time (s)")
    ax.set_ylabel("Joint Velocity (rad/s)")
    ax.set_title("Joint Velocities")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    ax.plot(history["sim_time"], history["distance"], marker='o')
    ax.set_xlabel("Simulation Time (s)")
    ax.set_ylabel("Distance to Target (m)")
    ax.set_title("End-effector Distance to Target")
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    ax.axis('off')
    info_text = f"""
PHYSICS SIMULATOR INFO
{'='*40}

Joint 1 Range: {min(history['q0']):.3f} - {max(history['q0']):.3f} rad
Joint 2 Range: {min(history['q1']):.3f} - {max(history['q1']):.3f} rad

Velocity 1 Range: {min(history['q_dot0']):.3f} - {max(history['q_dot0']):.3f} rad/s
Velocity 2 Range: {min(history['q_dot1']):.3f} - {max(history['q_dot1']):.3f} rad/s

Final Distance: {history['distance'][-1]:.4f} m
Sim Time: {history['sim_time'][-1]:.3f} s
Steps Executed: {len(history['sim_time'])}

Physics Timestep: 1 ms (1 kHz)
Control Frequency: 50 Hz (20 ms/control)
Total Physics Steps: ~{int(history['sim_time'][-1] / 0.001)}
    """
    ax.text(0.1, 0.5, info_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig("physics_simulator_test.png", dpi=150)
    print(f"\n✓ Plot saved to physics_simulator_test.png")
    plt.show()


if __name__ == "__main__":
    # Test 1: Simple vs Physics comparison
    test_simple_vs_physics()
    
    # Test 2: Detailed physics test
    test_physics_env_detailed()
    
    print("\n" + "=" * 80)
    print("✓ TESTS COMPLETE")
    print("=" * 80)
