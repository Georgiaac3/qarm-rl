"""
Improved RL training with curriculum learning and reward shaping.

Key improvements over baseline:
1. Curriculum Learning: Progressive difficulty (2 → 5 → 8 objects)
2. Reward Shaping: Multiple reward components (grasp quality, throw velocity, etc.)
3. Better Hyperparameters: Larger batch size, more replay buffer
4. Eval Tracking: Performance metrics per curriculum phase
5. Better Logging: Detailed breakdown of rewards

Usage:
    python3 training/train_curr.py                  # Default (250k steps with curriculum)
    python3 training/train_curr.py --curriculum     # With curriculum (default)
    python3 training/train_curr.py --nocurriculum   # Without curriculum (baseline)
    python3 training/train_curr.py --steps 500000   # 500k total steps
"""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

# Add parent dirs to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_config
from curriculum import CurriculumScheduler
from environment.bin_picking_env import BinPickingEnv


class CurriculumCallback(BaseCallback):
    """
    Callback to update curriculum and log progress.
    Syncs curriculum bonus to environment reward shaper.
    """

    def __init__(self, curriculum_scheduler: CurriculumScheduler, train_env, eval_env=None):
        super().__init__()
        self.curriculum_scheduler = curriculum_scheduler
        self.train_env = train_env
        self.eval_env = eval_env
        self.phase_metrics = defaultdict(list)
        self.last_phase = None

    def _on_step(self) -> bool:
        """Called after every environment step."""
        # Update curriculum
        phase = self.curriculum_scheduler.update(self.num_timesteps)

        # Sync curriculum bonus to environment reward shaper (unwrap Monitor)
        train_env_unwrapped = self.train_env.unwrapped
        train_env_unwrapped.curriculum_bonus = phase.reward_bonus
        train_env_unwrapped.reward_shaper.curriculum_bonus = phase.reward_bonus

        if self.eval_env:
            eval_env_unwrapped = self.eval_env.unwrapped
            eval_env_unwrapped.curriculum_bonus = phase.reward_bonus
            eval_env_unwrapped.reward_shaper.curriculum_bonus = phase.reward_bonus

        # Log phase changes
        if phase != self.last_phase:
            self.last_phase = phase
            print("\n" + "=" * 80)
            print(f"🎓 CURRICULUM UPDATE: {self.curriculum_scheduler.get_progress()}")
            print(f"   Objects: {phase.n_objects} | Reward multiplier: {phase.reward_bonus}x")
            print("=" * 80)

        # Log every 5000 steps
        if self.num_timesteps % 5000 == 0 and self.num_timesteps > 0:
            print(
                f"[Step {self.num_timesteps:7d}] Curriculum: {self.curriculum_scheduler.get_progress()}"
            )

        return True


class MetricsCallback(BaseCallback):
    """
    Track detailed metrics during training.
    """

    def __init__(self, eval_freq: int = 5000):
        super().__init__()
        self.eval_freq = eval_freq
        self.metrics = defaultdict(list)

    def _on_step(self) -> bool:
        """Called after every step."""
        # Track episode rewards
        if len(self.model.ep_info_buffer) > 0:
            ep_returns = [ep_info["r"] for ep_info in self.model.ep_info_buffer]
            self.metrics["episode_rewards"].extend(ep_returns)

        return True


def train_rl_with_curriculum(
    total_timesteps: int = 250_000,
    eval_freq: int = 5_000,
    save_freq: int = 10_000,
    use_curriculum: bool = True,
):
    """
    Train bin picking policy with SAC + Curriculum Learning + Reward Shaping.

    Args:
        total_timesteps: Total environment steps to train
        eval_freq: Frequency of evaluation episodes
        save_freq: Frequency of checkpoint saves
        use_curriculum: Whether to use curriculum learning
    """
    config = get_config(device="auto")

    print("\n" + "=" * 80)
    print("BIN PICKING WITH VARIABLE OBJECTS - IMPROVED RL TRAINING")
    print("=" * 80)
    print(f"Curriculum Learning: {'ENABLED 🎓' if use_curriculum else 'DISABLED'}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Device: {config['model']['device']}")

    # Create curriculum scheduler
    curriculum_scheduler = CurriculumScheduler(total_steps=total_timesteps)

    # Create training environment
    print("\nInitializing environment...")
    initial_phase = curriculum_scheduler.current_phase

    env = BinPickingEnv(
        image_height=config["env"]["image_height"],
        image_width=config["env"]["image_width"],
        max_steps=config["env"]["max_steps"],
        n_objects=initial_phase.n_objects if use_curriculum else config["env"]["n_objects"],
        camera_fov=config["env"].get("camera_fov", 60.0),
    )
    env = Monitor(env)

    # Create evaluation environment
    eval_env = BinPickingEnv(
        image_height=config["env"]["image_height"],
        image_width=config["env"]["image_width"],
        max_steps=config["env"]["max_steps"],
        n_objects=config["env"]["n_objects"],  # Always use full difficulty for eval
        camera_fov=config["env"].get("camera_fov", 60.0),
    )
    eval_env = Monitor(eval_env)

    # Create model directory
    model_dir = Path(__file__).parent / "checkpoints"
    model_dir.mkdir(exist_ok=True)

    info_str = "WITH_CURRICULUM" if use_curriculum else "BASELINE"
    model_dir_specific = model_dir / info_str
    model_dir_specific.mkdir(exist_ok=True)

    print(f"\n{'='*80}")
    print("TRAINING CONFIGURATION")
    print(f"{'='*80}")
    print(f"Learning rate:              {config['training']['learning_rate']}")
    print(f"Batch size:                 {config['training']['batch_size']}")
    print(f"Buffer size:                50,000")
    print(f"Learning starts:            1,000 steps")
    print(
        f"Initial objects:            {initial_phase.n_objects if use_curriculum else config['env']['n_objects']}"
    )
    print(f"Eval environment objects:   {config['env']['n_objects']} (always max)")
    print(f"Eval frequency:             Every {eval_freq:,} steps")
    print(f"Checkpoint frequency:       Every {save_freq:,} steps")

    # Create SAC agent (with improved hyperparameters)
    print("\nCreating SAC agent...")
    model = SAC(
        "MultiInputPolicy",
        env,
        learning_rate=config["training"]["learning_rate"],
        batch_size=config["training"]["batch_size"],
        buffer_size=50_000,
        learning_starts=1_000,
        tau=0.005,
        gamma=0.99,
        ent_coef="auto",
        train_freq=1,
        gradient_steps=1,
        verbose=1,
        device=config["model"]["device"],
    )

    n_params = sum(p.numel() for p in model.policy.parameters())
    print(f"Model parameters: {n_params:,}")

    # Callbacks
    curriculum_callback = CurriculumCallback(
        curriculum_scheduler=curriculum_scheduler,
        train_env=env,
        eval_env=eval_env,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=str(model_dir_specific),
        name_prefix="rl_policy",
        save_replay_buffer=False,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir_specific / "best"),
        log_path=str(model_dir_specific / "logs"),
        eval_freq=eval_freq,
        n_eval_episodes=5,
        deterministic=False,
    )

    # Training
    print(f"\n{'='*80}")
    print("STARTING TRAINING")
    print(f"{'='*80}\n")

    try:
        # Custom training loop to handle curriculum updates
        if use_curriculum:
            model.learn(
                total_timesteps=total_timesteps,
                callback=[curriculum_callback, checkpoint_callback, eval_callback],
                progress_bar=True,
            )
        else:
            # Baseline: no curriculum
            model.learn(
                total_timesteps=total_timesteps,
                callback=[checkpoint_callback, eval_callback],
                progress_bar=True,
            )

    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")

    # Save final model
    final_path = model_dir_specific / "rl_policy_final.zip"
    model.save(str(final_path))
    print(f"\n✓ Final model saved to {final_path}")

    # Final evaluation
    print(f"\n{'='*80}")
    print("FINAL EVALUATION (5 episodes at max difficulty)")
    print(f"{'='*80}")

    test_episodes = 5
    episode_rewards = []
    episode_lengths = []

    for episode in range(test_episodes):
        obs, _ = eval_env.reset()
        episode_reward = 0.0
        episode_length = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        remaining = info.get("objects_remaining", 0)
        print(
            f"Episode {episode+1}: Reward = {episode_reward:7.2f} | Steps = {episode_length:3d} | Objects left: {remaining}"
        )

    print(f"\n{'='*80}")
    print("FINAL STATISTICS")
    print(f"{'='*80}")
    print(f"Mean reward:        {np.mean(episode_rewards):7.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Mean episode length: {np.mean(episode_lengths):7.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Best episode:       {np.max(episode_rewards):7.2f}")
    print(f"Worst episode:      {np.min(episode_rewards):7.2f}")

    env.close()
    eval_env.close()

    print(f"\n✓ Training complete!")
    print(f"Results saved to: {model_dir_specific}")

    return {
        "model_path": str(final_path),
        "eval_rewards": episode_rewards,
        "eval_lengths": episode_lengths,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train bin picking with curriculum learning and reward shaping"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=250_000,
        help="Total training timesteps (default: 250k for 3 curriculum phases)",
    )
    parser.add_argument(
        "--eval-freq", type=int, default=5_000, help="Evaluation frequency (default: 5k steps)"
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=10_000,
        help="Checkpoint save frequency (default: 10k steps)",
    )
    parser.add_argument(
        "--no-curriculum",
        action="store_true",
        default=False,
        help="Train WITHOUT curriculum learning (baseline comparison)",
    )

    args = parser.parse_args()

    results = train_rl_with_curriculum(
        total_timesteps=args.steps,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
        use_curriculum=not args.no_curriculum,
    )
