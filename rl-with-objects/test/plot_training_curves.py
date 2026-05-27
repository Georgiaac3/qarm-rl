#!/usr/bin/env python3
"""Plot training/evaluation curves from Stable-Baselines3 EvalCallback logs."""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def load_evaluation_log(log_path: Path):
    if not log_path.exists():
        raise FileNotFoundError(f"Evaluation log not found: {log_path}")
    data = np.load(log_path)
    if not {'timesteps', 'results', 'ep_lengths'}.issubset(data.keys()):
        raise ValueError(
            f"Unexpected evaluation log format: {log_path}. "
            f"Expected keys: timesteps, results, ep_lengths"
        )
    return data


def plot_curves(data, output_path: Path, show: bool = False):
    timesteps = data['timesteps']
    rewards = data['results']

    mean_rewards = rewards.mean(axis=1)
    std_rewards = rewards.std(axis=1)
    min_rewards = rewards.min(axis=1)
    max_rewards = rewards.max(axis=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)

    axes[0].plot(timesteps, mean_rewards, marker='o', label='Mean Reward')
    axes[0].fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.2,
                         label='±1 std')
    axes[0].plot(timesteps, min_rewards, linestyle='--', color='tab:red', label='Min Reward')
    axes[0].plot(timesteps, max_rewards, linestyle='--', color='tab:green', label='Max Reward')
    axes[0].set_title('Evaluation Rewards vs Timesteps')
    axes[0].set_xlabel('Timesteps')
    axes[0].set_ylabel('Reward')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='best')

    fig.suptitle('RL Evaluation Curves', fontsize=16)
    fig.savefig(output_path, dpi=200)
    print(f"Saved training curves to: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Plot training/evaluation curves from stable-baselines3 logs.'
    )
    parser.add_argument(
        '--log', type=str,
        default='../training/checkpoints/logs/evaluations.npz',
        help='Path to the evaluation log .npz file',
    )
    parser.add_argument(
        '--output', type=str,
        default='../training/checkpoints/logs/training_curves.png',
        help='Path to save the output plot image',
    )
    parser.add_argument(
        '--show', action='store_true',
        help='Display the plot after generation',
    )

    args = parser.parse_args()
    script_dir = Path(__file__).parent
    log_path = (script_dir / args.log).resolve()
    output_path = (script_dir / args.output).resolve()

    data = load_evaluation_log(log_path)
    plot_curves(data, output_path, show=args.show)


if __name__ == '__main__':
    main()
