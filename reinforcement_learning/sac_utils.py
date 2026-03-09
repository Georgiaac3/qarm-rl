from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F

from core.config import settings
from reinforcement_learning.environment import QArmEnvironment
from reinforcement_learning.sac_components import PolicyNetwork, QNet, ReplayBuffer
from utils.logger import logger


def calculate_target(policy: PolicyNetwork, q1: QNet, q2: QNet, mini_batch: Tuple) -> torch.Tensor:
    """
    Calculate target for Critic training.

    target = r + gamma * (min(Q1, Q2) + alpha * H)

    Args:
        policy: Policy network
        q1: First Q-network
        q2: Second Q-network
        mini_batch: Batch of transitions

    Returns:
        Target Q-values
    """
    s, a, r, s_prime, done = mini_batch

    with torch.no_grad():
        a_prime, log_prob = policy(s_prime)
        entropy = -policy.log_alpha.exp() * log_prob
        q1_val, q2_val = q1(s_prime, a_prime), q2(s_prime, a_prime)
        q1_q2 = torch.cat([q1_val, q2_val], dim=1)
        min_q = torch.min(q1_q2, 1, keepdim=True)[0]
        target = r + settings.gamma * done * (min_q + entropy)

    return target


def train_qarm_sac(
    env: QArmEnvironment, num_episodes: int = 1000, max_steps: int = 200, print_interval: int = 10
) -> Tuple[PolicyNetwork, QNet, QNet]:
    """
    Train SAC agent on QArm environment.

    Args:
        env: QArmEnvironment instance
        num_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        print_interval: Logging frequency (in episodes)

    Returns:
        Tuple of (policy, q1, q2)
    """
    logger.info("Starting SAC training")
    logger.info(f"Episodes: {num_episodes}, Max steps: {max_steps}")

    memory = ReplayBuffer()

    # Create networks (2 Q-networks + 2 targets + 1 policy)
    q1, q2 = QNet(settings.lr_q), QNet(settings.lr_q)
    q1_target, q2_target = QNet(settings.lr_q), QNet(settings.lr_q)
    policy = PolicyNetwork(settings.lr_pi)

    # Initialize targets
    q1_target.load_state_dict(q1.state_dict())
    q2_target.load_state_dict(q2.state_dict())

    score = 0.0

    for n_epi in range(num_episodes):
        s = env.reset()
        done = False
        count = 0

        while count < max_steps and not done:
            # Select action
            action, log_prob = policy(torch.from_numpy(s).float())
            action_np = action.detach().numpy()

            # Execute action in environment
            s_prime, reward, done, info = env.step(action_np)

            # Store transition
            memory.put((s, action_np, reward, s_prime, done))

            score += reward
            s = s_prime
            count += 1

        # Learning (after collecting enough experiences)
        if memory.size() > 1000:
            for i in range(20):
                mini_batch = memory.sample(settings.batch_size)
                td_target = calculate_target(policy, q1_target, q2_target, mini_batch)
                q1.train_net(td_target, mini_batch)
                q2.train_net(td_target, mini_batch)
                policy.train_net(q1, q2, mini_batch)
                q1.soft_update(q1_target)
                q2.soft_update(q2_target)

        # Log results
        if n_epi % print_interval == 0 and n_epi != 0:
            avg_score = score / print_interval
            alpha_val = policy.log_alpha.exp().item()
            logger.info(f"Episode: {n_epi}, Avg Score: {avg_score:.2f}, Alpha: {alpha_val:.4f}")
            score = 0.0

    logger.info("Training completed")
    return policy, q1, q2


def test_qarm_sac(env, policy: PolicyNetwork, num_episodes: int = 5):
    """
    Test a trained SAC agent.

    Args:
        env: QArmEnvironment instance
        policy: Trained policy network
        num_episodes: Number of test episodes
    """
    logger.info("AGENT TESTING")

    for n_epi in range(num_episodes):
        s = env.reset()
        done = False
        total_reward = 0.0
        count = 0

        while count < 200 and not done:
            with torch.no_grad():
                # Deterministic mode: use mean
                x = torch.from_numpy(s).float().unsqueeze(0)
                x = F.relu(policy.fc1(x))
                a = torch.tanh(policy.fc_mu(x))
                a_np = a.squeeze(0).numpy()

            s, r, done, info = env.step(a_np)
            total_reward += r
            count += 1

            # Log state every 50 steps
            if count % 50 == 0:
                logger.debug(f"  Step {count}, Distance: {-r:.4f}m")

        logger.info(f"Episode {n_epi + 1}: Score = {total_reward:.2f}")


def save_model(policy: PolicyNetwork, q1: QNet, q2: QNet, filepath: str = "models/sac_qarm.pth"):
    """
    Save trained models
    """
    # Create directory if it doesn't exist
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "policy": policy.state_dict(),
            "q1": q1.state_dict(),
            "q2": q2.state_dict(),
            "log_alpha": policy.log_alpha,
        },
        filepath,
    )
    logger.info(f"Model saved: {filepath}")


def load_model(filepath: str = "models/sac_qarm.pth") -> Tuple[PolicyNetwork, QNet, QNet]:
    """
    Load trained models.
    """
    checkpoint = torch.load(filepath)

    policy = PolicyNetwork(settings.lr_pi)
    q1 = QNet(settings.lr_q)
    q2 = QNet(settings.lr_q)

    policy.load_state_dict(checkpoint["policy"])
    q1.load_state_dict(checkpoint["q1"])
    q2.load_state_dict(checkpoint["q2"])
    policy.log_alpha = checkpoint["log_alpha"]

    logger.info(f"Model loaded: {filepath}")
    return policy, q1, q2
