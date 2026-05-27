"""
This class creates an environment that rl agents can interact with. It follows the gymnasium api : it doesn't use genesis parallelisation features.
It launch the genesis env, and create a robot and enable udp connection between them : unvisible to the agent.
"""

from typing import Optional

import gymnasium as gym
import numpy as np


class QArmPIDGymEnv(gym.Env):
    """
    The steps of the simulation are not the same as the physic engine ones. Here, each steps of the rl algorithm performs an entire sequence of actions (an entire routine) in the physic engine.
    """

    def __init__(self):
        # Initialisation of the physic engine, genesis.

        # Describing the observation space
        # Since this environment only perfoms one single step, the observation space is not really relevant. It's a Multi-Armed Bandit problem.
        # Hence the dummy observation space.
        self.observation_space = gym.spaces.Discrete(1)

        # Describing the action space
        self.action_space = gym.spaces.Box(low=0, high=400, shape=(3, 3), dtype=np.float32)

    def _get_obs(self):
        """Convert internal state to observation format.

        Returns:
            Nothing relevant here.
        """
        return 0

    def _get_info(self):
        """Get auxiliary information about the current state.

        Returns:
            info (dict): a dictionary containing auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        return {}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment to an initial state and return an initial observation.

        Returns:
            obs (object): the initial observation of the space.
            info (dict): a dictionary containing auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        super().reset(seed=seed)

        # Randomly

        obs = self._get_obs()
        info = self._get_info()
        return obs, info
