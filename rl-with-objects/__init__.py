"""
RL with Variable Objects - Bin Picking Task

Vision-based robotic bin picking with TossingBot-style grasping and throwing.
"""

__version__ = "0.1.0"
__author__ = "RL Lab"

from . import config
from . import environment
from . import models
from . import training

__all__ = [
    "config",
    "environment", 
    "models",
    "training",
]
