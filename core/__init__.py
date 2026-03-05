"""
Package core - Modules centraux de l'application QARM-RL.

Ce package contient les modules de configuration et de logging.
"""

from core.config import settings
from utils.logger import logger

__all__ = ["settings", "logger"]
