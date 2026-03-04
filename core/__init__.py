"""
Package core - Modules centraux de l'application QARM-RL.

Ce package contient les modules de configuration et de logging.
"""

from core.config import settings
from core.logger import logger

__all__ = ["settings", "logger"]
