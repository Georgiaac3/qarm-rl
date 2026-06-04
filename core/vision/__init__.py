"""
Module de vision pour le robot QARM.
Gère la calibration caméra, le traitement de heatmaps et la conversion coordonnées.
"""

from core.vision.camera_calibration import CameraIntrinsics, CameraProjection
from core.vision.heatmap_processor import HeatmapEnhanced, HeatmapProcessor, DetectionPoint3D

__all__ = [
    "CameraIntrinsics",
    "CameraProjection",
    "HeatmapProcessor",
    "HeatmapEnhanced",
    "DetectionPoint3D",
]
