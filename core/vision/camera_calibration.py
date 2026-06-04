"""
Module de calibration caméra pour la conversion de coordonnées pixel vers coordonnées 3D réelles.
Utilise les paramètres de la caméra RealSense pour effectuer la projection inverse.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class CameraIntrinsics:
    """Paramètres intrinsèques de la caméra RealSense RGB-D."""
    
    fx: float  # Focale en x (pixels)
    fy: float  # Focale en y (pixels)
    cx: float  # Centre optique en x (pixels)
    cy: float  # Centre optique en y (pixels)
    width: int  # Largeur de l'image
    height: int  # Hauteur de l'image
    
    @classmethod
    def from_realsense(cls, intrinsics) -> "CameraIntrinsics":
        """
        Crée une instance à partir des paramètres intrinsèques RealSense.
        
        Args:
            intrinsics: Objet pyrealsense2.intrinsics
        
        Returns:
            Instance de CameraIntrinsics
        """
        return cls(
            fx=intrinsics.fx,
            fy=intrinsics.fy,
            cx=intrinsics.ppx,
            cy=intrinsics.ppy,
            width=intrinsics.width,
            height=intrinsics.height
        )


class CameraProjection:
    """
    Classe pour gérer la conversion entre coordonnées image et coordonnées 3D réelles.
    Supporte :
    - Pixel (u, v) + profondeur (z) → Point 3D (x, y, z)
    - Point 3D (x, y, z) → Pixel (u, v)
    """
    
    def __init__(self, intrinsics: CameraIntrinsics):
        """
        Initialise le projecteur caméra.
        
        Args:
            intrinsics: Paramètres intrinsèques de la caméra
        """
        self.intrinsics = intrinsics
        
        # Matrice de calibration (matrice K)
        self.K = np.array([
            [intrinsics.fx, 0, intrinsics.cx],
            [0, intrinsics.fy, intrinsics.cy],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # Matrice inverse pour la déprojecttion
        self.K_inv = np.linalg.inv(self.K)
    
    def pixel_to_3d(
        self,
        u: float,
        v: float,
        depth: float
    ) -> NDArray[np.float64]:
        """
        Convertit une coordonnée pixel avec profondeur en point 3D.
        
        Formule:
            [x]   [1/fx    0  -cx/fx] [u*d]
            [y] = [  0   1/fy -cy/fy] [v*d]
            [z]   [  0     0    1   ] [ d ]
        
        Args:
            u: Coordonnée x du pixel (0 à width)
            v: Coordonnée y du pixel (0 à height)
            depth: Profondeur en millimètres (ou en mètres selon la caméra)
        
        Returns:
            Point 3D [x, y, z] en coordonnées caméra
        """
        if depth <= 0:
            return np.array([np.nan, np.nan, np.nan], dtype=np.float64)
        
        # Coordonnées normalisées
        x_norm = (u - self.intrinsics.cx) / self.intrinsics.fx
        y_norm = (v - self.intrinsics.cy) / self.intrinsics.fy
        
        # Point 3D dans le repère caméra
        x = x_norm * depth
        y = y_norm * depth
        z = depth
        
        return np.array([x, y, z], dtype=np.float64)
    
    def pixels_to_3d(
        self,
        pixels: NDArray[np.float64],
        depths: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Convertit plusieurs coordonnées pixels avec profondeurs en points 3D.
        
        Args:
            pixels: Array de forme (N, 2) avec les coordonnées [u, v]
            depths: Array de forme (N,) avec les profondeurs
        
        Returns:
            Array de forme (N, 3) avec les points 3D [x, y, z]
        """
        if pixels.shape[0] == 0:
            return np.empty((0, 3), dtype=np.float64)
        
        N = pixels.shape[0]
        points_3d = np.zeros((N, 3), dtype=np.float64)
        
        for i in range(N):
            points_3d[i] = self.pixel_to_3d(pixels[i, 0], pixels[i, 1], depths[i])
        
        return points_3d
    
    def point_3d_to_pixel(
        self,
        x: float,
        y: float,
        z: float
    ) -> Tuple[float, float]:
        """
        Convertit un point 3D en coordonnée pixel (projection perspective).
        
        Formule:
            u = fx * (x/z) + cx
            v = fy * (y/z) + cy
        
        Args:
            x, y, z: Coordonnées 3D dans le repère caméra
        
        Returns:
            Tuple (u, v) avec les coordonnées pixels
        """
        if z <= 0:
            return (np.nan, np.nan)
        
        u = self.intrinsics.fx * (x / z) + self.intrinsics.cx
        v = self.intrinsics.fy * (y / z) + self.intrinsics.cy
        
        return (u, v)
    
    def is_in_image(self, u: float, v: float) -> bool:
        """Vérifie si les coordonnées pixel sont dans les limites de l'image."""
        return 0 <= u < self.intrinsics.width and 0 <= v < self.intrinsics.height
