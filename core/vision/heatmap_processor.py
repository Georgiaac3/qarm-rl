"""
Module d'extension de détection d'objets YOLO avec mappage heatmap vers coordonnées 3D réelles.
Permet de convertir les heatmaps en données 3D exploitables pour le programme RL.

Fonctionnalités principales:
- Extraction des pics de la heatmap
- Conversion des coordonnées pixels vers 3D réelles avec calibration
- Génération de heatmaps enrichies (avec coordonnées réelles et profondeur)
- Support des transformations de repère (caméra vers robot)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyrealsense2 as rs
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter, label as label_connected
from scipy.signal import argrelextrema

from core.vision.camera_calibration import CameraIntrinsics, CameraProjection
from utils.logger import logger


@dataclass
class DetectionPoint3D:
    """Représente un point d'objet détecté en 3D."""
    
    position_3d: NDArray[np.float64]  # [x, y, z] en mètres
    position_2d: Tuple[int, int]      # [u, v] en pixels
    confidence: float                 # [0, 1]
    depth: float                      # Profondeur en mètres
    label: str                        # Étiquette de classe YOLO
    
    def to_dict(self) -> dict:
        """Convertit en dictionnaire pour sérialisation."""
        return {
            "position_3d": tuple(self.position_3d),
            "position_2d": self.position_2d,
            "confidence": self.confidence,
            "depth": self.depth,
            "label": self.label
        }


@dataclass
class HeatmapEnhanced:
    """Heatmap enrichie avec informations 3D."""
    
    heatmap_2d: NDArray[np.float32]   # Heatmap en pixels (H, W)
    heatmap_3d: NDArray[np.float32]   # Heatmap 3D normalisée (profondeur)
    peak_coordinates: List[DetectionPoint3D]  # Points détectés en 3D
    depth_map: NDArray[np.float32]    # Carte de profondeur alignée (H, W)
    velocity_map: Optional[NDArray[np.float32]] = None  # Vitesse suggérée


class HeatmapProcessor:
    """
    Traite les heatmaps YOLO et les enrichit avec coordonnées 3D réelles.
    """
    
    def __init__(
        self,
        camera_intrinsics: Optional[CameraIntrinsics] = None,
        depth_scale: float = 0.001,  # Échelle de profondeur RealSense (0.001 m/unit)
    ):
        """
        Initialise le processeur de heatmap.
        
        Args:
            camera_intrinsics: Paramètres intrinsèques de la caméra (optionnel)
            depth_scale: Facteur d'échelle de la profondeur (0.001 m par unité)
        """
        self.camera_intrinsics = camera_intrinsics
        self.depth_scale = depth_scale
        if camera_intrinsics:
            self.projector = CameraProjection(camera_intrinsics)
        else:
            self.projector = None
    
    def initialize_from_realsense(self, depth_frame) -> None:
        """
        Initialise les paramètres de caméra depuis un frame RealSense.
        
        Args:
            depth_frame: Frame de profondeur RealSense
        """
        if hasattr(depth_frame, 'profile'):
            intr = depth_frame.profile.as_video_stream_profile().intrinsics
            self.camera_intrinsics = CameraIntrinsics.from_realsense(intr)
            self.projector = CameraProjection(self.camera_intrinsics)
            logger.info(f"Camera intrinsics initialized: fx={intr.fx}, fy={intr.fy}")
    
    def extract_heatmap_peaks(
        self,
        heatmap: NDArray[np.float32],
        min_distance: int = 20,
        threshold: float = 0.3
    ) -> List[Tuple[int, int, float]]:
        """
        Extrait les pics de la heatmap (maxima locaux).
        
        Args:
            heatmap: Heatmap 2D (H, W)
            min_distance: Distance minimale entre pics en pixels
            threshold: Seuil de valeur pour considérer un pic
        
        Returns:
            Liste de (u, v, valeur) des pics trouvés
        """
        peaks = []
        
        # Appliquer un filtre pour lisser et trouver les maxima locaux
        heatmap_smooth = gaussian_filter(heatmap, sigma=2)
        
        # Trouver maxima locaux en 2D (approche simple: max local dans une fenêtre)
        h, w = heatmap_smooth.shape
        
        for y in range(min_distance, h - min_distance):
            for x in range(min_distance, w - min_distance):
                window = heatmap_smooth[
                    y - min_distance:y + min_distance + 1,
                    x - min_distance:x + min_distance + 1
                ]
                
                if (heatmap_smooth[y, x] == window.max() and 
                    heatmap_smooth[y, x] > threshold):
                    peaks.append((x, y, float(heatmap_smooth[y, x])))
        
        # Retirer les doublons trop proches
        peaks = self._remove_close_peaks(peaks, min_distance)
        
        return peaks
    
    @staticmethod
    def _remove_close_peaks(
        peaks: List[Tuple[int, int, float]],
        min_distance: int
    ) -> List[Tuple[int, int, float]]:
        """Supprime les pics trop proches les uns des autres."""
        if not peaks:
            return peaks
        
        # Trier par confiance décroissante
        peaks = sorted(peaks, key=lambda p: p[2], reverse=True)
        
        filtered = []
        for peak in peaks:
            too_close = False
            for existing in filtered:
                dist = np.sqrt((peak[0] - existing[0])**2 + (peak[1] - existing[1])**2)
                if dist < min_distance:
                    too_close = True
                    break
            if not too_close:
                filtered.append(peak)
        
        return filtered
    
    def heatmap_to_3d(
        self,
        heatmap: NDArray[np.float32],
        depth_frame,
        detections: Optional[List[dict]] = None,
        extract_peaks: bool = True,
        min_distance: int = 20,
        threshold: float = 0.3
    ) -> HeatmapEnhanced:
        """
        Convertit une heatmap 2D en représentation 3D enrichie.
        
        Args:
            heatmap: Heatmap 2D (H, W) avec valeurs [0, 1]
            depth_frame: Frame de profondeur RealSense
            detections: Détections YOLO optionnelles (si None, extrait les pics)
            extract_peaks: Si True, extrait les pics de la heatmap
            min_distance: Distance minimale entre pics
            threshold: Seuil de confiance pour les pics
        
        Returns:
            HeatmapEnhanced avec données 3D
        """
        # Initialiser les intrinsèques si nécessaire
        if self.projector is None:
            self.initialize_from_realsense(depth_frame)
        
        # Extraire la carte de profondeur
        depth_array = np.asanyarray(depth_frame.get_data())
        depth_map = depth_array.astype(np.float32) * self.depth_scale
        
        # Extraire les points 3D
        peak_3d_list = []
        
        if detections:
            # Utiliser les détections YOLO fournies
            for det in detections:
                u, v = det.get("center_2d", (0, 0))
                conf = det.get("confidence", 0.5)
                label = det.get("label", "unknown")
                
                depth = depth_map[int(v), int(u)]
                
                if depth > 0:
                    point_3d = self.projector.pixel_to_3d(u, v, depth)
                    peak_3d_list.append(DetectionPoint3D(
                        position_3d=point_3d,
                        position_2d=(int(u), int(v)),
                        confidence=conf,
                        depth=depth,
                        label=label
                    ))
        
        if extract_peaks:
            # Extraire les pics supplémentaires de la heatmap
            peaks = self.extract_heatmap_peaks(heatmap, min_distance, threshold)
            
            for u, v, conf in peaks:
                depth = depth_map[int(v), int(u)]
                
                if depth > 0:
                    point_3d = self.projector.pixel_to_3d(u, v, depth)
                    peak_3d_list.append(DetectionPoint3D(
                        position_3d=point_3d,
                        position_2d=(int(u), int(v)),
                        confidence=conf,
                        depth=depth,
                        label="peak"
                    ))
        
        # Créer la heatmap 3D basée sur la profondeur
        heatmap_3d = self._create_3d_heatmap(
            heatmap,
            depth_map,
            peak_3d_list
        )
        
        return HeatmapEnhanced(
            heatmap_2d=heatmap,
            heatmap_3d=heatmap_3d,
            peak_coordinates=peak_3d_list,
            depth_map=depth_map
        )
    
    def _create_3d_heatmap(
        self,
        heatmap_2d: NDArray[np.float32],
        depth_map: NDArray[np.float32],
        peaks: List[DetectionPoint3D],
        sigma_depth: float = 0.1  # Écart-type de profondeur en mètres
    ) -> NDArray[np.float32]:
        """
        Crée une heatmap 3D pondérée par la profondeur.
        Les objets plus proches ont un poids plus important.
        
        Args:
            heatmap_2d: Heatmap 2D de base
            depth_map: Carte de profondeur
            peaks: Points détectés
            sigma_depth: Écart-type pour la pondération de profondeur
        
        Returns:
            Heatmap 3D normalisée
        """
        heatmap_3d = heatmap_2d.copy()
        
        if peaks:
            # Pondérer par la proximité (objets proches = poids élevé)
            depth_weights = np.ones_like(depth_map)
            
            for peak in peaks:
                x, y, z = peak.position_3d
                # Pondération gaussienne sur la profondeur
                depth_diff = np.abs(depth_map - z)
                depth_weight = np.exp(-(depth_diff ** 2) / (2 * sigma_depth ** 2))
                depth_weights *= (1 + depth_weight * 0.5)  # Améliorer les zones proches
            
            # Normaliser
            depth_weights = (depth_weights - depth_weights.min()) / (
                depth_weights.max() - depth_weights.min() + 1e-6
            )
            
            heatmap_3d = heatmap_2d * depth_weights
        
        # Normaliser à [0, 1]
        if heatmap_3d.max() > 0:
            heatmap_3d = heatmap_3d / heatmap_3d.max()
        
        return heatmap_3d
    
    def get_dominant_objects(
        self,
        heatmap_enhanced: HeatmapEnhanced,
        max_objects: int = 5
    ) -> List[Dict]:
        """
        Retourne les objets dominants de la scène.
        
        Args:
            heatmap_enhanced: Heatmap enrichie
            max_objects: Nombre maximum d'objets à retourner
        
        Returns:
            Liste des objets dominants (dictionnaires)
        """
        # Trier par confiance
        sorted_peaks = sorted(
            heatmap_enhanced.peak_coordinates,
            key=lambda p: p.confidence,
            reverse=True
        )[:max_objects]
        
        return [peak.to_dict() for peak in sorted_peaks]
