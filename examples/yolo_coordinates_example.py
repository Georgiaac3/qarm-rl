"""
Exemple d'utilisation des coordonnées réelles YOLO pour le programme RL.

Montre comment:
1. Obtenir les heatmaps enrichies avec coordonnées 3D réelles
2. Convertir les pixels YOLO en coordonnées exploitables pour le RL
3. Utiliser les données dans le pipeline RL
"""

import numpy as np

# ============================================================================
# EXEMPLE 1: Utiliser les heatmaps enrichies avec coordonnées 3D
# ============================================================================
def example_enhanced_heatmap(robot):
    """
    Récupère les heatmaps enrichies avec:
    - heatmap_2d: Probabilité d'agrippe en pixels
    - heatmap_3d: Probabilité pondérée par la profondeur
    - detections_3d: Objets détectés en coordonnées réelles (mètres)
    """
    frames = robot.camera.get_frames()
    if frames is None:
        return None

    color_frame, depth_frame = frames

    # === NOUVELLE FONCTIONNALITÉ: Heatmap enrichie ===
    heatmap_2d, heatmap_3d, detections_3d = robot.camera.get_heatmap_enhanced(
        color_frame, depth_frame, sigma=30.0
    )

    print("\n=== HEATMAP ENRICHIE ===")
    print(f"Heatmap 2D shape: {heatmap_2d.shape} (pixels)")
    print(f"Heatmap 3D shape: {heatmap_3d.shape} (weighted by depth)")
    print(f"Nombre d'objets détectés: {len(detections_3d)}\n")

    # Afficher les détections en 3D
    for i, det in enumerate(detections_3d):
        print(f"Objet {i + 1}:")
        print(f"  Label: {det['label']}")
        print(f"  Position réelle (mètres): x={det['position_3d'][0]:.3f}, "
              f"y={det['position_3d'][1]:.3f}, z={det['position_3d'][2]:.3f}")
        print(f"  Position pixels: u={det['position_2d'][0]}, v={det['position_2d'][1]}")
        print(f"  Confiance: {det['confidence']:.2f}")
        print(f"  Profondeur: {det['depth']:.3f} m\n")

    return heatmap_2d, heatmap_3d, detections_3d


# ============================================================================
# EXEMPLE 2: Utiliser les points 3D directement pour le RL
# ============================================================================
def example_3d_detection_points(robot):
    """
    Récupère les objets détectés directement en coordonnées réelles (mètres).
    Parfait pour alimenter directement le programme RL.
    """
    frames = robot.camera.get_frames()
    if frames is None:
        return None

    color_frame, depth_frame = frames

    # === NOUVELLE FONCTIONNALITÉ: Points 3D directs ===
    detections_3d = robot.camera.get_detection_points_3d(
        color_frame,
        depth_frame,
        max_objects=5  # Top 5 objets par confiance
    )

    print("\n=== OBJETS DÉTECTÉS EN 3D (POUR LE RL) ===")
    for i, obj in enumerate(detections_3d):
        x, y, z = obj["position_3d"]
        print(f"Objet {i + 1}: {obj['label']}")
        print(f"  Position réelle: ({x:.4f}, {y:.4f}, {z:.4f}) mètres")
        print(f"  Confiance: {obj['confidence']:.2f}\n")

    return detections_3d


# ============================================================================
# EXEMPLE 3: Créer une observation RL à partir de la heatmap
# ============================================================================
def example_rl_observation_from_heatmap(robot, target_size=(64, 64)):
    """
    Prépare une observation pour le RL:
    1. Heatmap enrichie 3D redimensionnée
    2. Meilleur objet détecté en coordonnées réelles
    """
    frames = robot.camera.get_frames()
    if frames is None:
        return None

    color_frame, depth_frame = frames
    import cv2

    # Obtenir les heatmaps
    heatmap_2d, heatmap_3d, detections_3d = robot.camera.get_heatmap_enhanced(
        color_frame, depth_frame
    )

    # Redimensionner la heatmap pour le RL (ex: (64, 64))
    heatmap_resized = cv2.resize(heatmap_3d, target_size, interpolation=cv2.INTER_LINEAR)

    # Normaliser à [0, 1]
    heatmap_normalized = heatmap_resized / (heatmap_resized.max() + 1e-6)

    # Obtenir le meilleur objet
    best_object = None
    if detections_3d:
        best_object = detections_3d[0]  # Déjà trié par confiance

    # Construire l'observation RL
    rl_observation = {
        "heatmap": heatmap_normalized,  # Shape: (64, 64) - Image locale
        "target_position": best_object["position_3d"] if best_object else None,  # (x, y, z)
        "target_confidence": best_object["confidence"] if best_object else 0.0,
        "all_targets": detections_3d,  # Liste complète pour contexte
    }

    print("\n=== OBSERVATION RL ===")
    print(f"Heatmap input shape: {rl_observation['heatmap'].shape}")
    if rl_observation["target_position"]:
        print(f"Target position: {rl_observation['target_position']}")
        print(f"Target confidence: {rl_observation['target_confidence']:.2f}")
    print(f"Number of targets: {len(rl_observation['all_targets'])}\n")

    return rl_observation


# ============================================================================
# EXEMPLE 4: Intégration avec le RL Environment
# ============================================================================
def example_rl_environment_integration():
    """
    Montre comment intégrer les détections YOLO dans QArmGymEnv.
    """
    from core.qarm.real import QARMReal
    from reinforcement_learning.gym_environment import QArmGymEnv

    # Initialiser le robot
    robot = QARMReal()
    robot.connect()

    # Créer l'environnement RL
    env = QArmGymEnv(qarm_interface=robot, max_steps=1000)

    # Dans la boucle d'entraînement RL:
    obs, _ = env.reset()

    for step in range(100):
        # Capturer les frames
        frames = robot.camera.get_frames()
        if frames is None:
            continue

        color_frame, depth_frame = frames

        # === UTILISER LES DÉTECTIONS YOLO ===
        # Option 1: Utiliser les points 3D comme target
        detected_targets = robot.camera.get_detection_points_3d(
            color_frame, depth_frame, max_objects=1
        )

        if detected_targets:
            target_pos = detected_targets[0]["position_3d"]
            print(f"Detected target at: {target_pos}")

            # Prendre une action RL en direction de la cible
            action = np.random.randn(4)  # À remplacer par votre politique
            obs, reward, terminated, truncated, info = env.step(action)

        # Option 2: Utiliser la heatmap enrichie
        heatmap_2d, heatmap_3d, detections = robot.camera.get_heatmap_enhanced(
            color_frame, depth_frame
        )

        # Redimensionner et utiliser dans le RL
        import cv2
        heatmap_for_rl = cv2.resize(heatmap_3d, (64, 64))

        if terminated or truncated:
            obs, _ = env.reset()

    env.close()


# ============================================================================
# EXEMPLE 5: Conversion manuelle pixel → 3D réel
# ============================================================================
def example_pixel_to_3d_conversion(robot):
    """
    Montre la conversion manuelle de coordonnées pixels vers coordonnées 3D.
    """
    from core.vision.camera_calibration import CameraProjection
    import pyrealsense2 as rs

    frames = robot.camera.get_frames()
    if frames is None:
        return None

    color_frame, depth_frame = frames

    # Obtenir les intrinsèques de la caméra
    intr = depth_frame.profile.as_video_stream_profile().intrinsics

    # Créer le projecteur
    from core.vision import CameraIntrinsics
    camera_intrinsics = CameraIntrinsics.from_realsense(intr)
    projector = CameraProjection(camera_intrinsics)

    # Exemple: Convertir un point pixel (320, 240) avec profondeur 0.5m
    u, v = 320, 240
    depth = depth_frame.get_distance(u, v)

    if depth > 0:
        point_3d = projector.pixel_to_3d(u, v, depth)
        print(f"\n=== CONVERSION PIXEL → 3D ===")
        print(f"Pixel: ({u}, {v})")
        print(f"Profondeur: {depth:.3f} m")
        print(f"Position 3D: ({point_3d[0]:.3f}, {point_3d[1]:.3f}, {point_3d[2]:.3f}) m")
    else:
        print(f"Pas de profondeur valide pour ({u}, {v})")

    return point_3d if depth > 0 else None


# ============================================================================
# RÉSUMÉ DES NOUVELLES FONCTIONNALITÉS
# ============================================================================
"""
NOUVELLES MÉTHODES DISPONIBLES:

1. robot.camera.get_heatmap_enhanced(frame, depth_frame)
   → Retourne: (heatmap_2d, heatmap_3d, detections_3d)
   → heatmap_2d: Heatmap standard en pixels
   → heatmap_3d: Heatmap pondérée par la profondeur (meilleur pour les objets proches)
   → detections_3d: Liste des objets en coordonnées réelles (mètres)

2. robot.camera.get_detection_points_3d(frame, depth_frame)
   → Retourne: Liste des objets détectés avec position_3d en mètres
   → Parfait pour alimenter directement le RL

3. Classe HeatmapProcessor (core.vision.heatmap_processor)
   → Traite les heatmaps et extrait les pics en 3D
   → Convertit pixels → coordonnées réelles
   → Crée des heatmaps pondérées par la profondeur

CONVERSION DE COORDONNÉES:

Pixels → 3D réel:
  position_3d = projector.pixel_to_3d(u, v, depth)
  # Retourne (x, y, z) en mètres

3D réel → Pixels:
  u, v = projector.point_3d_to_pixel(x, y, z)
  # Retourne (u, v) en pixels

UTILISATION TYPIQUE DANS LE RL:

1. Capturer frames
2. Obtenir détections: detections = robot.camera.get_detection_points_3d(...)
3. Extraire target position: target_pos = detections[0]["position_3d"]
4. Utiliser dans RL reward/observation:
   - Récompense: -distance(end_effector, target_pos)
   - Observation: [angles, target_position]
"""


if __name__ == "__main__":
    print(__doc__)
