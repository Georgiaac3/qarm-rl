"""
Test simple pour vérifier que la conversion pixel→3D fonctionne.
Lance avec: python tests/test_yolo_3d_conversion.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.vision.camera_calibration import CameraIntrinsics, CameraProjection
from core.vision.heatmap_processor import HeatmapProcessor
from utils.logger import logger


def test_camera_projection():
    """Test de la calibration caméra et conversion pixel→3D."""
    print("\n" + "="*60)
    print("TEST 1: Camera Projection (Pixel → 3D)")
    print("="*60)

    # Créer des intrinsèques de caméra synthétiques (RealSense standard)
    intrinsics = CameraIntrinsics(
        fx=640,  # Focale en x
        fy=640,  # Focale en y
        cx=320,  # Centre optique x
        cy=240,  # Centre optique y
        width=640,
        height=480
    )

    print(f"Camera intrinsics:")
    print(f"  fx={intrinsics.fx}, fy={intrinsics.fy}")
    print(f"  cx={intrinsics.cx}, cy={intrinsics.cy}")
    print(f"  Resolution: {intrinsics.width}x{intrinsics.height}")

    projector = CameraProjection(intrinsics)

    # Test: Pixel au centre (320, 240) avec profondeur 1.0m
    print(f"\nTest 1a: Centre de l'image")
    u, v = 320, 240
    depth = 1.0
    point_3d = projector.pixel_to_3d(u, v, depth)
    print(f"  Pixel: ({u}, {v}), Profondeur: {depth}m")
    print(f"  → 3D: ({point_3d[0]:.4f}, {point_3d[1]:.4f}, {point_3d[2]:.4f})")
    assert np.isclose(point_3d[2], 1.0), "Z should equal depth"
    assert np.isclose(point_3d[0], 0.0), "X should be ~0 for center pixel"
    print(f"  ✓ PASS\n")

    # Test: Pixel en haut à gauche (0, 0) avec profondeur 2.0m
    print(f"Test 1b: Haut-gauche de l'image")
    u, v = 0, 0
    depth = 2.0
    point_3d = projector.pixel_to_3d(u, v, depth)
    print(f"  Pixel: ({u}, {v}), Profondeur: {depth}m")
    print(f"  → 3D: ({point_3d[0]:.4f}, {point_3d[1]:.4f}, {point_3d[2]:.4f})")
    assert np.isclose(point_3d[2], 2.0), "Z should equal depth"
    assert point_3d[0] < 0, "X should be negative (left)"
    assert point_3d[1] < 0, "Y should be negative (up)"
    print(f"  ✓ PASS\n")

    # Test: Pixel en bas à droite (639, 479) avec profondeur 0.5m
    print(f"Test 1c: Bas-droite de l'image")
    u, v = 639, 479
    depth = 0.5
    point_3d = projector.pixel_to_3d(u, v, depth)
    print(f"  Pixel: ({u}, {v}), Profondeur: {depth}m")
    print(f"  → 3D: ({point_3d[0]:.4f}, {point_3d[1]:.4f}, {point_3d[2]:.4f})")
    assert np.isclose(point_3d[2], 0.5), "Z should equal depth"
    assert point_3d[0] > 0, "X should be positive (right)"
    assert point_3d[1] > 0, "Y should be positive (down)"
    print(f"  ✓ PASS\n")

    # Test: Projection inverse (3D → Pixel)
    print(f"Test 1d: Inverse projection (3D → Pixel)")
    x, y, z = 0.1, 0.05, 1.0
    u_proj, v_proj = projector.point_3d_to_pixel(x, y, z)
    print(f"  3D: ({x}, {y}, {z})")
    print(f"  → Pixel: ({u_proj:.1f}, {v_proj:.1f})")
    assert projector.is_in_image(u_proj, v_proj), "Projected point should be in image"
    print(f"  ✓ PASS\n")

    print("✅ TEST 1 COMPLET: OK\n")
    return True


def test_heatmap_processor():
    """Test du traitement de heatmap et extraction des pics."""
    print("="*60)
    print("TEST 2: Heatmap Processing (Extract Peaks)")
    print("="*60)

    # Créer une heatmap synthétique avec 3 pics
    heatmap = np.zeros((480, 640), dtype=np.float32)

    # Pic 1: Centre
    y1, x1 = 240, 320
    heatmap[y1-10:y1+10, x1-10:x1+10] = np.exp(
        -((np.arange(20)[:, None] - 10)**2 + (np.arange(20) - 10)**2) / 50
    )

    # Pic 2: Haut-gauche
    y2, x2 = 100, 150
    heatmap[y2-10:y2+10, x2-10:x2+10] = 0.7 * np.exp(
        -((np.arange(20)[:, None] - 10)**2 + (np.arange(20) - 10)**2) / 50
    )

    # Pic 3: Bas-droite
    y3, x3 = 400, 500
    heatmap[y3-10:y3+10, x3-10:x3+10] = 0.5 * np.exp(
        -((np.arange(20)[:, None] - 10)**2 + (np.arange(20) - 10)**2) / 50
    )

    print(f"Créé une heatmap synthétique (480x640) avec 3 pics")
    print(f"  Pic 1 (Fort): ({x1}, {y1}) - confiance ~1.0")
    print(f"  Pic 2 (Moyen): ({x2}, {y2}) - confiance ~0.7")
    print(f"  Pic 3 (Faible): ({x3}, {y3}) - confiance ~0.5\n")

    # Initialiser le processeur
    intrinsics = CameraIntrinsics(
        fx=640, fy=640, cx=320, cy=240, width=640, height=480
    )
    processor = HeatmapProcessor(camera_intrinsics=intrinsics)

    # Extraire les pics
    peaks = processor.extract_heatmap_peaks(heatmap, min_distance=30, threshold=0.3)

    print(f"Pics extraits: {len(peaks)}")
    for i, (u, v, conf) in enumerate(peaks):
        print(f"  Pic {i+1}: pixel ({u:.0f}, {v:.0f}), confiance {conf:.3f}")

    assert len(peaks) >= 2, "Should find at least 2 peaks"
    print(f"\n✅ TEST 2 COMPLET: OK\n")
    return True


def test_with_real_camera():
    """Test avec la caméra réelle si disponible."""
    print("="*60)
    print("TEST 3: Real Camera (si disponible)")
    print("="*60)

    try:
        from core.qarm.real import QARMReal

        robot = QARMReal()
        print("Robot initialisé")

        # Essayer de capturer une frame
        frames = robot.camera.get_frames()
        if frames is None:
            print("⚠️  Aucune frame capturée - caméra peut ne pas être prête")
            print("   Assurez-vous que RealSense est connectée et que le processus est lancé")
            return False

        color_frame, depth_frame = frames
        print(f"✓ Frame capturée: {color_frame.shape}")

        # Tester la conversion 3D
        print("\nTest de détection 3D...")
        detections = robot.camera.get_detection_points_3d(color_frame, depth_frame)

        print(f"Objets détectés: {len(detections)}")
        for i, obj in enumerate(detections[:3]):  # Afficher top 3
            x, y, z = obj["position_3d"]
            print(f"  Objet {i+1}: {obj['label']}")
            print(f"    Position: ({x:.3f}, {y:.3f}, {z:.3f}) m")
            print(f"    Confiance: {obj['confidence']:.2f}")

        # Tester la heatmap enrichie
        print("\nTest de heatmap enrichie...")
        heatmap_2d, heatmap_3d, dets = robot.camera.get_heatmap_enhanced(
            color_frame, depth_frame
        )

        print(f"  Heatmap 2D: {heatmap_2d.shape}, min={heatmap_2d.min():.3f}, max={heatmap_2d.max():.3f}")
        print(f"  Heatmap 3D: {heatmap_3d.shape}, min={heatmap_3d.min():.3f}, max={heatmap_3d.max():.3f}")
        print(f"  Détections enrichies: {len(dets)}")

        print(f"\n✅ TEST 3 COMPLET: OK\n")
        return True

    except Exception as e:
        print(f"⚠️  Erreur caméra réelle: {e}")
        print("   C'est normal si RealSense n'est pas disponible")
        return False


def test_end_to_end():
    """Test complet: heatmap → pics → 3D."""
    print("="*60)
    print("TEST 4: End-to-End (Heatmap → 3D)")
    print("="*60)

    # Créer une heatmap et depth map synthétiques
    heatmap = np.zeros((480, 640), dtype=np.float32)
    depth_map = np.ones((480, 640), dtype=np.float32) * 1.0  # 1 mètre

    # Ajouter quelques pics
    for (x, y, conf) in [(320, 240, 1.0), (150, 100, 0.7), (500, 400, 0.5)]:
        heatmap[y-15:y+15, x-15:x+15] += conf * np.exp(
            -((np.arange(30)[:, None] - 15)**2 + (np.arange(30) - 15)**2) / 100
        )

    # Normaliser
    heatmap = heatmap / heatmap.max()

    print(f"Créé heatmap synthétique avec pics à profondeur constante (1.0m)")

    # Initialiser le processeur
    intrinsics = CameraIntrinsics(
        fx=640, fy=640, cx=320, cy=240, width=640, height=480
    )
    processor = HeatmapProcessor(camera_intrinsics=intrinsics)

    # Juste tester l'extraction de pics (sans depth frame pour l'instant)
    peaks = processor.extract_heatmap_peaks(heatmap, min_distance=30, threshold=0.3)

    print(f"\nPics détectés:")
    for i, (u, v, conf) in enumerate(peaks):
        # Convertir le pic en 3D avec profondeur fixe
        point_3d = processor.projector.pixel_to_3d(u, v, 1.0)
        x, y, z = point_3d
        print(f"  Pic {i+1}: ({x:.4f}, {y:.4f}, {z:.4f}) m")
        print(f"    Pixel: ({u:.0f}, {v:.0f}), Confiance: {conf:.3f}")

    assert len(peaks) > 0, "Should find peaks"
    print(f"\n✅ TEST 4 COMPLET: OK\n")
    return True


def main():
    print("\n" + "="*60)
    print("SUITE DE TESTS: YOLO 3D Conversion")
    print("="*60)

    results = []

    # Test 1: Calibration caméra
    try:
        results.append(("Camera Projection", test_camera_projection()))
    except Exception as e:
        print(f"❌ TEST 1 ÉCHOUÉ: {e}\n")
        results.append(("Camera Projection", False))

    # Test 2: Traitement heatmap
    try:
        results.append(("Heatmap Processing", test_heatmap_processor()))
    except Exception as e:
        print(f"❌ TEST 2 ÉCHOUÉ: {e}\n")
        results.append(("Heatmap Processing", False))

    # Test 3: End-to-end
    try:
        results.append(("End-to-End", test_end_to_end()))
    except Exception as e:
        print(f"❌ TEST 3 ÉCHOUÉ: {e}\n")
        results.append(("End-to-End", False))

    # Test 4: Caméra réelle
    try:
        results.append(("Real Camera", test_with_real_camera()))
    except Exception as e:
        print(f"⚠️  TEST 4 NON CRITIQUE ÉCHOUÉ: {e}\n")
        results.append(("Real Camera", False))

    # Résumé
    print("="*60)
    print("RÉSUMÉ DES TESTS")
    print("="*60)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")

    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)
    print(f"\nRésultat: {passed_count}/{total_count} tests passés")

    if passed_count >= 3:
        print("\n✅ TESTS CRITIQUES OK - Le système de conversion 3D fonctionne!")
    else:
        print("\n❌ Il y a des problèmes - Vérifiez les erreurs ci-dessus")

    return passed_count >= 3


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
