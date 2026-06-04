"""
Script simple de test YOLO 3D sur le bras.
Lance avec: python test_yolo_live.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from core.qarm.real import QARMReal
from utils.logger import logger


def main():
    print("\n" + "="*60)
    print("TEST YOLO 3D - ROBOT EN DIRECT")
    print("="*60)
    
    try:
        # Initialiser le robot et la caméra
        print("\n[1] Initialisation du robot...")
        robot = QARMReal()
        robot.camera.start()  # Démarrer le thread caméra
        print("✓ Robot prêt\n")
        
        # Boucle de capture
        print("[2] Capture des frames...")
        print("Appuyez sur 'q' pour quitter\n")
        
        frame_count = 0
        while True:
            # Capturer les frames
            frames = robot.camera.get_frames()
            if frames is None:
                print("En attente de frames...")
                continue
            
            color_frame, depth_frame = frames
            frame_count += 1
            
            # ==== LA MAGIE: YOLO 3D ====
            detections = robot.camera.get_detection_points_3d(
                color_frame, depth_frame, max_objects=3
            )
            
            # Afficher les résultats
            print(f"\n--- Frame {frame_count} ---")
            if detections:
                print(f"✓ {len(detections)} objet(s) détecté(s):")
                for i, obj in enumerate(detections, 1):
                    x, y, z = obj["position_3d"]
                    print(f"  [{i}] {obj['label']}")
                    print(f"      Position: ({x:.3f}, {y:.3f}, {z:.3f}) m")
                    print(f"      Confiance: {obj['confidence']:.2f}")
            else:
                print("✗ Aucun objet détecté")
            
            # Afficher la heatmap (optionnel - ralentit peut-être)
            try:
                heatmap = robot.camera.get_heatmap(color_frame, depth_frame)
                if heatmap is not None:
                    # Afficher la heatmap (visualisation OpenCV)
                    heatmap_colored = cv2.applyColorMap(
                        (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
                    )
                    cv2.imshow("Heatmap YOLO", heatmap_colored)
            except Exception as e:
                pass
            
            # Afficher la vidéo
            display_frame = color_frame.copy()
            
            # Dessiner les détections sur la frame
            for obj in detections:
                u, v = obj["position_2d"]
                label = obj["label"]
                conf = obj["confidence"]
                
                # Cercle autour du centre
                cv2.circle(display_frame, (int(u), int(v)), 10, (0, 255, 0), 2)
                
                # Texte
                text = f"{label} {conf:.2f}"
                cv2.putText(
                    display_frame,
                    text,
                    (int(u) + 15, int(v)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )
            
            cv2.imshow("YOLO Detection", display_frame)
            
            # Quitter sur 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[3] Arrêt...")
                break
        
        # Cleanup
        robot.camera.stop()
        cv2.destroyAllWindows()
        print("✓ Terminé avec succès!\n")
        
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
