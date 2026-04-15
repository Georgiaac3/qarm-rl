import sys
import time
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.qarm.real import QARMReal
from utils.logger import logger

def to_numpy(frame):
    """Convert RealSense frame → numpy array si nécessaire"""
    if frame is None:
        return None
    if isinstance(frame, np.ndarray):
        return frame
    return np.asanyarray(frame.get_data())

def test_camera_connection():
    logger.info("=" * 60)
    logger.info("Starting Camera Thread Test")
    logger.info("=" * 60)

    try:
        robot = QARMReal()
        robot.connect()
        time.sleep(2)  # Warm-up
        logger.info("Camera warm-up done")

        frame_rate_buffer = []
        fps_avg_len = 30
        frame_id = 0
        last_detections = []
        last_detection_time = 0
        PERSIST_TIME = 0.5  # seconds
        DETECTION_EVERY = 10

        while True:
            t_start = time.perf_counter()
            frames = robot.camera.get_frames()
            if frames is None:
                continue

            color_frame, depth_frame = frames
            display_frame = to_numpy(color_frame).copy()

            detections = []
            current_time = time.time()

            if frame_id % DETECTION_EVERY == 0:
                try:
                    detections = robot.camera.get_detections(color_frame)
                    if detections:
                        last_detections = detections
                        last_detection_time = current_time
                        logger.info(f"DETECTIONS: {detections}")
                except Exception as e:
                    logger.warning(f"Detection error: {e}")
            
            if current_time - last_detection_time < PERSIST_TIME:
                for detection in last_detections:
                    xmin, ymin, xmax, ymax = detection["bbox"]
                    label = detection["label"]
                    conf = detection["confidence"]

                    cv2.rectangle(display_frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

                    text = f"{label} {conf:.2f}"
                    cv2.putText(display_frame, text, (xmin, ymin - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Calcul FPS
            t_stop = time.perf_counter()
            fps = 1.0 / (t_stop - t_start)
            frame_rate_buffer.append(fps)
            if len(frame_rate_buffer) > fps_avg_len:
                frame_rate_buffer.pop(0)
            avg_fps = np.mean(frame_rate_buffer)
            cv2.putText(display_frame, f"FPS: {avg_fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Affichage
            cv2.imshow("Camera Thread Test", display_frame)

            # Quit
            if cv2.waitKey(1) & 0xFF in [ord('q'), ord('Q')]:
                break

            frame_id += 1

        cv2.destroyAllWindows()
        robot.camera.stop()
        robot.close()
        logger.info("✓ Test completed successfully!")

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        try:
            robot.close()
        except:
            pass

if __name__ == "__main__":
    test_camera_connection()