import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.qarm.real import QARMReal
from utils.logger import logger


def to_numpy(frame):
    if frame is None:
        return None
    if isinstance(frame, np.ndarray):
        return frame
    return np.asanyarray(frame.get_data())


def test_camera_connection():
    logger.info("=" * 60)
    logger.info("Starting Camera Thread Test")
    logger.info("=" * 60)

    robot = None
    try:
        robot = QARMReal()
        robot.connect()
        time.sleep(2)

        frame_rate_buffer = []
        fps_avg_len = 30
        frame_id = 0
        last_detections = []
        last_detection_time = 0
        PERSIST_TIME = 0.5  # seconds to keep displaying old bbox detections
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
                    detections = robot.camera.get_detections(color_frame, depth_frame)
                    if detections:
                        last_detections = detections
                        last_detection_time = current_time
                        logger.info(f"DETECTIONS: {detections}")
                except Exception as e:
                    logger.warning(f"Detection error: {e}")

            # Get heatmap
            try:
                heatmap = robot.camera.get_heatmap(color_frame, depth_frame)
                # Colorize heatmap for visualization (jet colormap)
                heatmap_colored = cv2.applyColorMap(
                    (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
                )
            except Exception as e:
                logger.warning(f"Heatmap error: {e}")
                heatmap_colored = None

            if current_time - last_detection_time < PERSIST_TIME:
                for detection in last_detections:
                    xmin, ymin, xmax, ymax = detection["bbox"]
                    label = detection["label"]
                    conf = detection["confidence"]

                    cv2.rectangle(display_frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

                    text = f"{label} {conf:.2f}"
                    cv2.putText(
                        display_frame,
                        text,
                        (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

            # Display FPS
            t_stop = time.perf_counter()
            fps = 1.0 / (t_stop - t_start)
            frame_rate_buffer.append(fps)
            if len(frame_rate_buffer) > fps_avg_len:
                frame_rate_buffer.pop(0)
            avg_fps = np.mean(frame_rate_buffer)
            cv2.putText(
                display_frame,
                f"FPS: {avg_fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            # Combine display_frame and heatmap
            display_combined = display_frame.copy()
            if heatmap_colored is not None:
                # Stack frames horizontally
                display_combined = np.hstack([display_frame, heatmap_colored])
                # Add labels
                cv2.putText(
                    display_combined,
                    "Camera",
                    (10, display_frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                )
                cv2.putText(
                    display_combined,
                    "Grasp Heatmap",
                    (display_frame.shape[1] + 10, display_frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                )

            # Display
            cv2.imshow("Camera Thread Test", display_combined)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                logger.info("Exit requested by user")
                break

            frame_id += 1
        logger.info("✓ Test completed successfully!")

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
    finally:
        try:
            if robot is not None:
                robot.close()
        except Exception:
            pass


if __name__ == "__main__":
    test_camera_connection()
