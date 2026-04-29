"""
Test script for RealSense camera with threading.
Connects to the QARM robot and tests camera frame capture and object detection.
"""

import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.qarm.real import QARMReal
from utils.logger import logger

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def test_camera_connection():
    """Test camera thread and frame capture."""
    logger.info("=" * 60)
    logger.info("Starting Camera Thread Test")
    logger.info("=" * 60)

    try:
        # Initialize robot connection (which includes RealSense camera)
        logger.info("Initializing QARM robot connection...")
        robot = QARMReal()
        logger.info("✓ Robot initialized")

        # Connect to robot (this also starts the camera thread)
        logger.info("Connecting to robot and starting camera thread...")
        robot.connect()
        logger.info("✓ Connected and camera thread started")

        # Wait for camera to warm up
        logger.info("Warming up camera...")
        time.sleep(2)
        logger.info("✓ Camera warmed up")

        # Test 1: Retrieve frames from the queue
        logger.info("\n" + "=" * 60)
        logger.info("Test 1: Frame Capture")
        logger.info("=" * 60)

        for i in range(5):
            frames = robot.camera.get_frames()
            if frames is not None:
                color_frame, depth_frame = frames
                logger.info(
                    f"✓ Frame {i+1}: Color shape={color_frame.shape}, Depth shape={depth_frame.shape}"
                )
            else:
                logger.warning(f"✗ Frame {i+1}: No frames available")
            time.sleep(0.2)

        # Test 2: Object detection
        logger.info("\n" + "=" * 60)
        logger.info("Test 2: Object Detection")
        logger.info("=" * 60)

        detection_count = 0
        for i in range(10):
            frames = robot.camera.get_frames()
            if frames is not None:
                color_frame, depth_frame = frames
                detection = robot.camera.detect_object(color_frame, depth_frame)

                if detection is not None:
                    coords, confidence, pixel_coords = detection
                    detection_count += 1
                    logger.info(
                        f"✓ Detection {detection_count}: "
                        f"3D coords={coords}, confidence={confidence:.2f}, "
                        f"pixel coords={pixel_coords}"
                    )
                else:
                    logger.debug(f"Frame {i+1}: No objects detected")
            else:
                logger.warning(f"Frame {i+1}: No frames available")

            time.sleep(0.2)

        logger.info(f"\nTotal detections found: {detection_count}/10 frames")

        # Test 3: Display frames with OpenCV (if available)
        logger.info("\n" + "=" * 60)
        logger.info("Test 3: Display Frames (ESC to exit)")
        logger.info("=" * 60)

        start_time = time.time()
        frame_count = 0
        detection_found = False

        while time.time() - start_time < 10:  # Run for 10 seconds
            frames = robot.camera.get_frames()
            if frames is not None:
                color_frame, depth_frame = frames
                frame_count += 1

                # Try to detect object
                detection = robot.camera.detect_object(color_frame, depth_frame)

                # Display frame
                display_frame = color_frame.copy()

                if detection is not None and not detection_found:
                    coords, confidence, (u, v) = detection
                    detection_found = True
                    # Draw circle at detected point
                    cv2.circle(display_frame, (u, v), 5, (0, 255, 0), -1)
                    cv2.putText(
                        display_frame,
                        f"Conf: {confidence:.2f}",
                        (u + 10, v),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )
                    logger.info(
                        f"Object detected at pixel ({u}, {v}) with confidence {confidence:.2f}"
                    )

                # Add frame counter
                cv2.putText(
                    display_frame,
                    f"Frame: {frame_count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                cv2.imshow("RealSense Camera - Thread Test", display_frame)

                # ESC to exit display loop
                if cv2.waitKey(1) & 0xFF == 27:
                    logger.info("Display stopped by user")
                    break

            time.sleep(0.01)

        cv2.destroyAllWindows()
        logger.info(f"✓ Displayed {frame_count} frames")

        # Test 4: Camera thread status
        logger.info("\n" + "=" * 60)
        logger.info("Test 4: Camera Thread Status")
        logger.info("=" * 60)

        logger.info(f"✓ Camera is running: {robot.camera.is_running}")
        logger.info(
            f"✓ Camera thread alive: {robot.camera.camera_thread.is_alive() if robot.camera.camera_thread else False}"
        )
        logger.info(f"✓ Frames in queue: {robot.camera.frame_queue.qsize()}")

        # Cleanup
        logger.info("\n" + "=" * 60)
        logger.info("Cleanup")
        logger.info("=" * 60)

        logger.info("Stopping camera thread...")
        robot.camera.stop()
        logger.info("✓ Camera thread stopped")

        logger.info("Closing robot connection...")
        robot.close()
        logger.info("✓ Robot connection closed")

        logger.info("\n" + "=" * 60)
        logger.info("✓ All tests completed successfully!")
        logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.info("\nTest interrupted by user")
        try:
            robot.close()
        except:
            pass
    except Exception as e:
        logger.error(f"✗ Test failed with error: {e}", exc_info=True)
        try:
            robot.close()
        except:
            pass


if __name__ == "__main__":
    test_camera_connection()
