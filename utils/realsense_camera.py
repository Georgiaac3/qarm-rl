from utils.logger import logger
import os
import sys
from typing import Optional, Tuple, List
import threading
import queue
import time

import numpy as np
from ultralytics import YOLO
import pyrealsense2 as rs

class RealsenseCamera:
    """
    Manages RealSense camera pipeline and YOLO object detection.
    Provides 3D coordinates of detected objects with confidence scores.
    """

    def __init__(
        self,
        model_path: str = "models/yolo_model.pt",
        confidence_threshold: float = 0.85,
        camera_width: int = 640,
        camera_height: int = 480,
        fps: int = 30,
    ):
        """
        Initialize RealSense camera and YOLO model.

        Args:
            model_path: Path to YOLO model file
            confidence_threshold: Minimum confidence for detections
            camera_width: Camera resolution width
            camera_height: Camera resolution height
            fps: Frames per second
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.fps = fps

        # Verify model exists
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            sys.exit(1)

        # Load YOLO model
        self.model = YOLO(model_path, task='detect')
        logger.info("YOLO model loaded successfully")

        # Initialize RealSense pipeline
        self.pipeline = rs.pipeline()
        self._initialize_pipeline()

        # FPS tracking
        self.frame_rate_buffer: List[float] = []
        self.fps_buffer_size = 30

        # Threading setup
        self.frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self.camera_thread: Optional[threading.Thread] = None
        self.stop_event: threading.Event = threading.Event()
        self.is_running: bool = False

    def _initialize_pipeline(self) -> None:
        """Configure and start RealSense pipeline."""
        config = rs.config()
        config.enable_stream(rs.stream.color, self.camera_width, self.camera_height, rs.format.bgr8, self.fps)
        config.enable_stream(rs.stream.depth, self.camera_width, self.camera_height, rs.format.z16, self.fps)
        self.pipeline.start(config)
        logger.info("RealSense pipeline started")

    def _camera_thread_worker(self) -> None:
        """
        Worker function that runs in a separate thread.
        Continuously captures frames and puts them in the queue.
        """
        logger.info("Camera thread started")
        try:
            while not self.stop_event.is_set():
                try:
                    frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                    align = rs.align(rs.stream.color)
                    aligned_frames = align.process(frames)

                    color_frame = aligned_frames.get_color_frame()
                    depth_frame = aligned_frames.get_depth_frame()

                    if color_frame and depth_frame:
                        color_data = np.asanyarray(color_frame.get_data())
                        # depth_data = np.asanyarray(depth_frame.get_data())

                        # Stocke uniquement des arrays dans la queue
                        try:
                            self.frame_queue.put_nowait((color_data, depth_frame))
                        except queue.Full:
                            try:
                                self.frame_queue.get_nowait()
                                self.frame_queue.put_nowait((color_data, depth_frame))
                            except queue.Empty:
                                pass

                except Exception as e:
                    logger.error(f"Error in camera thread: {e}")
                    time.sleep(0.01)
        finally:
            logger.info("Camera thread stopped")

    def start(self) -> None:
        """Start the camera thread."""
        if not self.is_running:
            self.stop_event.clear()
            self.camera_thread = threading.Thread(target=self._camera_thread_worker, daemon=True)
            self.camera_thread.start()
            self.is_running = True
            logger.info("Camera thread created and started")

    def stop(self) -> None:
        """Stop the camera thread."""
        if self.is_running:
            self.stop_event.set()
            if self.camera_thread:
                self.camera_thread.join(timeout=2.0)
            self.is_running = False
            logger.info("Camera thread stopped")

    def get_frames(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Retrieve aligned color and depth frames from the camera queue.

        Returns:
            Tuple of (color_frame, depth_frame) or None if frames unavailable
        """
        try:
            color_data, depth_frame = self.frame_queue.get(timeout=1.0)
            return color_data, depth_frame
        except queue.Empty:
            return None
        except Exception as e:
            logger.error(f"Error retrieving frames: {e}")
            return None

    def detect_object(self, frame: np.ndarray, depth_frame) -> Optional[Tuple[np.ndarray, float, Tuple[int, int]]]:
        """
        Detect object in frame using YOLO and retrieve 3D coordinates.

        Args:
            frame: Color frame from camera
            depth_frame: Depth frame from camera

        Returns:
            Tuple of (3D coordinates, confidence, pixel coordinates) or None
        """
        results = self.model(frame, verbose=False)
        detections = results[0].boxes

        for det in detections:
            conf = det.conf.item()
            if conf < self.confidence_threshold:
                continue

            # Get bounding box
            xyxy = det.xyxy.cpu().numpy().squeeze().astype(int)
            xmin, ymin, xmax, ymax = xyxy

            # Calculate center pixel
            u = int((xmin + xmax) / 2)
            v = int((ymin + ymax) / 2)

            # Get depth distance
            distance = depth_frame.get_distance(u, v)
            if distance <= 0:
                continue

            # Convert pixel to 3D world coordinates
            intr = depth_frame.profile.as_video_stream_profile().intrinsics
            point_3d = rs.rs2_deproject_pixel_to_point(intr, [u, v], distance)

            return np.array(point_3d), conf, (u, v)

        return None


    def cleanup(self) -> None:
        """Stop camera thread and clean up resources."""
        self.stop()
        self.pipeline.stop()
        logger.info("RealSense pipeline stopped and resources cleaned up")