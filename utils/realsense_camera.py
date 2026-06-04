"""
To use it you basically need to do:

frames = robot.camera.get_frames()
if frames is not None:
    color_frame, depth_frame = frames

    # Get detections with 3D coordinates
    detections = robot.camera.get_detections(color_frame, depth_frame)
    for det in detections:
        print(f"Object: {det['label']}")
        print(f"  2D center: {det['center_2d']}")
        print(f"  3D center: {det['center_3d']}")  # (x, y, z) in meters

    # Get heatmap for RL input (grasp probability map)
    heatmap = robot.camera.get_heatmap(color_frame, depth_frame)
    # heatmap is shape (480, 640) with values in [0, 1]
    # Can be fed directly to RL model

    # Or get both grasp probability and velocity map
    heatmap, velocity_map = robot.camera.get_heatmap_with_velocity(color_frame, depth_frame)
    # velocity_map guides the grasp speed (0 to 1)
"""

import queue
import threading
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs
from scipy.ndimage import gaussian_filter
from ultralytics import YOLO

from core.config import settings
from core.vision.heatmap_processor import HeatmapProcessor
from utils.logger import logger


class RealsenseCamera:
    """
    Manages RealSense camera pipeline and YOLO object detection.
    Provides 2D coordinates of detected objects with confidence scores.
    """

    def __init__(
        self,
        model_path: str = settings.yolo_model,
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

        # Load YOLO model
        self.model = YOLO(model_path)
        logger.info("YOLO model loaded successfully")

        # Initialize RealSense pipeline
        self.pipeline = rs.pipeline()
        self._initialize_pipeline()

        # Initialize heatmap processor for 3D coordinates
        self.heatmap_processor = HeatmapProcessor(depth_scale=0.001)

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
        config.enable_stream(
            rs.stream.color, self.camera_width, self.camera_height, rs.format.bgr8, self.fps
        )
        config.enable_stream(
            rs.stream.depth, self.camera_width, self.camera_height, rs.format.z16, self.fps
        )
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

    def detect_object(
        self, frame: np.ndarray, depth_frame
    ) -> Optional[Tuple[np.ndarray, float, Tuple[int, int]]]:
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

        for detection in detections:
            conf = detection.conf.item()
            if conf < self.confidence_threshold:
                continue

            # Get bounding box
            xyxy = detection.xyxy.cpu().numpy().squeeze().astype(int)
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

    def get_detections(self, frame: np.ndarray, depth_frame=None):
        """
        Run YOLO detection and return all bounding boxes with labels.

        Args:
            frame: input image
            depth_frame: Optional depth frame for 3D coordinates

        Returns:
            List of dict:
            [
                {
                    "label": str,
                    "confidence": float,
                    "bbox": (xmin, ymin, xmax, ymax),
                    "center_2d": (u, v),
                    "center_3d": (x, y, z) or None if depth_frame not provided
                }
            ]
        """
        results = self.model(frame, verbose=False)
        detections = results[0].boxes

        output = []
        for detection in detections:
            conf = float(detection.conf[0])
            if conf < self.confidence_threshold:
                continue

            # bbox
            xmin, ymin, xmax, ymax = detection.xyxy[0].cpu().numpy().astype(int)

            # center pixel
            u = int((xmin + xmax) / 2)
            v = int((ymin + ymax) / 2)

            # class
            cls_id = int(detection.cls[0])
            label = self.model.names[cls_id]

            # Get 3D coordinates if depth_frame provided
            center_3d = None
            if depth_frame is not None:
                distance = depth_frame.get_distance(u, v)
                if distance > 0:
                    intr = depth_frame.profile.as_video_stream_profile().intrinsics
                    point_3d = rs.rs2_deproject_pixel_to_point(intr, [u, v], distance)
                    center_3d = tuple(point_3d)

            output.append(
                {
                    "label": label,
                    "confidence": conf,
                    "bbox": (xmin, ymin, xmax, ymax),
                    "center_2d": (u, v),
                    "center_3d": center_3d,
                }
            )

        return output

    def get_heatmap(self, frame: np.ndarray, depth_frame=None, sigma: float = 30.0) -> np.ndarray:
        """
        Generate a heatmap of grasp probabilities from YOLO detections.
        Each detected object creates a Gaussian blob centered at its location.
        Amplitude is proportional to confidence score.

        Args:
            frame: Input image
            depth_frame: Optional depth frame for 3D weighting
            sigma: Standard deviation of Gaussian blobs (in pixels)

        Returns:
            Heatmap array (H, W) with values in [0, 1] representing grasp probability
        """
        # Initialize heatmap
        heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)

        # Get detections
        results = self.model(frame, verbose=False)
        detections = results[0].boxes

        for detection in detections:
            conf = float(detection.conf[0])
            if conf < self.confidence_threshold:
                continue

            # Get center coordinates
            xmin, ymin, xmax, ymax = detection.xyxy[0].cpu().numpy().astype(int)
            u = int((xmin + xmax) / 2)
            v = int((ymin + ymax) / 2)

            # Create Gaussian blob for this detection
            y, x = np.ogrid[
                : frame.shape[0],
                : frame.shape[1],
            ]
            gaussian = np.exp(-((x - u) ** 2 + (y - v) ** 2) / (2 * sigma**2))
            
            # Weight by confidence
            heatmap += gaussian * conf

        # Normalize to [0, 1]
        max_val = heatmap.max()
        if max_val > 0:
            heatmap = heatmap / max_val

        return heatmap

    def get_heatmap_with_velocity(
        self, frame: np.ndarray, depth_frame=None, sigma: float = 30.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate heatmap for grasp location and velocity map for grasp speed.

        Args:
            frame: Input image
            depth_frame: Optional depth frame
            sigma: Standard deviation of Gaussian blobs

        Returns:
            Tuple of (heatmap, velocity_map)
            - heatmap: Grasp probability map (H, W)
            - velocity_map: Suggested velocities (H, W) based on confidence
                           Higher confidence = higher velocity (0 to 1)
        """
        heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
        velocity_map = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)

        results = self.model(frame, verbose=False)
        detections = results[0].boxes

        for detection in detections:
            conf = float(detection.conf[0])
            if conf < self.confidence_threshold:
                continue

            xmin, ymin, xmax, ymax = detection.xyxy[0].cpu().numpy().astype(int)
            u = int((xmin + xmax) / 2)
            v = int((ymin + ymax) / 2)

            # Create Gaussian blob
            y, x = np.ogrid[: frame.shape[0], : frame.shape[1]]
            gaussian = np.exp(-((x - u) ** 2 + (y - v) ** 2) / (2 * sigma**2))

            # Update heatmap with confidence
            heatmap += gaussian * conf
            
            # Velocity proportional to confidence
            # (more confident detection = faster grasp)
            velocity_map += gaussian * conf

        # Normalize
        max_val = heatmap.max()
        if max_val > 0:
            heatmap = heatmap / max_val
            velocity_map = velocity_map / velocity_map.max() if velocity_map.max() > 0 else velocity_map

        return heatmap, velocity_map

    def get_heatmap_enhanced(self, frame: np.ndarray, depth_frame=None, sigma: float = 30.0):
        """
        Generate an enhanced heatmap with 3D real-world coordinates.
        
        Returns heatmap with both 2D pixel coordinates and 3D real-world positions.
        Useful for RL training that requires real-world coordinates.

        Args:
            frame: Input image
            depth_frame: RealSense depth frame
            sigma: Standard deviation of Gaussian blobs (in pixels)

        Returns:
            Tuple of (heatmap_2d, heatmap_3d, detections_3d)
            - heatmap_2d: 2D heatmap (H, W) with pixel coordinates
            - heatmap_3d: 3D-aware heatmap weighted by depth
            - detections_3d: List of detection dicts with 3D coordinates
        """
        if depth_frame is None:
            logger.warning("depth_frame is None, cannot generate enhanced heatmap")
            return None, None, []

        # Initialize heatmap processor with depth frame info
        self.heatmap_processor.initialize_from_realsense(depth_frame)

        # Get standard heatmap
        heatmap_2d = self.get_heatmap(frame, depth_frame, sigma)

        # Get detections
        detections = self.get_detections(frame, depth_frame)

        # Process with heatmap processor
        heatmap_enhanced = self.heatmap_processor.heatmap_to_3d(
            heatmap_2d,
            depth_frame,
            detections=detections,
            extract_peaks=True,
            min_distance=15,
            threshold=0.2
        )

        # Convert detection points to dict for easy use
        detections_3d = [
            {
                "label": det.label,
                "position_3d": tuple(det.position_3d),
                "position_2d": det.position_2d,
                "confidence": det.confidence,
                "depth": det.depth,
            }
            for det in heatmap_enhanced.peak_coordinates
        ]

        return heatmap_enhanced.heatmap_2d, heatmap_enhanced.heatmap_3d, detections_3d

    def get_detection_points_3d(
        self,
        frame: np.ndarray,
        depth_frame=None,
        max_objects: int = 5
    ) -> List[dict]:
        """
        Get detected objects as 3D points in real-world coordinates.
        
        Perfect for RL input! Returns coordinates in meters, not pixels.

        Args:
            frame: Input image
            depth_frame: RealSense depth frame
            max_objects: Maximum number of objects to return

        Returns:
            List of detection dicts:
            [
                {
                    "label": str,
                    "position_3d": (x, y, z),  # in meters
                    "position_2d": (u, v),    # in pixels
                    "confidence": float,
                    "depth": float,           # in meters
                },
                ...
            ]
        """
        if depth_frame is None:
            logger.warning("depth_frame is None, cannot extract 3D points")
            return []

        # Get detections with 3D coordinates
        detections = self.get_detections(frame, depth_frame)

        # Sort by confidence and limit
        detections_sorted = sorted(
            detections,
            key=lambda d: d.get("confidence", 0),
            reverse=True
        )[:max_objects]

        # Convert to cleaner format with explicit 3D coordinates
        result = []
        for det in detections_sorted:
            if det["center_3d"] is not None:
                result.append({
                    "label": det["label"],
                    "position_3d": det["center_3d"],  # (x, y, z) in meters
                    "position_2d": det["center_2d"],  # (u, v) in pixels
                    "confidence": det["confidence"],
                    "depth": det["center_3d"][2],  # Extract z as depth
                })

        return result

    def cleanup(self) -> None:
        """Stop camera thread and clean up resources."""
        self.stop()
        self.pipeline.stop()
        logger.info("RealSense pipeline stopped and resources cleaned up")
