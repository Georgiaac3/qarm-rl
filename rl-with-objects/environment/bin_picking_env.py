"""
TossingBot-inspired environment for bin picking with variable objects.

Multi-modal input:
- RGB image (H x W x 3)
- Depth image (H x W x 1)
- Heatmap image (H x W x 1) showing object density

Multi-modal output (pixel-wise):
- Grasp success probability (H x W x 1): 0-1
- Velocity correction (H x W x 2): [dx, dy] normalized velocity

Episode task: Pick and toss objects out of bin using predicted grasps.
"""

from typing import Dict, Tuple

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .objects import ObjectFactory, PhysicalObject
from .reward_shaper import RewardShaper


class BinPickingEnv(gym.Env):
    """
    Multi-modal RL environment for bin picking with vision-based grasping.

    Observation:
        - 'rgb': (H, W, 3) RGB image normalized [0, 1]
        - 'depth': (H, W, 1) Depth normalized by max range
        - 'heatmap': (H, W, 1) Object density heatmap

    Action:
        - Pixel location (x, y) and velocity correction (vx, vy)
        - Action space: (H, W, 4) where each pixel encodes [grasp_prob, vx, vy, confidence]

    Reward:
        - +10 for successful grasp (distance to object < threshold)
        - +50 for object leaving bin (z > 0.3m or distance > 0.7m)
        - +100 for successful toss and land outside
        - -1 per step (time penalty)
        - -5 for collision with bin/arm
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        image_height: int = 64,
        image_width: int = 64,
        max_steps: int = 200,
        n_objects: int = 5,
        camera_fov: float = 60.0,
        workspace_bounds: Dict[str, Tuple[float, float]] = None,
    ):
        """
        Initialize bin picking environment.

        Args:
            image_height: Height of RGB/Depth/Heatmap images
            image_width: Width of RGB/Depth/Heatmap images
            max_steps: Maximum steps per episode
            n_objects: Number of objects in bin
            camera_fov: Camera field of view in degrees
            workspace_bounds: Dict with keys 'x', 'y', 'z' containing (min, max) tuples
        """
        super().__init__()

        self.image_height = image_height
        self.image_width = image_width
        self.max_steps = max_steps
        self.n_objects = n_objects
        self.camera_fov = camera_fov
        self.step_count = 0

        # Workspace bounds (table coordinates)
        if workspace_bounds is None:
            workspace_bounds = {
                "x": (0.1, 0.7),  # 60cm wide
                "y": (-0.3, 0.3),  # 60cm deep
                "z": (0.0, 0.5),  # 50cm tall (floor at z=0)
            }
        self.workspace_bounds = workspace_bounds

        # Camera setup
        self.camera_position = np.array([0.4, 0.0, 0.5])  # Above center of table
        self.camera_lookat = np.array([0.4, 0.0, 0.0])  # Look at table center

        # Objects in scene
        self.objects: list[PhysicalObject] = []
        self.grasped_object: PhysicalObject = None
        self.sim_time = 0.0

        # Reward shaping
        self.reward_shaper = RewardShaper(curriculum_bonus=1.0)
        self.curriculum_bonus = 1.0  # Updated by training callback

        # Define observation space (multi-modal dict)
        self.observation_space = spaces.Dict(
            {
                "rgb": spaces.Box(
                    low=0, high=1, shape=(image_height, image_width, 3), dtype=np.float32
                ),
                "depth": spaces.Box(
                    low=0, high=1, shape=(image_height, image_width, 1), dtype=np.float32
                ),
                "heatmap": spaces.Box(
                    low=0, high=1, shape=(image_height, image_width, 1), dtype=np.float32
                ),
            }
        )

        # Define action space (pixel-wise predictions)
        # Action: for selected pixel, predict grasp_prob and velocity correction
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, 0.0]),  # [grasp_x, grasp_y, vel_x, vel_y] normalized
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            shape=(4,),  # Actually (H, W, 4) but we discretize to selected pixel
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        """Reset environment and generate new scene."""
        super().reset(seed=seed)

        self.step_count = 0
        self.sim_time = 0.0
        self.grasped_object = None

        # Reset reward shaper
        self.reward_shaper.reset()
        self.reward_shaper.curriculum_bonus = self.curriculum_bonus

        # Generate random bin
        self.objects = ObjectFactory.create_random_bin(n_objects=self.n_objects)

        obs = self._get_observation()
        info = {
            "n_objects": len(self.objects),
            "objects_in_bin": len([obj for obj in self.objects if obj.position[2] >= 0.0]),
        }

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute one step.

        Action: [grasp_pixel_x, grasp_pixel_y, velocity_x, velocity_y]
            - First 2 elements: normalized pixel coordinates of grasp point
            - Last 2 elements: velocity correction (horizontal plane)

        Returns:
            obs, reward, terminated, truncated, info
        """
        # Track previous state for reward calculation
        objects_remaining_before = len([obj for obj in self.objects if obj.position[2] >= 0.0])

        # Parse action
        grasp_px = int(np.clip((action[0] + 1) * self.image_width / 2, 0, self.image_width - 1))
        grasp_py = int(np.clip((action[1] + 1) * self.image_height / 2, 0, self.image_height - 1))
        vel_correction = action[2:4]  # [vx, vy] normalized

        # Simulate grasp attempt at pixel (grasp_px, grasp_py)
        grasp_success, grasped_obj, grasp_distance = self._attempt_grasp(grasp_px, grasp_py)

        # Apply velocity correction if grasp succeeded
        throw_velocity = 0.0
        if grasp_success:
            # Scale velocity correction (max 5 m/s in each direction)
            world_vel = vel_correction * 5.0
            throw_velocity = np.linalg.norm(world_vel)
            self._apply_velocity_correction(grasped_obj, world_vel)

        # Simulate physics for 100ms
        dt = 0.01  # 10ms per substep
        collision_detected = False
        thrown = False
        throw_distance = 0.0

        for _ in range(10):  # 100ms total
            for obj in self.objects:
                if obj.position[2] < -0.05:  # Below table, ignore
                    continue
                
                # If object is grasped, hold position (no gravity)
                if obj.is_grasped:
                    # Keep at current position, only apply thrown velocity
                    obj.velocity[2] *= 0.99  # Don't apply gravity
                else:
                    # Normal physics for non-grasped objects
                    obj.update_physics(dt=dt)

                # Collision detection: simple bin boundaries
                bounds = self.workspace_bounds

                # For grasped objects, allow upward movement (don't clamp z bottom)
                if obj.is_grasped:
                    # Only clamp x, y (horizontal bounds)
                    obj.position[0] = np.clip(obj.position[0], bounds["x"][0], bounds["x"][1])
                    obj.position[1] = np.clip(obj.position[1], bounds["y"][0], bounds["y"][1])
                    # Allow z to go above workspace (throw out)
                    obj.position[2] = np.clip(obj.position[2], bounds["z"][0], np.inf)
                else:
                    # Normal clamping for un-grasped objects
                    obj.position[0] = np.clip(obj.position[0], bounds["x"][0], bounds["x"][1])
                    obj.position[1] = np.clip(obj.position[1], bounds["y"][0], bounds["y"][1])
                    obj.position[2] = np.clip(obj.position[2], bounds["z"][0], bounds["z"][1])

                # Zero out velocity if hitting boundary (energy loss via collision)
                collision_flags = (
                    obj.position[0] == bounds["x"][0]
                    or obj.position[0] == bounds["x"][1]
                    or obj.position[1] == bounds["y"][0]
                    or obj.position[1] == bounds["y"][1]
                    or (obj.position[2] == bounds["z"][0] and not obj.is_grasped)
                )  # Only count z collision if not grasped
                
                if collision_flags:
                    obj.velocity *= 0.5  # Damping on collision
                    # Only count collision if object is not grasped
                    if not obj.is_grasped:
                        collision_detected = True

                # Check if object left bin (thrown out by velocity)
                if grasped_obj and obj == grasped_obj:
                    bin_center = np.array([0.4, 0.0])
                    throw_distance = np.linalg.norm(obj.position[:2] - bin_center)
                    # Bin radius ~ 0.35m, throw success if outside (distance > 0.25 = easier)
                    if obj.position[2] > 0.25 or throw_distance > 0.25:
                        thrown = True
                        # Once thrown, object is no longer grasped
                        obj.is_grasped = False

        self.sim_time += 0.1
        self.step_count += 1

        # Remove objects that left the bin (thrown out)
        if thrown and grasped_obj:
            self.objects.remove(grasped_obj)
            grasped_obj.is_grasped = False
            self.grasped_object = None

        # Use advanced reward shaping
        objects_remaining = len([obj for obj in self.objects if obj.position[2] >= 0.0])
        reward_dict = self.reward_shaper.calculate_reward(
            grasp_success=grasp_success,
            grasp_distance=grasp_distance if grasp_success else np.inf,
            grasped_object=grasped_obj,
            thrown=thrown,
            throw_distance=throw_distance,
            throw_velocity=throw_velocity,
            objects_remaining=objects_remaining,
            step_count=self.step_count,
            collision=collision_detected,
        )

        reward = reward_dict["total"]

        # Check termination
        terminated = False

        if objects_remaining == 0:
            # Already included in reward_shaper bonus
            terminated = True

        truncated = self.step_count >= self.max_steps

        obs = self._get_observation()
        info = {
            "grasp_success": grasp_success,
            "objects_remaining": objects_remaining,
            "sim_time": self.sim_time,
            "reward_breakdown": reward_dict["components"],
            "thrown": thrown,
            "throw_velocity": throw_velocity,
        }

        return obs, reward, terminated, truncated, info

    def _attempt_grasp(self, pixel_x: int, pixel_y: int) -> Tuple[bool, PhysicalObject, float]:
        """
        Attempt grasp at given pixel coordinates.

        Returns:
            (success, grasped_object, grasp_distance)
        """
        # Find closest object using pixel distance (2D, overhead camera)
        # This is more forgiving for 3D objects projected to 2D image
        min_distance = np.inf
        closest_obj = None

        for obj in self.objects:
            # Project object to pixel space
            obj_px, obj_py = self._world_to_pixel(obj.position)
            
            # Distance in pixel space
            pixel_distance = np.sqrt((pixel_x - obj_px) ** 2 + (pixel_y - obj_py) ** 2)
            
            if pixel_distance < min_distance:
                min_distance = pixel_distance
                closest_obj = obj

        # Grasp succeeds if within grasp radius in pixel space
        # Radius in pixels: radius_meters * pixels_per_meter
        # Estimate: ~64 pixels = 0.6m workspace width, so ~107 pixels/meter
        PIXELS_PER_METER = 64 / 0.6  # ~107
        grasp_threshold_px = closest_obj.radius * PIXELS_PER_METER * 6 if closest_obj else 0
        success = min_distance < grasp_threshold_px if closest_obj else False

        # For reward, also compute actual 3D distance for quality bonus
        world_pos = self._pixel_to_world(pixel_x, pixel_y)
        grasp_3d_distance = np.linalg.norm(closest_obj.position - world_pos) if closest_obj else np.inf

        if success:
            closest_obj.is_grasped = True
            closest_obj.grasp_position = world_pos.copy()
            self.grasped_object = closest_obj

        return success, closest_obj, grasp_3d_distance

    def _apply_velocity_correction(self, obj: PhysicalObject, vel_correction: np.ndarray):
        """Apply velocity correction to grasped object (throw impulse).
        
        Args:
            obj: Object to throw
            vel_correction: Pre-scaled velocity in m/s (from action * 5)
        """
        # Add scaled velocity to horizontal plane
        obj.velocity[0] += vel_correction[0]
        obj.velocity[1] += vel_correction[1]
        # Upward component (z boost for throw)
        obj.velocity[2] += 2.0  # 2 m/s upward boost

    def _pixel_to_world(self, pixel_x: int, pixel_y: int) -> np.ndarray:
        """Convert pixel coordinates to world 3D position."""
        # Simplified: project onto table surface (z=0)
        # Assumes orthographic projection from above

        # Normalized pixel coords [-1, 1]
        norm_x = (pixel_x / self.image_width - 0.5) * 2
        norm_y = (pixel_y / self.image_height - 0.5) * 2

        # World coordinates
        bounds = self.workspace_bounds
        world_x = 0.4 + norm_x * (bounds["x"][1] - bounds["x"][0]) / 2
        world_y = 0.0 + norm_y * (bounds["y"][1] - bounds["y"][0]) / 2
        world_z = 0.0  # Table surface

        return np.array([world_x, world_y, world_z])

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Generate multi-modal observation (RGB, Depth, Heatmap)."""
        rgb = self._render_rgb()
        depth = self._render_depth()
        heatmap = self._render_heatmap()

        return {
            "rgb": rgb.astype(np.float32) / 255.0,
            "depth": depth.astype(np.float32),
            "heatmap": heatmap.astype(np.float32),
        }

    def _render_rgb(self) -> np.ndarray:
        """Render RGB image of scene with improved contrast."""
        # Black background for better contrast (was gray 200)
        rgb = np.ones((self.image_height, self.image_width, 3), dtype=np.uint8) * 20  # Dark gray

        for obj in self.objects:
            if obj.position[2] < -0.05:  # Below table
                continue

            # Project to pixel
            px, py = self._world_to_pixel(obj.position)

            if 0 <= px < self.image_width and 0 <= py < self.image_height:
                # Draw circle/square based on shape
                radius_px = int(max(3, obj.radius * 150))  # Increased scale for visibility
                color = tuple(
                    int(c) for c in obj.color[::-1]
                )  # BGR for OpenCV, convert to tuple of ints

                cv2.circle(rgb, (px, py), radius_px, color, -1)
                if obj.is_grasped:
                    cv2.circle(
                        rgb, (px, py), radius_px + 2, (0, 255, 0), 2
                    )  # Green border for grasped

        return rgb

    def _render_depth(self) -> np.ndarray:
        """Render depth image of scene with improved contrast."""
        depth = np.zeros((self.image_height, self.image_width, 1), dtype=np.float32)

        # Render objects as bright regions on dark background
        for obj in self.objects:
            if obj.position[2] < -0.05:
                continue

            px, py = self._world_to_pixel(obj.position)
            radius_px = int(max(3, obj.radius * 150))

            # Normalized depth (0=far, 1=close)
            # Objects at z=0.3m should be very visible
            depth_value = (obj.position[2] - self.workspace_bounds["z"][0]) / (
                self.workspace_bounds["z"][1] - self.workspace_bounds["z"][0]
            )
            depth_value = np.clip(depth_value, 0, 1)

            # Create Gaussian splat
            y, x = np.ogrid[-radius_px : radius_px + 1, -radius_px : radius_px + 1]
            gaussian = np.exp(-(x**2 + y**2) / (2 * (radius_px / 3) ** 2))

            y_start = max(0, py - radius_px)
            y_end = min(self.image_height, py + radius_px + 1)
            x_start = max(0, px - radius_px)
            x_end = min(self.image_width, px + radius_px + 1)

            g_y_start = max(0, radius_px - (py - y_start))
            g_y_end = g_y_start + (y_end - y_start)
            g_x_start = max(0, radius_px - (px - x_start))
            g_x_end = g_x_start + (x_end - x_start)

            g_y_end = min(gaussian.shape[0], g_y_end)
            g_x_end = min(gaussian.shape[1], g_x_end)

            if y_start < y_end and x_start < x_end and g_y_start < g_y_end and g_x_start < g_x_end:
                # Scale gaussian by depth value for better visibility
                depth_splat = gaussian[g_y_start:g_y_end, g_x_start:g_x_end] * depth_value
                depth[y_start:y_end, x_start:x_end, 0] = np.maximum(
                    depth[y_start:y_end, x_start:x_end, 0], depth_splat
                )

        # Normalize and enhance contrast
        depth = np.clip(depth, 0, 1)
        if depth.max() > 0:
            depth = depth / depth.max()  # Normalize to [0, 1]

        return depth

    def _render_heatmap(self) -> np.ndarray:
        """Render object density heatmap with Gaussian splatting (improved contrast)."""
        heatmap = np.zeros((self.image_height, self.image_width, 1), dtype=np.float32)

        for obj in self.objects:
            if obj.position[2] < -0.05:
                continue

            px, py = self._world_to_pixel(obj.position)
            radius_px = int(max(3, obj.radius * 150))  # Increased for visibility

            # Gaussian splat with sharper profile
            y, x = np.ogrid[-radius_px : radius_px + 1, -radius_px : radius_px + 1]
            gaussian = np.exp(-(x**2 + y**2) / (2 * (radius_px / 2.5) ** 2))  # Sharper peak

            y_start = max(0, py - radius_px)
            y_end = min(self.image_height, py + radius_px + 1)
            x_start = max(0, px - radius_px)
            x_end = min(self.image_width, px + radius_px + 1)

            # Clamp gaussian indices properly
            g_y_start = max(0, radius_px - (py - y_start))
            g_y_end = g_y_start + (y_end - y_start)
            g_x_start = max(0, radius_px - (px - x_start))
            g_x_end = g_x_start + (x_end - x_start)

            # Make sure we don't exceed gaussian bounds
            g_y_end = min(gaussian.shape[0], g_y_end)
            g_x_end = min(gaussian.shape[1], g_x_end)

            if y_start < y_end and x_start < x_end and g_y_start < g_y_end and g_x_start < g_x_end:
                heatmap[y_start:y_end, x_start:x_end, 0] += gaussian[
                    g_y_start:g_y_end, g_x_start:g_x_end
                ]

        # Normalize with better contrast
        heatmap = np.clip(heatmap, 0, 1)
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()  # Normalize to [0, 1]

        return heatmap

    def _world_to_pixel(self, world_pos: np.ndarray) -> Tuple[int, int]:
        """Convert world coordinates to pixel coordinates."""
        bounds = self.workspace_bounds

        # Normalize to [-1, 1]
        norm_x = (world_pos[0] - 0.4) / ((bounds["x"][1] - bounds["x"][0]) / 2)
        norm_y = (world_pos[1] - 0.0) / ((bounds["y"][1] - bounds["y"][0]) / 2)

        # To pixels
        px = int((norm_x + 1) * self.image_width / 2)
        py = int((norm_y + 1) * self.image_height / 2)

        return px, py

    def render(self):
        """Rendering placeholder."""
        pass

    def close(self):
        """Close environment."""
        pass
