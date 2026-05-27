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

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Tuple, Dict
import cv2

from .objects import ObjectFactory, PhysicalObject


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
                'x': (0.1, 0.7),    # 60cm wide
                'y': (-0.3, 0.3),   # 60cm deep
                'z': (0.0, 0.5),    # 50cm tall (floor at z=0)
            }
        self.workspace_bounds = workspace_bounds
        
        # Camera setup
        self.camera_position = np.array([0.4, 0.0, 0.5])  # Above center of table
        self.camera_lookat = np.array([0.4, 0.0, 0.0])      # Look at table center
        
        # Objects in scene
        self.objects: list[PhysicalObject] = []
        self.grasped_object: PhysicalObject = None
        self.sim_time = 0.0
        
        # Define observation space (multi-modal dict)
        self.observation_space = spaces.Dict({
            'rgb': spaces.Box(low=0, high=1, shape=(image_height, image_width, 3), dtype=np.float32),
            'depth': spaces.Box(low=0, high=1, shape=(image_height, image_width, 1), dtype=np.float32),
            'heatmap': spaces.Box(low=0, high=1, shape=(image_height, image_width, 1), dtype=np.float32),
        })
        
        # Define action space (pixel-wise predictions)
        # Action: for selected pixel, predict grasp_prob and velocity correction
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, 0.0]),  # [grasp_x, grasp_y, vel_x, vel_y] normalized
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            shape=(4,),  # Actually (H, W, 4) but we discretize to selected pixel
            dtype=np.float32
        )
    
    def reset(self, seed=None, options=None):
        """Reset environment and generate new scene."""
        super().reset(seed=seed)
        
        self.step_count = 0
        self.sim_time = 0.0
        self.grasped_object = None
        
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
        # Parse action
        grasp_px = int(np.clip((action[0] + 1) * self.image_width / 2, 0, self.image_width - 1))
        grasp_py = int(np.clip((action[1] + 1) * self.image_height / 2, 0, self.image_height - 1))
        vel_correction = action[2:4]  # [vx, vy] normalized
        
        # Simulate grasp attempt at pixel (grasp_px, grasp_py)
        grasp_success, grasped_obj = self._attempt_grasp(grasp_px, grasp_py)
        
        # Apply velocity correction if grasp succeeded
        reward = 0.0
        if grasp_success:
            reward += 10.0  # Grasp bonus
            # Scale velocity correction (max 5 m/s in each direction)
            world_vel = vel_correction * 5.0
            self._apply_velocity_correction(grasped_obj, world_vel)
        
        # Simulate physics for 100ms
        dt = 0.01  # 10ms per substep
        for _ in range(10):  # 100ms total
            for obj in self.objects:
                if obj.position[2] < -0.05:  # Below table, ignore
                    continue
                obj.update_physics(dt=dt)
                
                # Collision detection: simple bin boundaries
                bounds = self.workspace_bounds
                
                # Clamp position to bin bounds (simple box collision)
                obj.position[0] = np.clip(obj.position[0], bounds['x'][0], bounds['x'][1])
                obj.position[1] = np.clip(obj.position[1], bounds['y'][0], bounds['y'][1])
                obj.position[2] = np.clip(obj.position[2], bounds['z'][0], bounds['z'][1])
                
                # Zero out velocity if hitting boundary (energy loss via collision)
                if (obj.position[0] == bounds['x'][0] or obj.position[0] == bounds['x'][1] or
                    obj.position[1] == bounds['y'][0] or obj.position[1] == bounds['y'][1] or
                    obj.position[2] == bounds['z'][0]):  # Hit ground
                    obj.velocity *= 0.5  # Damping on collision
                
                # Check if object left bin (thrown out by velocity)
                if grasped_obj and obj == grasped_obj:
                    if obj.position[2] > 0.3 or np.linalg.norm(obj.position[:2] - [0.4, 0.0]) > 0.7:
                        reward += 50.0  # Left bin bonus
                        grasped_obj = None
        
        self.sim_time += 0.1
        self.step_count += 1
        
        # Time penalty
        reward -= 1.0
        
        # Check termination
        terminated = False
        # Objects are in bin if z >= 0 (floor is at z=0)
        objects_remaining = len([obj for obj in self.objects if obj.position[2] >= 0.0])
        
        if objects_remaining == 0:
            reward += 100.0  # Cleared bin!
            terminated = True
        
        truncated = self.step_count >= self.max_steps
        
        obs = self._get_observation()
        info = {
            "grasp_success": grasp_success,
            "objects_remaining": objects_remaining,
            "sim_time": self.sim_time,
        }
        
        return obs, reward, terminated, truncated, info
    
    def _attempt_grasp(self, pixel_x: int, pixel_y: int) -> Tuple[bool, PhysicalObject]:
        """
        Attempt grasp at given pixel coordinates.
        
        Returns:
            (success, grasped_object)
        """
        # Convert pixel to world coordinates using camera model
        world_pos = self._pixel_to_world(pixel_x, pixel_y)
        
        # Find closest object
        min_distance = np.inf
        closest_obj = None
        
        for obj in self.objects:
            distance = np.linalg.norm(obj.position - world_pos)
            if distance < min_distance:
                min_distance = distance
                closest_obj = obj
        
        # Grasp succeeds if within object radius (with some tolerance)
        success = min_distance < closest_obj.radius * 2 if closest_obj else False
        
        if success:
            closest_obj.is_grasped = True
            closest_obj.grasp_position = world_pos.copy()
            self.grasped_object = closest_obj
        
        return success, closest_obj
    
    def _apply_velocity_correction(self, obj: PhysicalObject, vel_correction: np.ndarray):
        """Apply velocity correction to grasped object (throw impulse)."""
        # Add velocity to horizontal plane
        obj.velocity[0] += vel_correction[0]
        obj.velocity[1] += vel_correction[1]
        # Small upward component
        obj.velocity[2] += 0.5  # 0.5 m/s upward
    
    def _pixel_to_world(self, pixel_x: int, pixel_y: int) -> np.ndarray:
        """Convert pixel coordinates to world 3D position."""
        # Simplified: project onto table surface (z=0)
        # Assumes orthographic projection from above
        
        # Normalized pixel coords [-1, 1]
        norm_x = (pixel_x / self.image_width - 0.5) * 2
        norm_y = (pixel_y / self.image_height - 0.5) * 2
        
        # World coordinates
        bounds = self.workspace_bounds
        world_x = 0.4 + norm_x * (bounds['x'][1] - bounds['x'][0]) / 2
        world_y = 0.0 + norm_y * (bounds['y'][1] - bounds['y'][0]) / 2
        world_z = 0.0  # Table surface
        
        return np.array([world_x, world_y, world_z])
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Generate multi-modal observation (RGB, Depth, Heatmap)."""
        rgb = self._render_rgb()
        depth = self._render_depth()
        heatmap = self._render_heatmap()
        
        return {
            'rgb': rgb.astype(np.float32) / 255.0,
            'depth': depth.astype(np.float32),
            'heatmap': heatmap.astype(np.float32),
        }
    
    def _render_rgb(self) -> np.ndarray:
        """Render RGB image of scene."""
        rgb = np.ones((self.image_height, self.image_width, 3), dtype=np.uint8) * 200  # Gray background
        
        for obj in self.objects:
            if obj.position[2] < -0.05:  # Below table
                continue
            
            # Project to pixel
            px, py = self._world_to_pixel(obj.position)
            
            if 0 <= px < self.image_width and 0 <= py < self.image_height:
                # Draw circle/square based on shape
                radius_px = int(max(1, obj.radius * 100))  # empirical scaling
                color = tuple(int(c) for c in obj.color[::-1])  # BGR for OpenCV, convert to tuple of ints
                
                cv2.circle(rgb, (px, py), radius_px, color, -1)
                if obj.is_grasped:
                    cv2.circle(rgb, (px, py), radius_px + 2, (0, 255, 0), 2)  # Green border for grasped
        
        return rgb
    
    def _render_depth(self) -> np.ndarray:
        """Render depth image of scene using Gaussian splatting."""
        depth = np.zeros((self.image_height, self.image_width, 1), dtype=np.float32)
        
        for obj in self.objects:
            if obj.position[2] < -0.05:
                continue
            
            px, py = self._world_to_pixel(obj.position)
            radius_px = max(2, int(obj.radius * 100))
            
            # Normalized depth (0=far, 1=close)
            depth_norm = (obj.position[2] - self.workspace_bounds['z'][0]) / \
                         (self.workspace_bounds['z'][1] - self.workspace_bounds['z'][0])
            depth_norm = np.clip(depth_norm, 0, 1)
            
            # Gaussian splat (same as heatmap)
            y, x = np.ogrid[-radius_px:radius_px+1, -radius_px:radius_px+1]
            gaussian = np.exp(-(x**2 + y**2) / (2 * (radius_px/2)**2))
            
            y_start = max(0, py - radius_px)
            y_end = min(self.image_height, py + radius_px + 1)
            x_start = max(0, px - radius_px)
            x_end = min(self.image_width, px + radius_px + 1)
            
            g_y_start = max(0, radius_px - (py - y_start))
            g_y_end = g_y_start + (y_end - y_start)
            g_x_start = max(0, radius_px - (px - x_start))
            g_x_end = g_x_start + (x_end - x_start)
            
            # Clamp gaussian indices
            g_y_end = min(gaussian.shape[0], g_y_end)
            g_x_end = min(gaussian.shape[1], g_x_end)
            
            if y_start < y_end and x_start < x_end and g_y_start < g_y_end and g_x_start < g_x_end:
                depth[y_start:y_end, x_start:x_end, 0] += gaussian[g_y_start:g_y_end, g_x_start:g_x_end] * depth_norm
        
        # Normalize
        depth = np.clip(depth, 0, 1)
        return depth
    
    def _render_heatmap(self) -> np.ndarray:
        """Render object density heatmap with Gaussian splatting."""
        heatmap = np.zeros((self.image_height, self.image_width, 1), dtype=np.float32)
        
        for obj in self.objects:
            if obj.position[2] < -0.05:
                continue
            
            px, py = self._world_to_pixel(obj.position)
            radius_px = max(2, int(obj.radius * 100))
            
            # Gaussian splat
            y, x = np.ogrid[-radius_px:radius_px+1, -radius_px:radius_px+1]
            gaussian = np.exp(-(x**2 + y**2) / (2 * (radius_px/2)**2))
            
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
                heatmap[y_start:y_end, x_start:x_end, 0] += gaussian[g_y_start:g_y_end, g_x_start:g_x_end]
        
        # Normalize
        heatmap = np.clip(heatmap, 0, 1)
        return heatmap
    
    def _world_to_pixel(self, world_pos: np.ndarray) -> Tuple[int, int]:
        """Convert world coordinates to pixel coordinates."""
        bounds = self.workspace_bounds
        
        # Normalize to [-1, 1]
        norm_x = (world_pos[0] - 0.4) / ((bounds['x'][1] - bounds['x'][0]) / 2)
        norm_y = (world_pos[1] - 0.0) / ((bounds['y'][1] - bounds['y'][0]) / 2)
        
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

