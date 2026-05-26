"""
Object definitions for robotic bin tossing task.

Objects have variable physical properties:
- Size (radius 5-25mm)
- Mass (10-200g)
- Shape (sphere, cube, cylinder)
- Color (RGB) for heatmap generation
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import List


class ObjectShape(Enum):
    """Object shape types."""
    SPHERE = "sphere"
    CUBE = "cube"
    CYLINDER = "cylinder"


@dataclass
class PhysicalObject:
    """Physical properties of an object in the scene."""
    
    # Identity
    object_id: int
    shape: ObjectShape
    
    # Position & orientation
    position: np.ndarray  # [x, y, z] in meters
    orientation: np.ndarray  # [qx, qy, qz, qw] quaternion
    velocity: np.ndarray  # [vx, vy, vz] m/s
    angular_velocity: np.ndarray  # [wx, wy, wz] rad/s
    
    # Physical properties
    radius: float  # meters (or half-width for cubes)
    mass: float  # kg
    density: float  # kg/m³ (for rendering heatmap)
    
    # Visual properties (for heatmap)
    color: np.ndarray  # [R, G, B] 0-255
    
    # State tracking
    is_grasped: bool = False
    grasp_position: np.ndarray = None  # Grasp point in 3D
    
    def __post_init__(self):
        """Validate and initialize object."""
        assert 0.005 <= self.radius <= 0.025, f"Radius must be 5-25mm, got {self.radius*1000}mm"
        assert 0.01 <= self.mass <= 0.2, f"Mass must be 10-200g, got {self.mass*1000}g"
        assert len(self.position) == 3
        assert len(self.velocity) == 3
        
    def get_volume(self) -> float:
        """Get object volume in m³."""
        if self.shape == ObjectShape.SPHERE:
            return (4/3) * np.pi * (self.radius ** 3)
        elif self.shape == ObjectShape.CUBE:
            side = 2 * self.radius
            return side ** 3
        else:  # CYLINDER
            return np.pi * (self.radius ** 2) * (2 * self.radius)
    
    def get_moment_of_inertia(self) -> np.ndarray:
        """Get moment of inertia tensor (diagonal)."""
        if self.shape == ObjectShape.SPHERE:
            I = (2/5) * self.mass * (self.radius ** 2)
            return np.array([I, I, I])
        elif self.shape == ObjectShape.CUBE:
            side = 2 * self.radius
            I = (1/6) * self.mass * (side ** 2)
            return np.array([I, I, I])
        else:  # CYLINDER
            Iz = (1/2) * self.mass * (self.radius ** 2)
            Ixy = (1/4) * self.mass * (self.radius ** 2) + (1/12) * self.mass * (2*self.radius) ** 2
            return np.array([Ixy, Ixy, Iz])
    
    def update_physics(self, dt: float, gravity: float = 9.81):
        """Update object physics with simple Euler integration."""
        # Gravity
        self.velocity[2] -= gravity * dt
        
        # Update position
        self.position += self.velocity * dt
        # Note: For simple simulation, we skip quaternion update
        # In a full implementation, angular_velocity (3D) would convert to quaternion delta
        # self.orientation += convert_angular_velocity_to_quaternion(self.angular_velocity) * dt
        
        # Simple damping
        self.velocity *= 0.99  # Air resistance
        self.angular_velocity *= 0.98


class ObjectFactory:
    """Factory to generate objects with realistic properties."""
    
    # Object templates: (name, shape, radius[m], mass[kg], color[RGB])
    TEMPLATES = {
        "small_plastic": (ObjectShape.SPHERE, 0.008, 0.01, np.array([255, 100, 100])),  # 8mm, 10g, red
        "small_metal": (ObjectShape.SPHERE, 0.008, 0.05, np.array([200, 200, 200])),     # 8mm, 50g, gray
        "medium_plastic": (ObjectShape.CUBE, 0.012, 0.03, np.array([100, 255, 100])),   # 12mm, 30g, green
        "medium_metal": (ObjectShape.SPHERE, 0.012, 0.08, np.array([255, 255, 100])),   # 12mm, 80g, yellow
        "large_plastic": (ObjectShape.CYLINDER, 0.018, 0.06, np.array([100, 100, 255])), # 18mm, 60g, blue
        "large_metal": (ObjectShape.SPHERE, 0.018, 0.15, np.array([128, 128, 128])),    # 18mm, 150g, dark gray
        "heavy": (ObjectShape.CUBE, 0.015, 0.2, np.array([255, 128, 0])),               # 15mm, 200g, orange
    }
    
    @staticmethod
    def create(
        object_type: str,
        object_id: int,
        position: np.ndarray = None,
        orientation: np.ndarray = None,
    ) -> PhysicalObject:
        """
        Create an object from a template.
        
        Args:
            object_type: Key from TEMPLATES dict
            object_id: Unique identifier
            position: [x, y, z] starting position (or random)
            orientation: [qx, qy, qz, qw] quaternion (or identity)
        
        Returns:
            PhysicalObject instance
        """
        if object_type not in ObjectFactory.TEMPLATES:
            raise ValueError(f"Unknown object type: {object_type}. Available: {list(ObjectFactory.TEMPLATES.keys())}")
        
        shape, radius, mass, color = ObjectFactory.TEMPLATES[object_type]
        
        if position is None:
            # Random position in table workspace (0.2-0.6m, -0.3-0.3m, 0.3m height)
            # Start objects high enough so they persist through episode
            position = np.array([
                np.random.uniform(0.2, 0.6),
                np.random.uniform(-0.3, 0.3),
                0.3  # Much higher initial position
            ], dtype=np.float64)
        
        if orientation is None:
            # Random rotation (identity + small random)
            orientation = np.array([0, 0, 0, 1], dtype=np.float64)  # [qx, qy, qz, qw]
        
        return PhysicalObject(
            object_id=object_id,
            shape=shape,
            position=position,
            orientation=orientation,
            velocity=np.zeros(3),
            angular_velocity=np.zeros(3),
            radius=radius,
            mass=mass,
            density=mass / ObjectFactory.TEMPLATES[object_type][1],  # mass / volume
            color=color.astype(np.uint8),
        )
    
    @staticmethod
    def create_random_bin(n_objects: int = 5) -> List[PhysicalObject]:
        """
        Create a bin with random objects.
        
        Args:
            n_objects: Number of objects to generate
        
        Returns:
            List of PhysicalObject instances
        """
        object_types = list(ObjectFactory.TEMPLATES.keys())
        objects = []
        
        for i in range(n_objects):
            obj_type = np.random.choice(object_types)
            obj = ObjectFactory.create(
                object_type=obj_type,
                object_id=i,
            )
            objects.append(obj)
        
        return objects


if __name__ == "__main__":
    # Test object creation
    print("Testing ObjectFactory...")
    
    # Create single object
    obj = ObjectFactory.create("small_plastic", object_id=0)
    print(f"\nObject 0: {obj.shape.value} - {obj.mass*1000:.0f}g, radius={obj.radius*1000:.1f}mm")
    print(f"  Volume: {obj.get_volume()*1e6:.2f} mm³")
    print(f"  Color: {obj.color}")
    
    # Create random bin
    print("\nRandom bin with 5 objects:")
    bin_objects = ObjectFactory.create_random_bin(n_objects=5)
    for obj in bin_objects:
        print(f"  {obj.object_id}: {obj.shape.value:8s} - {obj.mass*1000:6.0f}g, {obj.radius*1000:5.1f}mm radius")
