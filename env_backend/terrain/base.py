"""Base class for terrain generators."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, List
import numpy as np
from xml.etree import ElementTree as ET

from ..configs.env_config import BaseTerrainConfig, TerrainType


class TerrainGenerator(ABC):
    """
    Abstract base class for terrain generators.
    
    Each terrain type (boxes, stairs, rough, etc.) should inherit from this
    and implement the generate() method to create MJCF elements.
    """
    
    def __init__(self, config: BaseTerrainConfig):
        """
        Initialize the terrain generator.
        
        Args:
            config: Configuration dataclass for the terrain type
        """
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        
    @property
    @abstractmethod
    def terrain_type(self) -> TerrainType:
        """Return the terrain type this generator creates."""
        pass
    
    @abstractmethod
    def generate(self) -> ET.Element:
        """
        Generate the terrain as MJCF XML elements.
        
        Returns:
            An ElementTree Element containing the worldbody elements
            for this terrain (geoms, bodies, etc.)
        """
        pass
    
    def _sample_uniform(self, range_tuple: Tuple[float, float]) -> float:
        """Sample uniformly from a (min, max) tuple."""
        return self.rng.uniform(range_tuple[0], range_tuple[1])
    
    def _sample_int(self, range_tuple: Tuple[int, int]) -> int:
        """Sample integer uniformly from a (min, max) tuple (inclusive)."""
        return self.rng.integers(range_tuple[0], range_tuple[1] + 1)
    
    def _lerp_color(
        self, 
        color1: Tuple[float, ...], 
        color2: Tuple[float, ...], 
        t: float
    ) -> Tuple[float, ...]:
        """Linearly interpolate between two colors."""
        return tuple(c1 + t * (c2 - c1) for c1, c2 in zip(color1, color2))
    
    def _is_in_spawn_area(self, x: float, y: float) -> bool:
        """Check if a position is within the spawn area (should be kept clear)."""
        spawn_x, spawn_y = self.config.spawn_area_size
        offset_x, offset_y = self.config.spawn_offset
        
        return (
            abs(x - offset_x) < spawn_x / 2 and
            abs(y - offset_y) < spawn_y / 2
        )
    
    def _create_ground_plane(self) -> ET.Element:
        """Create a ground plane geom."""
        geom = ET.Element("geom")
        geom.set("name", "ground")
        geom.set("type", "plane")
        geom.set("size", f"{self.config.arena_size[0]} {self.config.arena_size[1]} 0.1")
        geom.set("pos", "0 0 0")
        geom.set("rgba", " ".join(map(str, self.config.ground_rgba)))
        geom.set("friction", " ".join(map(str, self.config.ground_friction)))
        return geom
    
    def get_spawn_position(self) -> Tuple[float, float, float]:
        """
        Get a safe spawn position for the robot.
        
        Returns:
            (x, y, z) position tuple
        """
        x = self.config.spawn_offset[0]
        y = self.config.spawn_offset[1]
        z = 0.5  # Default height above ground, robot should handle landing
        return (x, y, z)
    
    def reset(self, seed: Optional[int] = None):
        """Reset the generator with a new seed."""
        if seed is not None:
            self.config.seed = seed
        self.rng = np.random.default_rng(self.config.seed)
