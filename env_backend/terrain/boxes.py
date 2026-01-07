"""Box field terrain generator - creates a grid of boxes with varying heights."""

from typing import Tuple
import numpy as np
from xml.etree import ElementTree as ET

from .base import TerrainGenerator
from ..configs.env_config import BoxesTerrainConfig, TerrainType


class BoxesTerrainGenerator(TerrainGenerator):
    """
    Generates a box field terrain similar to the one used in locomotion research.
    
    Creates a grid of boxes with randomized:
    - Heights (within configured range)
    - Sizes (within configured range)  
    - Colors (grayscale variation for visual diversity)
    
    The spawn area is kept clear for robot initialization.
    """
    
    def __init__(self, config: BoxesTerrainConfig = None):
        if config is None:
            config = BoxesTerrainConfig()
        super().__init__(config)
        self.config: BoxesTerrainConfig = config
        
    @property
    def terrain_type(self) -> TerrainType:
        return TerrainType.BOXES
    
    def generate(self) -> ET.Element:
        """
        Generate the box field terrain.
        
        Returns:
            ElementTree Element containing all box geoms
        """
        worldbody = ET.Element("worldbody")
        
        # Add ground plane
        ground = self._create_ground_plane()
        worldbody.append(ground)
        
        # Calculate grid parameters
        arena_x, arena_y = self.config.arena_size
        
        # Average box size for grid calculation
        avg_box_x = (self.config.box_size_x[0] + self.config.box_size_x[1]) / 2
        avg_box_y = (self.config.box_size_y[0] + self.config.box_size_y[1]) / 2
        
        cell_x = avg_box_x + self.config.grid_spacing
        cell_y = avg_box_y + self.config.grid_spacing
        
        # Number of boxes in each direction
        n_boxes_x = int(arena_x / cell_x)
        n_boxes_y = int(arena_y / cell_y)
        
        # Generate height map with smooth variations
        height_map = self._generate_height_map(n_boxes_x, n_boxes_y)
        
        # Create boxes
        box_id = 0
        for i in range(n_boxes_x):
            for j in range(n_boxes_y):
                # Calculate box center position
                x = (i - n_boxes_x / 2 + 0.5) * cell_x
                y = (j - n_boxes_y / 2 + 0.5) * cell_y
                
                # Skip spawn area
                if self._is_in_spawn_area(x, y):
                    continue
                
                # Sample box dimensions
                size_x = self._sample_uniform(self.config.box_size_x) / 2  # Half-size for MuJoCo
                size_y = self._sample_uniform(self.config.box_size_y) / 2
                
                # Get height from height map and add some noise
                base_height = height_map[i, j]
                height_noise = self._sample_uniform((-0.02, 0.02))
                height = np.clip(
                    base_height + height_noise,
                    self.config.box_height[0],
                    self.config.box_height[1]
                )
                size_z = height / 2  # Half-size for MuJoCo
                
                # Position (z is at half-height so box sits on ground)
                pos_z = size_z
                
                # Color based on height (darker = lower, lighter = higher)
                height_normalized = (height - self.config.box_height[0]) / (
                    self.config.box_height[1] - self.config.box_height[0] + 1e-6
                )
                color = self._lerp_color(
                    self.config.box_rgba_range[0],
                    self.config.box_rgba_range[1],
                    height_normalized
                )
                
                # Create box geom
                box = ET.SubElement(worldbody, "geom")
                box.set("name", f"box_{box_id}")
                box.set("type", "box")
                box.set("size", f"{size_x:.4f} {size_y:.4f} {size_z:.4f}")
                box.set("pos", f"{x:.4f} {y:.4f} {pos_z:.4f}")
                box.set("rgba", " ".join(f"{c:.3f}" for c in color))
                box.set("friction", " ".join(map(str, self.config.ground_friction)))
                
                box_id += 1
        
        return worldbody
    
    def _generate_height_map(self, nx: int, ny: int) -> np.ndarray:
        """
        Generate a smooth height map using multiple octaves of noise.
        
        This creates more natural-looking terrain with both large-scale
        and small-scale height variations.
        """
        height_map = np.zeros((nx, ny))
        
        # Simple multi-octave noise (Perlin-like effect without dependencies)
        for octave in range(3):
            freq = 2 ** octave * self.config.height_noise_scale
            amplitude = 0.5 ** octave
            
            # Generate smooth noise at this frequency
            noise = self._generate_smooth_noise(nx, ny, freq)
            height_map += amplitude * noise
        
        # Normalize to [0, 1] range
        height_map = (height_map - height_map.min()) / (height_map.max() - height_map.min() + 1e-6)
        
        # Scale to height range
        min_h, max_h = self.config.box_height
        height_map = min_h + height_map * (max_h - min_h)
        
        return height_map
    
    def _generate_smooth_noise(self, nx: int, ny: int, frequency: float) -> np.ndarray:
        """Generate smooth noise using interpolated random values."""
        # Create a smaller grid of random values
        grid_size = max(2, int(max(nx, ny) / frequency))
        small_grid = self.rng.random((grid_size, grid_size))
        
        # Interpolate to full size
        x = np.linspace(0, grid_size - 1, nx)
        y = np.linspace(0, grid_size - 1, ny)
        
        # Bilinear interpolation
        x0 = np.floor(x).astype(int)
        x1 = np.minimum(x0 + 1, grid_size - 1)
        y0 = np.floor(y).astype(int)
        y1 = np.minimum(y0 + 1, grid_size - 1)
        
        xd = x - x0
        yd = y - y0
        
        result = np.zeros((nx, ny))
        for i in range(nx):
            for j in range(ny):
                c00 = small_grid[x0[i], y0[j]]
                c10 = small_grid[x1[i], y0[j]]
                c01 = small_grid[x0[i], y1[j]]
                c11 = small_grid[x1[i], y1[j]]
                
                c0 = c00 * (1 - xd[i]) + c10 * xd[i]
                c1 = c01 * (1 - xd[i]) + c11 * xd[i]
                result[i, j] = c0 * (1 - yd[j]) + c1 * yd[j]
        
        return result
