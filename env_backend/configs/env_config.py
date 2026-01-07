from dataclasses import dataclass, field
from typing import Tuple, List, Optional
from enum import Enum


class TerrainType(Enum):
    """Supported terrain types for environment generation."""
    FLAT = "flat"
    BOXES = "boxes"
    STAIRS = "stairs"
    STEPPING_STONES = "stepping_stones"
    ROUGH = "rough"
    SLOPE = "slope"
    GAPS = "gaps"


@dataclass
class BaseTerrainConfig:
    """Base configuration for all terrain types."""
    terrain_type: TerrainType = TerrainType.FLAT
    
    # Arena dimensions
    arena_size: Tuple[float, float] = (10.0, 10.0)  # (x, y) in meters
    
    # Spawn area (flat region for robot spawn)
    spawn_area_size: Tuple[float, float] = (1.5, 1.5)
    spawn_offset: Tuple[float, float] = (0.0, 0.0)  # Offset from center
    
    # Ground properties
    ground_friction: Tuple[float, float, float] = (1.0, 0.005, 0.0001)
    ground_rgba: Tuple[float, float, float, float] = (0.8, 0.8, 0.8, 1.0)
    
    # Random seed for reproducibility
    seed: Optional[int] = None


@dataclass
class BoxesTerrainConfig(BaseTerrainConfig):
    """
    Configuration for box field terrain (like the attached image).
    Creates a grid of boxes with randomized heights.
    """
    terrain_type: TerrainType = TerrainType.BOXES
    
    # Box size ranges (min, max) in meters
    box_size_x: Tuple[float, float] = (0.3, 0.6)
    box_size_y: Tuple[float, float] = (0.3, 0.6)
    box_height: Tuple[float, float] = (0.02, 0.15)
    
    # Grid configuration
    grid_spacing: float = 0.05  # Gap between boxes
    
    # Height variation pattern
    height_noise_scale: float = 1.0  # Perlin-like noise scale
    
    # Visual properties
    box_rgba_range: Tuple[Tuple[float, ...], Tuple[float, ...]] = (
        (0.2, 0.2, 0.2, 1.0),  # Dark
        (0.9, 0.9, 0.9, 1.0),  # Light
    )


@dataclass
class StairsTerrainConfig(BaseTerrainConfig):
    """Configuration for staircase terrain."""
    terrain_type: TerrainType = TerrainType.STAIRS
    
    # Stair dimensions
    step_width: Tuple[float, float] = (0.3, 0.5)
    step_height: Tuple[float, float] = (0.05, 0.15)
    step_depth: Tuple[float, float] = (0.25, 0.35)
    
    # Number of steps
    num_steps: Tuple[int, int] = (5, 15)
    
    # Direction (up/down pattern)
    bidirectional: bool = True  # Goes up then down


@dataclass
class SteppingStonesConfig(BaseTerrainConfig):
    """Configuration for stepping stones terrain with gaps."""
    terrain_type: TerrainType = TerrainType.STEPPING_STONES
    
    # Stone dimensions
    stone_size: Tuple[float, float] = (0.2, 0.4)
    stone_height: Tuple[float, float] = (0.05, 0.2)
    
    # Placement
    gap_size: Tuple[float, float] = (0.1, 0.3)
    placement_noise: float = 0.05  # Random offset from grid
    
    # Density (fraction of grid cells that have stones)
    density: float = 0.7


@dataclass
class RoughTerrainConfig(BaseTerrainConfig):
    """Configuration for rough/uneven terrain using heightfield."""
    terrain_type: TerrainType = TerrainType.ROUGH
    
    # Heightfield resolution
    resolution: Tuple[int, int] = (100, 100)
    
    # Height variation
    height_range: Tuple[float, float] = (-0.05, 0.05)
    
    # Noise parameters (for Perlin-like generation)
    noise_frequency: float = 0.1
    noise_octaves: int = 4
    noise_persistence: float = 0.5


@dataclass
class SlopeTerrainConfig(BaseTerrainConfig):
    """Configuration for sloped terrain."""
    terrain_type: TerrainType = TerrainType.SLOPE
    
    # Slope angle in degrees
    angle: Tuple[float, float] = (5.0, 15.0)
    
    # Direction (0 = +x, 90 = +y, etc.)
    direction: Tuple[float, float] = (0.0, 360.0)
    
    # Add roughness on top of slope
    add_roughness: bool = False
    roughness_amplitude: float = 0.02


@dataclass
class GapsTerrainConfig(BaseTerrainConfig):
    """Configuration for terrain with gaps/chasms."""
    terrain_type: TerrainType = TerrainType.GAPS
    
    # Platform dimensions
    platform_length: Tuple[float, float] = (0.5, 1.5)
    platform_width: float = 3.0
    
    # Gap dimensions
    gap_length: Tuple[float, float] = (0.1, 0.4)
    
    # Number of gaps
    num_gaps: Tuple[int, int] = (3, 8)


@dataclass 
class CurriculumConfig:
    """Configuration for curriculum learning over terrains."""
    
    # Start with easier terrains
    initial_difficulty: float = 0.0  # 0.0 = easiest, 1.0 = hardest
    
    # Difficulty progression
    difficulty_increment: float = 0.1
    
    # Terrain mix at different difficulty levels
    terrain_weights: dict = field(default_factory=lambda: {
        TerrainType.FLAT: 1.0,
        TerrainType.BOXES: 0.0,
        TerrainType.STAIRS: 0.0,
        TerrainType.STEPPING_STONES: 0.0,
        TerrainType.ROUGH: 0.0,
    })


@dataclass
class SceneConfig:
    """Complete scene configuration including terrain and simulation parameters."""
    
    # Terrain configuration
    terrain: BaseTerrainConfig = field(default_factory=BaseTerrainConfig)
    
    # Simulation parameters
    timestep: float = 0.002
    gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81)
    
    # Visual settings
    use_textures: bool = True
    shadow_enabled: bool = True
    
    # Lighting
    ambient_light: Tuple[float, float, float] = (0.4, 0.4, 0.4)
    diffuse_light: Tuple[float, float, float] = (0.6, 0.6, 0.6)
    
    # Camera defaults
    camera_distance: float = 3.0
    camera_elevation: float = -20.0
    camera_azimuth: float = 90.0
