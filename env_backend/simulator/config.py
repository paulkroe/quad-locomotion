"""Configuration dataclasses for the simulator."""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any
from pathlib import Path


@dataclass
class RenderConfig:
    """Configuration for rendering."""
    
    # Image dimensions
    width: int = 640
    height: int = 480
    
    # Camera settings
    camera_name: Optional[str] = None  # None = free camera
    camera_id: int = -1  # -1 = free camera, or use camera_name
    
    # Free camera settings (when camera_id=-1 and camera_name=None)
    lookat: Tuple[float, float, float] = (0.0, 0.0, 0.3)
    distance: float = 3.0
    azimuth: float = 90.0
    elevation: float = -20.0
    
    # Render options
    depth: bool = False
    segmentation: bool = False
    
    # Visual flags
    flags: Dict[str, bool] = field(default_factory=lambda: {
        "shadow": True,
        "reflection": False,
        "skybox": True,
    })


@dataclass
class SimulatorConfig:
    """Configuration for the MuJoCo simulator."""
    
    # Scene paths
    robot_xml_path: Optional[str] = None
    terrain_xml_path: Optional[str] = None
    scene_xml_path: Optional[str] = None  # Pre-combined scene
    
    # Simulation parameters
    timestep: float = 0.002  # Simulation timestep
    n_substeps: int = 4  # Physics substeps per step() call
    
    # Control settings
    control_timestep: float = 0.02  # Policy control rate (n_substeps * timestep)
    
    # Robot initial state
    init_qpos: Optional[Tuple[float, ...]] = None
    init_qvel: Optional[Tuple[float, ...]] = None
    
    # Randomization
    random_init: bool = True
    init_noise_scale: float = 0.1
    
    # Rendering
    render_config: RenderConfig = field(default_factory=RenderConfig)
    
    # Episode settings
    max_episode_steps: int = 1000
    
    # Reward/observation configuration (will be extended)
    observation_keys: Tuple[str, ...] = (
        "qpos",
        "qvel",
    )
    
    def __post_init__(self):
        """Validate configuration."""
        if self.scene_xml_path is None and self.robot_xml_path is None:
            raise ValueError(
                "Must provide either scene_xml_path or robot_xml_path"
            )
        
        # Compute n_substeps from control_timestep if not manually set
        if self.control_timestep > 0:
            computed_substeps = int(self.control_timestep / self.timestep)
            if computed_substeps != self.n_substeps:
                self.n_substeps = computed_substeps
