"""
Scene builder for creating complete MuJoCo environments.

Combines terrain generation with simulation settings, lighting, 
and robot loading to create full MJCF scenes.
"""

from typing import Optional, Union, Dict, Type
from xml.etree import ElementTree as ET
from pathlib import Path
import tempfile

from .base import TerrainGenerator
from .boxes import BoxesTerrainGenerator
from ..configs.env_config import (
    SceneConfig, 
    BaseTerrainConfig,
    BoxesTerrainConfig,
    TerrainType,
)


# Registry mapping terrain types to their generators
TERRAIN_GENERATORS: Dict[TerrainType, Type[TerrainGenerator]] = {
    TerrainType.BOXES: BoxesTerrainGenerator,
    # Add more as implemented:
    # TerrainType.STAIRS: StairsTerrainGenerator,
    # TerrainType.ROUGH: RoughTerrainGenerator,
}


class SceneBuilder:
    """
    Builds complete MuJoCo scenes with terrain and optional robot.
    
    Usage:
        config = SceneConfig(terrain=BoxesTerrainConfig())
        builder = SceneBuilder(config)
        
        # Get MJCF as string
        mjcf_string = builder.build()
        
        # Or save to file
        builder.save("scene.xml")
        
        # Or load directly into MuJoCo
        model = builder.load_mujoco()
    """
    
    def __init__(self, config: SceneConfig = None):
        """
        Initialize the scene builder.
        
        Args:
            config: Scene configuration. If None, uses defaults.
        """
        self.config = config or SceneConfig()
        self._terrain_generator: Optional[TerrainGenerator] = None
        
    @property
    def terrain_generator(self) -> TerrainGenerator:
        """Get or create the terrain generator for the current config."""
        if self._terrain_generator is None:
            terrain_config = self.config.terrain
            terrain_type = terrain_config.terrain_type
            
            if terrain_type not in TERRAIN_GENERATORS:
                raise ValueError(
                    f"Unsupported terrain type: {terrain_type}. "
                    f"Available: {list(TERRAIN_GENERATORS.keys())}"
                )
            
            generator_class = TERRAIN_GENERATORS[terrain_type]
            self._terrain_generator = generator_class(terrain_config)
            
        return self._terrain_generator
    
    def build(self) -> str:
        """
        Build the complete MJCF scene as a string.
        
        Returns:
            MJCF XML string
        """
        root = self._create_mjcf_root()
        
        # Add compiler settings
        root.append(self._create_compiler())
        
        # Add simulation options
        root.append(self._create_option())
        
        # Add visual settings
        root.append(self._create_visual())
        
        # Add assets (textures, materials)
        root.append(self._create_asset())
        
        # Generate and add terrain worldbody
        terrain_worldbody = self.terrain_generator.generate()
        
        # Add lighting to worldbody
        self._add_lighting(terrain_worldbody)
        
        # Add default camera
        self._add_camera(terrain_worldbody)
        
        root.append(terrain_worldbody)
        
        # Pretty print
        self._indent(root)
        
        return ET.tostring(root, encoding="unicode")
    
    def build_with_robot(self, robot_xml_path: Union[str, Path]) -> str:
        """
        Build scene with a robot included.
        
        Args:
            robot_xml_path: Path to robot MJCF/URDF file
            
        Returns:
            MJCF XML string with robot included
        """
        # Build base scene
        scene_xml = self.build()
        root = ET.fromstring(scene_xml)
        
        # Find worldbody
        worldbody = root.find("worldbody")
        
        # Add robot as include
        robot_path = Path(robot_xml_path)
        include = ET.SubElement(worldbody, "include")
        include.set("file", str(robot_path))
        
        self._indent(root)
        return ET.tostring(root, encoding="unicode")
    
    def save(self, filepath: Union[str, Path], include_robot: Optional[str] = None) -> Path:
        """
        Save the scene to an XML file.
        
        Args:
            filepath: Output path for the MJCF file
            include_robot: Optional path to robot MJCF to include
            
        Returns:
            Path to saved file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if include_robot:
            xml_content = self.build_with_robot(include_robot)
        else:
            xml_content = self.build()
        
        filepath.write_text(xml_content)
        return filepath
    
    def load_mujoco(self, robot_xml_path: Optional[str] = None):
        """
        Load the scene directly into MuJoCo.
        
        Args:
            robot_xml_path: Optional path to robot MJCF to include
            
        Returns:
            mujoco.MjModel instance
        """
        import mujoco
        
        if robot_xml_path:
            xml_content = self.build_with_robot(robot_xml_path)
        else:
            xml_content = self.build()
        
        return mujoco.MjModel.from_xml_string(xml_content)
    
    def reset(self, seed: Optional[int] = None):
        """
        Reset the scene with a new random seed.
        
        This regenerates the terrain with new randomization.
        """
        self._terrain_generator = None
        if seed is not None:
            self.config.terrain.seed = seed
    
    def _create_mjcf_root(self) -> ET.Element:
        """Create the root mujoco element."""
        root = ET.Element("mujoco")
        root.set("model", "terrain_scene")
        return root
    
    def _create_compiler(self) -> ET.Element:
        """Create compiler settings."""
        compiler = ET.Element("compiler")
        compiler.set("angle", "radian")
        compiler.set("coordinate", "local")
        compiler.set("inertiafromgeom", "true")
        return compiler
    
    def _create_option(self) -> ET.Element:
        """Create simulation options."""
        option = ET.Element("option")
        option.set("timestep", str(self.config.timestep))
        option.set("gravity", " ".join(map(str, self.config.gravity)))
        option.set("iterations", "50")
        option.set("solver", "Newton")
        return option
    
    def _create_visual(self) -> ET.Element:
        """Create visual settings."""
        visual = ET.Element("visual")
        
        # Global settings
        glob = ET.SubElement(visual, "global")
        glob.set("offwidth", "1920")
        glob.set("offheight", "1080")
        
        # Quality settings
        quality = ET.SubElement(visual, "quality")
        quality.set("shadowsize", "4096")
        
        # Headlight
        headlight = ET.SubElement(visual, "headlight")
        headlight.set("ambient", " ".join(map(str, self.config.ambient_light)))
        headlight.set("diffuse", " ".join(map(str, self.config.diffuse_light)))
        
        return visual
    
    def _create_asset(self) -> ET.Element:
        """Create assets (textures, materials)."""
        asset = ET.Element("asset")
        
        if self.config.use_textures:
            # Ground texture
            texture = ET.SubElement(asset, "texture")
            texture.set("name", "grid_texture")
            texture.set("type", "2d")
            texture.set("builtin", "checker")
            texture.set("rgb1", "0.9 0.9 0.9")
            texture.set("rgb2", "0.7 0.7 0.7")
            texture.set("width", "512")
            texture.set("height", "512")
            
            # Ground material
            material = ET.SubElement(asset, "material")
            material.set("name", "grid_material")
            material.set("texture", "grid_texture")
            material.set("texrepeat", "10 10")
            material.set("reflectance", "0.1")
        
        return asset
    
    def _add_lighting(self, worldbody: ET.Element):
        """Add lights to the worldbody."""
        # Main directional light
        light = ET.SubElement(worldbody, "light")
        light.set("name", "main_light")
        light.set("pos", "0 0 5")
        light.set("dir", "0 0 -1")
        light.set("diffuse", " ".join(map(str, self.config.diffuse_light)))
        light.set("specular", "0.3 0.3 0.3")
        light.set("castshadow", str(self.config.shadow_enabled).lower())
    
    def _add_camera(self, worldbody: ET.Element):
        """Add default camera to worldbody."""
        camera = ET.SubElement(worldbody, "camera")
        camera.set("name", "track_cam")
        camera.set("mode", "trackcom")
        camera.set("pos", f"{self.config.camera_distance} 0 2")
        camera.set("xyaxes", "0 1 0 -0.5 0 1")
        
        # Fixed overview camera
        overview = ET.SubElement(worldbody, "camera")
        overview.set("name", "overview")
        overview.set("pos", "5 5 8")
        overview.set("quat", "0.5 0.5 -0.5 -0.5")
    
    def _indent(self, elem: ET.Element, level: int = 0):
        """Add pretty-print indentation to XML."""
        indent = "\n" + "  " * level
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = indent + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = indent
            for child in elem:
                self._indent(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = indent
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = indent


def create_random_scene(
    terrain_type: TerrainType = TerrainType.BOXES,
    difficulty: float = 0.5,
    seed: Optional[int] = None,
    arena_size: tuple = (10.0, 10.0),
) -> SceneBuilder:
    """
    Convenience function to create a randomized scene.
    
    Args:
        terrain_type: Type of terrain to generate
        difficulty: Difficulty level 0.0 (easy) to 1.0 (hard)
        seed: Random seed for reproducibility
        arena_size: Size of the arena in meters
        
    Returns:
        Configured SceneBuilder ready to build
    """
    # Scale parameters based on difficulty
    if terrain_type == TerrainType.BOXES:
        # Harder = taller boxes with more height variation
        min_height = 0.02 + difficulty * 0.05
        max_height = 0.08 + difficulty * 0.20
        
        terrain_config = BoxesTerrainConfig(
            arena_size=arena_size,
            box_height=(min_height, max_height),
            height_noise_scale=0.5 + difficulty * 1.0,
            seed=seed,
        )
    else:
        terrain_config = BaseTerrainConfig(
            terrain_type=terrain_type,
            arena_size=arena_size,
            seed=seed,
        )
    
    scene_config = SceneConfig(terrain=terrain_config)
    return SceneBuilder(scene_config)
