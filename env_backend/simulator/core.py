"""
Core MuJoCo simulator for quadruped locomotion.

Provides a clean API for:
- Loading robot + terrain scenes
- Stepping simulation
- Getting observations
- Applying actions
- Rendering
"""

from typing import Optional, Tuple, Dict, Any, Union, List
from pathlib import Path
import numpy as np
import mujoco
from xml.etree import ElementTree as ET

from .config import SimulatorConfig, RenderConfig
from .renderer import OffscreenRenderer


class MujocoSimulator:
    """
    MuJoCo simulator for quadruped locomotion training.
    
    Designed for RL training with:
    - Gym-like API (reset, step, render)
    - Clean separation of concerns
    - Efficient rendering
    - Easy parallelization
    
    Usage:
        config = SimulatorConfig(
            robot_xml_path="path/to/robot.xml",
            terrain_xml_path="path/to/terrain.xml",
        )
        sim = MujocoSimulator(config)
        
        obs = sim.reset()
        for _ in range(1000):
            action = policy(obs)
            obs, reward, done, truncated, info = sim.step(action)
            if done or truncated:
                obs = sim.reset()
        
        sim.close()
    """
    
    def __init__(self, config: SimulatorConfig):
        """
        Initialize the simulator.
        
        Args:
            config: Simulator configuration
        """
        self.config = config
        self._step_count = 0
        self._episode_count = 0
        self._renderer: Optional[OffscreenRenderer] = None
        
        # Build and load the scene
        self._scene_xml = self._build_scene()
        self._model = mujoco.MjModel.from_xml_string(self._scene_xml)
        self._data = mujoco.MjData(self._model)
        
        # Store initial state for reset
        self._init_qpos = self._data.qpos.copy()
        self._init_qvel = self._data.qvel.copy()
        
        # Override with config if provided
        if config.init_qpos is not None:
            self._init_qpos[:len(config.init_qpos)] = config.init_qpos
        if config.init_qvel is not None:
            self._init_qvel[:len(config.init_qvel)] = config.init_qvel
        
        # Compute action and observation spaces info
        self._setup_spaces()
        
        # Random number generator
        self._rng = np.random.default_rng()
    
    def _build_scene(self) -> str:
        """Build the complete scene XML from robot and terrain."""
        if self.config.scene_xml_path:
            # Load pre-combined scene
            return Path(self.config.scene_xml_path).read_text()
        
        if self.config.terrain_xml_path:
            # Load terrain
            terrain_xml = Path(self.config.terrain_xml_path).read_text()
            terrain_root = ET.fromstring(terrain_xml)
            
            if self.config.robot_xml_path:
                # Load robot and merge into terrain
                robot_xml = Path(self.config.robot_xml_path).read_text()
                robot_root = ET.fromstring(robot_xml)
                
                # Get terrain worldbody
                terrain_worldbody = terrain_root.find("worldbody")
                if terrain_worldbody is None:
                    terrain_worldbody = ET.SubElement(terrain_root, "worldbody")
                
                # Get robot worldbody content and merge
                robot_worldbody = robot_root.find("worldbody")
                if robot_worldbody is not None:
                    for child in robot_worldbody:
                        terrain_worldbody.append(child)
                
                # Merge other robot sections (actuator, default, etc.)
                for section in ['default', 'actuator', 'sensor', 'tendon', 'equality']:
                    robot_section = robot_root.find(section)
                    if robot_section is not None:
                        # Check if terrain already has this section
                        terrain_section = terrain_root.find(section)
                        if terrain_section is None:
                            terrain_root.append(robot_section)
                        else:
                            # Merge children
                            for child in robot_section:
                                terrain_section.append(child)
                
                # Merge asset section
                robot_asset = robot_root.find("asset")
                if robot_asset is not None:
                    # Resolve any relative mesh/asset file paths so that the
                    # combined scene XML can be loaded via from_xml_string()
                    # (when the robot XML uses a local meshdir like "assets").
                    robot_compiler = robot_root.find("compiler")
                    robot_meshdir = None
                    if robot_compiler is not None:
                        robot_meshdir = robot_compiler.get("meshdir")

                    asset_base = Path(self.config.robot_xml_path).parent
                    if robot_meshdir:
                        asset_base = asset_base / robot_meshdir

                    # Update mesh file attributes to absolute paths
                    for mesh in robot_asset.findall("mesh"):
                        file_attr = mesh.get("file")
                        if file_attr and not Path(file_attr).is_absolute():
                            mesh.set("file", str((asset_base / file_attr).resolve()))

                    terrain_asset = terrain_root.find("asset")
                    if terrain_asset is None:
                        terrain_root.insert(0, robot_asset)
                    else:
                        for child in robot_asset:
                            terrain_asset.append(child)
            
            return ET.tostring(terrain_root, encoding="unicode")
        
        # Just robot, no terrain - still need to resolve mesh paths
        robot_path = Path(self.config.robot_xml_path)
        robot_xml = robot_path.read_text()
        robot_root = ET.fromstring(robot_xml)
        
        # Resolve mesh paths for standalone robot
        robot_compiler = robot_root.find("compiler")
        robot_meshdir = None
        if robot_compiler is not None:
            robot_meshdir = robot_compiler.get("meshdir")
        
        asset_base = robot_path.parent
        if robot_meshdir:
            asset_base = asset_base / robot_meshdir
        
        robot_asset = robot_root.find("asset")
        if robot_asset is not None:
            for mesh in robot_asset.findall("mesh"):
                file_attr = mesh.get("file")
                if file_attr and not Path(file_attr).is_absolute():
                    mesh.set("file", str((asset_base / file_attr).resolve()))
        
        # Add a ground plane if running without terrain
        worldbody = robot_root.find("worldbody")
        if worldbody is not None:
            # Check if there's already a floor/ground
            has_ground = any(
                geom.get("name") in ["floor", "ground"] or geom.get("type") == "plane"
                for geom in worldbody.findall("geom")
            )
            if not has_ground:
                # Add a simple ground plane
                ground = ET.Element("geom")
                ground.set("name", "floor")
                ground.set("type", "plane")
                ground.set("size", "100 100 0.1")
                ground.set("rgba", "0.8 0.8 0.8 1")
                ground.set("contype", "1")
                ground.set("conaffinity", "1")
                worldbody.insert(0, ground)
                
                # Add light if missing
                has_light = len(worldbody.findall("light")) > 0
                if not has_light:
                    light = ET.Element("light")
                    light.set("name", "spotlight")
                    light.set("pos", "0 -2 3")
                    light.set("dir", "0 0.5 -1")
                    light.set("diffuse", "0.8 0.8 0.8")
                    light.set("specular", "0.3 0.3 0.3")
                    worldbody.insert(0, light)
        
        return ET.tostring(robot_root, encoding="unicode")
    
    def _setup_spaces(self):
        """Set up action and observation space information."""
        # Action space (actuators)
        self.n_actuators = self._model.nu
        self.action_low = self._model.actuator_ctrlrange[:, 0].copy()
        self.action_high = self._model.actuator_ctrlrange[:, 1].copy()
        
        # Check for unbounded actuators
        if np.any(np.isinf(self.action_low)) or np.any(np.isinf(self.action_high)):
            # Default to [-1, 1] for unbounded
            self.action_low = np.where(
                np.isinf(self.action_low), -1.0, self.action_low
            )
            self.action_high = np.where(
                np.isinf(self.action_high), 1.0, self.action_high
            )
        
        # Observation space (computed dynamically based on config)
        self._obs_dim = self._compute_obs_dim()
    
    def _compute_obs_dim(self) -> int:
        """Compute observation dimension based on config."""
        dim = 0
        for key in self.config.observation_keys:
            if key == "qpos":
                dim += self._model.nq
            elif key == "qvel":
                dim += self._model.nv
            elif key == "cfrc_ext":
                dim += self._model.nbody * 6
            # Add more observation types as needed
        return dim
    
    @property
    def model(self) -> mujoco.MjModel:
        """Get the MuJoCo model."""
        return self._model
    
    @property
    def data(self) -> mujoco.MjData:
        """Get the MuJoCo data."""
        return self._data
    
    @property
    def dt(self) -> float:
        """Control timestep (time between step() calls)."""
        return self._model.opt.timestep * self.config.n_substeps
    
    @property
    def sim_dt(self) -> float:
        """Simulation timestep."""
        return self._model.opt.timestep
    
    @property
    def time(self) -> float:
        """Current simulation time."""
        return self._data.time
    
    def seed(self, seed: Optional[int] = None):
        """Set random seed."""
        self._rng = np.random.default_rng(seed)
        return seed
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the simulation to initial state.
        
        Args:
            seed: Random seed for this episode
            options: Additional reset options
            
        Returns:
            observation: Initial observation
            info: Additional information
        """
        if seed is not None:
            self.seed(seed)
        
        self._step_count = 0
        self._episode_count += 1
        
        # Reset simulation state
        mujoco.mj_resetData(self._model, self._data)
        
        # Set initial position/velocity
        self._data.qpos[:] = self._init_qpos
        self._data.qvel[:] = self._init_qvel
        
        # Add noise if configured
        if self.config.random_init:
            noise_scale = self.config.init_noise_scale
            self._data.qpos[:] += self._rng.uniform(
                -noise_scale, noise_scale, self._model.nq
            )
            self._data.qvel[:] += self._rng.uniform(
                -noise_scale * 0.1, noise_scale * 0.1, self._model.nv
            )
        
        # Forward kinematics
        mujoco.mj_forward(self._model, self._data)
        
        obs = self._get_obs()
        info = self._get_info()
        
        return obs, info
    
    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Step the simulation.
        
        Args:
            action: Action to apply (size = n_actuators)
            
        Returns:
            observation: Next observation
            reward: Reward for this step
            terminated: Whether episode ended due to terminal condition
            truncated: Whether episode ended due to time limit
            info: Additional information
        """
        # Clip action to valid range
        action = np.clip(action, self.action_low, self.action_high)
        
        # Apply action
        self._data.ctrl[:] = action
        
        # Step simulation
        for _ in range(self.config.n_substeps):
            mujoco.mj_step(self._model, self._data)
        
        self._step_count += 1
        
        # Get outputs
        obs = self._get_obs()
        reward = self._compute_reward(action)
        terminated = self._check_termination()
        truncated = self._step_count >= self.config.max_episode_steps
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def _get_obs(self) -> np.ndarray:
        """Construct observation vector."""
        obs_parts = []
        
        for key in self.config.observation_keys:
            if key == "qpos":
                obs_parts.append(self._data.qpos.copy())
            elif key == "qvel":
                obs_parts.append(self._data.qvel.copy())
            elif key == "cfrc_ext":
                obs_parts.append(self._data.cfrc_ext.flatten())
            # Add more observation types as needed
        
        if obs_parts:
            return np.concatenate(obs_parts)
        return np.array([])
    
    def _compute_reward(self, action: np.ndarray) -> float:
        """
        Compute reward for the current state.
        
        Override this method in subclasses for custom reward functions.
        Default: simple forward velocity reward.
        """
        # Base reward: forward velocity (x direction)
        forward_vel = self._data.qvel[0] if self._model.nv > 0 else 0.0
        
        # Survival bonus
        alive_bonus = 0.1
        
        # Action penalty (encourage smooth actions)
        action_cost = 0.001 * np.sum(np.square(action))
        
        return forward_vel + alive_bonus - action_cost
    
    def _check_termination(self) -> bool:
        """
        Check if episode should terminate.
        
        Override in subclasses for custom termination conditions.
        Default: terminate if robot falls (torso too low).
        """
        # Get torso height if available
        try:
            torso_id = mujoco.mj_name2id(
                self._model, mujoco.mjtObj.mjOBJ_BODY, "torso"
            )
            if torso_id != -1:
                torso_height = self._data.xpos[torso_id, 2]
                if torso_height < 0.1:  # Fallen
                    return True
        except:
            pass
        
        return False
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about the current state."""
        return {
            "step": self._step_count,
            "time": self.time,
            "episode": self._episode_count,
        }
    
    def render(
        self,
        camera_name: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> np.ndarray:
        """
        Render the current scene.
        
        Args:
            camera_name: Camera to render from (None for default)
            width: Override render width
            height: Override render height
            
        Returns:
            RGB image as (H, W, 3) uint8 array
        """
        # Lazy initialize renderer
        if self._renderer is None:
            render_config = self.config.render_config
            if width:
                render_config.width = width
            if height:
                render_config.height = height
            self._renderer = OffscreenRenderer(self._model, render_config)
        elif width or height:
            self._renderer.resize(
                width or self._renderer.width,
                height or self._renderer.height,
            )
        
        return self._renderer.render(self._data, camera_name=camera_name)
    
    def render_depth(self, camera_name: Optional[str] = None) -> np.ndarray:
        """Render depth image."""
        if self._renderer is None:
            self._renderer = OffscreenRenderer(
                self._model, self.config.render_config
            )
        rgb, depth = self._renderer.render(
            self._data, camera_name=camera_name, depth=True
        )
        return depth
    
    def get_body_pos(self, body_name: str) -> np.ndarray:
        """Get position of a body by name."""
        body_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, body_name
        )
        if body_id == -1:
            raise ValueError(f"Body '{body_name}' not found")
        return self._data.xpos[body_id].copy()
    
    def get_body_vel(self, body_name: str) -> np.ndarray:
        """Get velocity of a body by name."""
        body_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_BODY, body_name
        )
        if body_id == -1:
            raise ValueError(f"Body '{body_name}' not found")
        return self._data.cvel[body_id].copy()
    
    def get_sensor(self, sensor_name: str) -> np.ndarray:
        """Get sensor reading by name."""
        sensor_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name
        )
        if sensor_id == -1:
            raise ValueError(f"Sensor '{sensor_name}' not found")
        
        adr = self._model.sensor_adr[sensor_id]
        dim = self._model.sensor_dim[sensor_id]
        return self._data.sensordata[adr:adr + dim].copy()
    
    def get_contact_forces(self) -> List[Dict[str, Any]]:
        """Get all contact forces."""
        contacts = []
        for i in range(self._data.ncon):
            contact = self._data.contact[i]
            geom1 = mujoco.mj_id2name(
                self._model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1
            )
            geom2 = mujoco.mj_id2name(
                self._model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2
            )
            
            # Get contact force
            force = np.zeros(6)
            mujoco.mj_contactForce(self._model, self._data, i, force)
            
            contacts.append({
                "geom1": geom1,
                "geom2": geom2,
                "pos": contact.pos.copy(),
                "force": force[:3],  # Normal force
                "torque": force[3:],  # Friction torque
            })
        return contacts
    
    def set_state(self, qpos: np.ndarray, qvel: np.ndarray):
        """Set simulation state directly."""
        self._data.qpos[:] = qpos
        self._data.qvel[:] = qvel
        mujoco.mj_forward(self._model, self._data)
    
    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current simulation state."""
        return self._data.qpos.copy(), self._data.qvel.copy()
    
    def close(self):
        """Clean up resources."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
    
    def __del__(self):
        self.close()
