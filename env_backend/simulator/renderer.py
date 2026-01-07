"""Offscreen renderer for MuJoCo simulations."""

from typing import Optional, Tuple, Dict, Any, Union
import numpy as np
import mujoco

from .config import RenderConfig


class OffscreenRenderer:
    """
    Offscreen renderer for MuJoCo simulations.
    
    Provides efficient GPU-accelerated rendering without requiring a display.
    Designed for:
    - Training with visual observations
    - Recording videos
    - Streaming to browser
    
    Usage:
        renderer = OffscreenRenderer(model, config)
        rgb = renderer.render(data)  # Returns (H, W, 3) uint8 array
        
        # Or render from specific camera
        rgb = renderer.render(data, camera_name="track_cam")
        
        # Clean up when done
        renderer.close()
    """
    
    def __init__(
        self,
        model: mujoco.MjModel,
        config: RenderConfig = None,
    ):
        """
        Initialize the offscreen renderer.
        
        Args:
            model: MuJoCo model
            config: Render configuration
        """
        self.model = model
        self.config = config or RenderConfig()
        
        # Create renderer and scene
        self._renderer = mujoco.Renderer(
            model,
            height=self.config.height,
            width=self.config.width,
        )
        
        # Pre-allocate output arrays for efficiency
        self._rgb_buffer = np.zeros(
            (self.config.height, self.config.width, 3),
            dtype=np.uint8
        )
        self._depth_buffer = np.zeros(
            (self.config.height, self.config.width),
            dtype=np.float32
        )
        
        self._closed = False
        
    @property
    def width(self) -> int:
        return self.config.width
    
    @property
    def height(self) -> int:
        return self.config.height
    
    def render(
        self,
        data: mujoco.MjData,
        camera_name: Optional[str] = None,
        camera_id: Optional[int] = None,
        depth: bool = False,
        segmentation: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        """
        Render the current scene.
        
        Args:
            data: MuJoCo simulation data
            camera_name: Name of camera to render from (overrides config)
            camera_id: ID of camera (overrides camera_name)
            depth: Whether to return depth image
            segmentation: Whether to return segmentation mask
            
        Returns:
            RGB image as (H, W, 3) uint8 array, or tuple of arrays if
            depth/segmentation requested
        """
        if self._closed:
            raise RuntimeError("Renderer has been closed")
        
        # Determine camera
        if camera_id is not None:
            cam_id = camera_id
        elif camera_name is not None:
            cam_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name
            )
            if cam_id == -1:
                raise ValueError(f"Camera '{camera_name}' not found")
        elif self.config.camera_name is not None:
            cam_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.config.camera_name
            )
        else:
            cam_id = self.config.camera_id
        
        # Update scene
        self._renderer.update_scene(data, camera=cam_id)
        
        # Render RGB
        rgb = self._renderer.render()
        
        # Return based on what was requested
        outputs = [rgb]
        
        if depth:
            # mujoco.Renderer.enable_depth_rendering/disable_depth_rendering
            # do not take arguments; call enable then disable.
            self._renderer.enable_depth_rendering()
            depth_img = self._renderer.render()
            self._renderer.disable_depth_rendering()
            outputs.append(depth_img)
            
        if segmentation:
            self._renderer.enable_segmentation_rendering()
            seg_img = self._renderer.render()
            self._renderer.disable_segmentation_rendering()
            outputs.append(seg_img)
        
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)
    
    def render_free_camera(
        self,
        data: mujoco.MjData,
        lookat: Optional[Tuple[float, float, float]] = None,
        distance: Optional[float] = None,
        azimuth: Optional[float] = None,
        elevation: Optional[float] = None,
    ) -> np.ndarray:
        """
        Render from a freely positioned camera.
        
        Args:
            data: MuJoCo simulation data
            lookat: Point to look at (x, y, z)
            distance: Distance from lookat point
            azimuth: Horizontal angle in degrees
            elevation: Vertical angle in degrees
            
        Returns:
            RGB image as (H, W, 3) uint8 array
        """
        if self._closed:
            raise RuntimeError("Renderer has been closed")
        
        # Use config defaults if not specified
        lookat = lookat or self.config.lookat
        distance = distance if distance is not None else self.config.distance
        azimuth = azimuth if azimuth is not None else self.config.azimuth
        elevation = elevation if elevation is not None else self.config.elevation
        
        # Update scene with free camera
        self._renderer.update_scene(
            data,
            camera=-1,  # Free camera
            scene_option=None,
        )
        
        # Set camera parameters
        self._renderer.scene.camera.lookat[:] = lookat
        self._renderer.scene.camera.distance = distance
        self._renderer.scene.camera.azimuth = azimuth
        self._renderer.scene.camera.elevation = elevation
        
        return self._renderer.render()
    
    def render_robot_view(
        self,
        data: mujoco.MjData,
        body_name: str = "torso",
        offset: Tuple[float, float, float] = (0.3, 0.0, 0.1),
    ) -> np.ndarray:
        """
        Render from a view attached to the robot body.
        
        Useful for ego-centric observations.
        
        Args:
            data: MuJoCo simulation data  
            body_name: Name of body to attach camera to
            offset: Camera offset from body center
            
        Returns:
            RGB image as (H, W, 3) uint8 array
        """
        body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, body_name
        )
        if body_id == -1:
            raise ValueError(f"Body '{body_name}' not found")
        
        # Get body position and orientation
        body_pos = data.xpos[body_id]
        body_mat = data.xmat[body_id].reshape(3, 3)
        
        # Compute camera position
        cam_pos = body_pos + body_mat @ np.array(offset)
        
        # Look in direction of body forward axis
        lookat = body_pos + body_mat @ np.array([1.0, 0.0, 0.0])
        
        return self.render_free_camera(
            data,
            lookat=tuple(lookat),
            distance=np.linalg.norm(cam_pos - lookat),
        )
    
    def resize(self, width: int, height: int):
        """Resize the render output."""
        self.config.width = width
        self.config.height = height
        
        # Recreate renderer with new size
        self._renderer = mujoco.Renderer(
            self.model,
            height=height,
            width=width,
        )
        
        # Reallocate buffers
        self._rgb_buffer = np.zeros((height, width, 3), dtype=np.uint8)
        self._depth_buffer = np.zeros((height, width), dtype=np.float32)
    
    def close(self):
        """Clean up renderer resources."""
        if not self._closed:
            self._renderer.close()
            self._closed = True
    
    def __del__(self):
        self.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
