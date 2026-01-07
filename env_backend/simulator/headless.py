"""
Headless rendering utilities using EGL or OSMesa.

Provides rendering without a display for training and streaming.
"""

import os
from typing import Optional, Tuple
import numpy as np

# Force headless rendering backend before importing mujoco
# Try OSMesa first (CPU, more compatible), then EGL
if "MUJOCO_GL" not in os.environ:
    # Try to detect if EGL is available
    try:
        import ctypes
        ctypes.CDLL("libEGL.so.1")
        os.environ["MUJOCO_GL"] = "egl"
    except OSError:
        os.environ["MUJOCO_GL"] = "osmesa"

import mujoco

from .config import RenderConfig


class HeadlessRenderer:
    """
    Headless renderer for MuJoCo simulations.
    
    Uses EGL (GPU) or OSMesa (CPU) for rendering without a display.
    Optimized for streaming and training.
    
    Usage:
        renderer = HeadlessRenderer(model, width=640, height=480)
        rgb = renderer.render(data)
        renderer.close()
    """
    
    def __init__(
        self,
        model: mujoco.MjModel,
        width: int = 640,
        height: int = 480,
        config: Optional[RenderConfig] = None,
    ):
        """
        Initialize headless renderer.
        
        Args:
            model: MuJoCo model
            width: Image width
            height: Image height
            config: Optional render configuration
        """
        # Initialize closed flag first (for cleanup in case of errors)
        self._closed = False
        self._renderer = None
        
        self.model = model
        self.width = width
        self.height = height
        self.config = config or RenderConfig(width=width, height=height)
        
        # Create renderer
        self._renderer = mujoco.Renderer(model, height=height, width=width)
        
        # Camera settings
        self._lookat = np.array(self.config.lookat)
        self._distance = self.config.distance
        self._azimuth = self.config.azimuth
        self._elevation = self.config.elevation
    
    def render(
        self,
        data: mujoco.MjData,
        camera_name: Optional[str] = None,
        camera_id: Optional[int] = None,
    ) -> np.ndarray:
        """
        Render current scene to RGB image.
        
        Args:
            data: MuJoCo simulation data
            camera_name: Name of camera to use
            camera_id: ID of camera to use
            
        Returns:
            RGB image as (H, W, 3) uint8 array
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
                cam_id = -1  # Fall back to free camera
        else:
            cam_id = -1  # Free camera
        
        # Update scene with camera
        if cam_id == -1:
            # Use a virtual camera for free camera view
            camera = mujoco.MjvCamera()
            camera.lookat[:] = self._lookat
            camera.distance = self._distance
            camera.azimuth = self._azimuth
            camera.elevation = self._elevation
            self._renderer.update_scene(data, camera=camera)
        else:
            self._renderer.update_scene(data, camera=cam_id)
        
        return self._renderer.render()
    
    def render_tracking(
        self,
        data: mujoco.MjData,
        track_body: str = "torso",
        distance: float = 3.0,
        elevation: float = -20.0,
        azimuth: float = 90.0,
    ) -> np.ndarray:
        """
        Render while tracking a body.
        
        Args:
            data: MuJoCo simulation data
            track_body: Name of body to track
            distance: Camera distance from body
            elevation: Camera elevation angle
            azimuth: Camera azimuth angle
            
        Returns:
            RGB image as (H, W, 3) uint8 array
        """
        # Get body position
        body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, track_body
        )
        if body_id != -1:
            self._lookat = data.xpos[body_id].copy()
        
        self._distance = distance
        self._elevation = elevation
        self._azimuth = azimuth
        
        return self.render(data)
    
    def set_camera(
        self,
        lookat: Optional[Tuple[float, float, float]] = None,
        distance: Optional[float] = None,
        azimuth: Optional[float] = None,
        elevation: Optional[float] = None,
    ):
        """Update free camera parameters."""
        if lookat is not None:
            self._lookat = np.array(lookat)
        if distance is not None:
            self._distance = distance
        if azimuth is not None:
            self._azimuth = azimuth
        if elevation is not None:
            self._elevation = elevation
    
    def resize(self, width: int, height: int):
        """Resize the renderer."""
        self.width = width
        self.height = height
        if self._renderer is not None:
            self._renderer.close()
        self._renderer = mujoco.Renderer(
            self.model, height=height, width=width
        )
    
    def close(self):
        """Clean up resources."""
        if not self._closed and self._renderer is not None:
            try:
                self._renderer.close()
            except:
                pass
            self._closed = True
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
        return False
    
    def __del__(self):
        self.close()
