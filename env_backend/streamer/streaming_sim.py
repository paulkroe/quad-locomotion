"""
Streaming simulator wrapper.

Wraps a simulator to add streaming capabilities for browser visualization.
"""

import os
# Force headless rendering
if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

from typing import Optional, Dict, Any, Tuple
import numpy as np
import time

from ..simulator.config import SimulatorConfig, RenderConfig
from ..simulator.core import MujocoSimulator
from ..simulator.headless import HeadlessRenderer
from .server import StreamServer, StreamConfig


class StreamingSimulator:
    """
    Simulator wrapper that streams rendered frames to browsers.
    
    Combines MujocoSimulator with streaming capabilities for 
    real-time visualization without a local display.
    
    Usage:
        config = SimulatorConfig(robot_xml_path="robot.xml")
        sim = StreamingSimulator(config, stream_port=8765)
        
        # Start streaming (opens WebSocket server)
        sim.start_streaming()
        
        # Run simulation - frames are automatically streamed
        obs, info = sim.reset()
        for _ in range(1000):
            action = policy(obs)
            obs, reward, done, truncated, info = sim.step(action)
        
        sim.stop_streaming()
        sim.close()
    """
    
    def __init__(
        self,
        config: SimulatorConfig,
        stream_config: StreamConfig = None,
        render_every: int = 1,
        track_body: str = "torso",
    ):
        """
        Initialize streaming simulator.
        
        Args:
            config: Simulator configuration
            stream_config: Streaming configuration
            render_every: Render every N steps (for performance)
            track_body: Body to track with camera
        """
        self.sim = MujocoSimulator(config)
        self.stream_config = stream_config or StreamConfig()
        self.render_every = render_every
        self.track_body = track_body
        
        self._step_count = 0
        
        # Create headless renderer
        render_cfg = config.render_config
        self._renderer = HeadlessRenderer(
            self.sim.model,
            width=render_cfg.width,
            height=render_cfg.height,
            config=render_cfg,
        )
        
        # Stream server (lazy init)
        self._server: Optional[StreamServer] = None
        self._streaming = False
    
    def start_streaming(self):
        """Start the streaming server."""
        if self._streaming:
            return
        
        self._server = StreamServer(self.stream_config)
        self._server.start()
        self._streaming = True
        
        # Generate viewer HTML
        from .client import save_viewer_html
        save_viewer_html(
            "viewer.html",
            ws_url=f"ws://{self.stream_config.host}:{self.stream_config.port}"
        )
    
    def stop_streaming(self):
        """Stop the streaming server."""
        if self._server:
            self._server.stop()
            self._server = None
        self._streaming = False
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the simulation."""
        self._step_count = 0
        obs, info = self.sim.reset(seed=seed, options=options)
        
        # Stream initial frame
        if self._streaming:
            self._stream_frame(info)
        
        return obs, info
    
    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Step the simulation."""
        obs, reward, done, truncated, info = self.sim.step(action)
        self._step_count += 1
        
        # Stream frame at configured rate
        if self._streaming and self._step_count % self.render_every == 0:
            self._stream_frame(info)
        
        return obs, reward, done, truncated, info
    
    def _stream_frame(self, info: Dict[str, Any]):
        """Render and stream a frame."""
        if not self._server:
            return
        
        # Render with tracking
        frame = self._renderer.render_tracking(
            self.sim.data,
            track_body=self.track_body,
        )
        
        # Add simulation info as metadata
        metadata = {
            "step": self._step_count,
            "time": self.sim.time,
            **{k: v for k, v in info.items() if isinstance(v, (int, float, str))}
        }
        
        self._server.send_frame(frame, metadata)
    
    def render(self) -> np.ndarray:
        """Render current frame."""
        return self._renderer.render_tracking(
            self.sim.data,
            track_body=self.track_body,
        )
    
    # Delegate common properties/methods to underlying simulator
    @property
    def model(self):
        return self.sim.model
    
    @property
    def data(self):
        return self.sim.data
    
    @property
    def dt(self):
        return self.sim.dt
    
    @property
    def time(self):
        return self.sim.time
    
    @property
    def n_actuators(self):
        return self.sim.n_actuators
    
    def seed(self, seed: Optional[int] = None):
        return self.sim.seed(seed)
    
    def get_state(self):
        return self.sim.get_state()
    
    def set_state(self, qpos, qvel):
        return self.sim.set_state(qpos, qvel)
    
    def close(self):
        """Clean up resources."""
        self.stop_streaming()
        self._renderer.close()
        self.sim.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
        return False
