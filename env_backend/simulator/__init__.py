"""
MuJoCo Simulator for quadruped locomotion training.

Provides:
- Single environment simulation with clean RL APIs
- Vectorized/parallel environment support
- Offscreen/headless rendering for training and streaming
- Robot + terrain scene management
"""

# Set headless rendering before importing mujoco
import os
if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

from .core import MujocoSimulator
from .renderer import OffscreenRenderer
from .headless import HeadlessRenderer
from .vec_env import VecSimulator, SubprocVecSimulator
from .config import SimulatorConfig, RenderConfig

__all__ = [
    "MujocoSimulator",
    "OffscreenRenderer",
    "HeadlessRenderer",
    "VecSimulator",
    "SubprocVecSimulator",
    "SimulatorConfig",
    "RenderConfig",
]
