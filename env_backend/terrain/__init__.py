"""
Terrain generation system for MuJoCo environments.

This module provides a modular system for generating randomized terrains
for quadruped locomotion training with sim-to-real transfer in mind.
"""

from .base import TerrainGenerator
from .boxes import BoxesTerrainGenerator
from .scene_builder import SceneBuilder

__all__ = [
    "TerrainGenerator",
    "BoxesTerrainGenerator", 
    "SceneBuilder",
]
