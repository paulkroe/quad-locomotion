"""
Environment backend for quadruped locomotion training.

This package provides:
- Terrain generation for randomized training environments
- Scene building for MuJoCo simulations
- MuJoCo simulator with clean RL APIs
- Vectorized environments for parallel training
- Offscreen rendering for training and streaming
- PMTG (Policies Modulating Trajectory Generators) for gait generation
"""

from .terrain import SceneBuilder, TerrainGenerator, BoxesTerrainGenerator
from .configs.env_config import (
    TerrainType,
    BaseTerrainConfig,
    BoxesTerrainConfig,
    StairsTerrainConfig,
    SteppingStonesConfig,
    RoughTerrainConfig,
    SceneConfig,
    CurriculumConfig,
)
from .simulator import (
    MujocoSimulator,
    OffscreenRenderer,
    VecSimulator,
    SubprocVecSimulator,
    SimulatorConfig,
    RenderConfig,
)

# Control module is optional (may not be implemented yet)
try:
    from .control import (
        # PMTG
        PMTG,
        PMTGOutput,
        PMTGConfig,
        PMTGWithPD,
        create_pmtg_for_go2,
        # Kinematics
        Go2Kinematics,
        LegKinematics,
        # Trajectory
        FootTrajectoryGenerator,
        # Config
        LegID,
        GaitConfig,
        TrotGaitConfig,
        WalkGaitConfig,
        BoundGaitConfig,
        Go2LegConfig,
    )
    _HAS_CONTROL = True
except ImportError:
    _HAS_CONTROL = False

__all__ = [
    # Terrain
    "SceneBuilder",
    "TerrainGenerator", 
    "BoxesTerrainGenerator",
    # Terrain Configs
    "TerrainType",
    "BaseTerrainConfig",
    "BoxesTerrainConfig",
    "StairsTerrainConfig",
    "SteppingStonesConfig",
    "RoughTerrainConfig",
    "SceneConfig",
    "CurriculumConfig",
    # Simulator
    "MujocoSimulator",
    "OffscreenRenderer",
    "VecSimulator",
    "SubprocVecSimulator",
    "SimulatorConfig",
    "RenderConfig",
]

# Add control exports if available
if _HAS_CONTROL:
    __all__.extend([
        # Control / PMTG
        "PMTG",
        "PMTGOutput",
        "PMTGConfig",
        "PMTGWithPD",
        "create_pmtg_for_go2",
        "Go2Kinematics",
        "LegKinematics",
        "FootTrajectoryGenerator",
        "LegID",
        "GaitConfig",
        "TrotGaitConfig",
        "WalkGaitConfig",
        "BoundGaitConfig",
        "Go2LegConfig",
    ])
