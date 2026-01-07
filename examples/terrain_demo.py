#!/usr/bin/env python3
"""
Example: Generate and visualize box field terrain.

This demonstrates creating a terrain similar to the one in the reference image -
a field of boxes with varying heights for quadruped locomotion training.
"""

import numpy as np
import mujoco

from env_backend.configs.env_config import BoxesTerrainConfig, SceneConfig
from env_backend.terrain import SceneBuilder, BoxesTerrainGenerator


def main():
    # Configure the box field terrain
    # These settings create terrain similar to the reference image
    terrain_config = BoxesTerrainConfig(
        arena_size=(8.0, 8.0),  # 8x8 meter arena
        
        # Box dimensions
        box_size_x=(0.25, 0.45),  # Box width range
        box_size_y=(0.25, 0.45),  # Box depth range  
        box_height=(0.02, 0.18),  # Height range (low to tall boxes)
        
        # Grid settings
        grid_spacing=0.02,  # Small gaps between boxes
        
        # Height variation
        height_noise_scale=0.8,  # Smooth height transitions
        
        # Colors (grayscale like the reference)
        box_rgba_range=(
            (0.3, 0.3, 0.3, 1.0),  # Dark gray for low boxes
            (0.95, 0.95, 0.95, 1.0),  # Light gray for tall boxes
        ),
        
        # Spawn area (keep center clear for robot)
        spawn_area_size=(0.0, 0.0),
        spawn_offset=(0.0, 0.0),
        
        # Set seed for reproducibility (change for different terrains)
        seed=42,
    )
    
    # Create scene configuration
    scene_config = SceneConfig(
        terrain=terrain_config,
        timestep=0.002,
        shadow_enabled=True,
    )
    
    # Build the scene
    builder = SceneBuilder(scene_config)
    xml_string = builder.build()
    
    # Save to file (optional)
    output_path = builder.save("generated_scenes/boxes_terrain.xml")
    print(f"Scene saved to: {output_path}")
    
    # Load into MuJoCo
    model = mujoco.MjModel.from_xml_string(xml_string)
    data = mujoco.MjData(model)
    
    print(f"Terrain generated with {model.ngeom} geoms")
    print(f"Arena size: {terrain_config.arena_size}")
    print(f"Box height range: {terrain_config.box_height}")

def generate_multiple_terrains():
    """Generate multiple terrain variations for training."""
    
    print("Generating terrain variations...")
    
    for difficulty in [0.2, 0.5, 0.8]:
        for seed in range(3):
            # Scale box heights with difficulty
            min_height = 0.02 + difficulty * 0.03
            max_height = 0.10 + difficulty * 0.15
            
            terrain_config = BoxesTerrainConfig(
                arena_size=(10.0, 10.0),
                box_height=(min_height, max_height),
                height_noise_scale=0.5 + difficulty * 0.5,
                seed=seed + int(difficulty * 100),
            )
            
            scene_config = SceneConfig(terrain=terrain_config)
            builder = SceneBuilder(scene_config)
            
            filename = f"terrain_diff{int(difficulty*10):02d}_seed{seed}.xml"
            output_path = builder.save(f"generated_scenes/{filename}")
            print(f"  Generated: {filename}")
    
    print("Done! Generated 9 terrain variations.")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--batch":
        generate_multiple_terrains()
    else:
        main()
