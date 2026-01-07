#!/usr/bin/env python3
"""
Example: Full simulation pipeline with terrain and robot.

Demonstrates:
1. Generating randomized terrain
2. Loading a robot model
3. Running simulation with the MujocoSimulator
4. Offscreen rendering
5. Vectorized environments

Note: This example prefers the Unitree GO2 robot model by default.
Set the `ROBOT_XML` environment variable to point to a different robot
XML file. If no robot XML is available the demos will exit with a
clear message.
"""

import os
import numpy as np
from pathlib import Path

from env_backend import (
    # Terrain
    BoxesTerrainConfig,
    SceneConfig,
    SceneBuilder,
    # Simulator
    MujocoSimulator,
    SimulatorConfig,
    RenderConfig,
    VecSimulator,
)





def demo_single_env():
    """Demonstrate single environment simulation."""
    print("=" * 60)
    print("Demo: Single Environment Simulation")
    print("=" * 60)
    
    # Step 1: Generate terrain
    print("\n1. Generating terrain...")
    terrain_config = BoxesTerrainConfig(
        arena_size=(6.0, 6.0),
        box_height=(0.02, 0.12),
        spawn_area_size=(0.5, 0.5),
        seed=42,
    )
    scene_config = SceneConfig(terrain=terrain_config)
    builder = SceneBuilder(scene_config)
    
    # Save terrain
    terrain_path = Path("generated_scenes/demo_terrain.xml")
    builder.save(terrain_path)
    print(f"   Saved terrain to: {terrain_path}")
    
    # Step 2: Select robot model (allow override via ROBOT_XML env var)
    print("\n2. Selecting robot model...")
    DEFAULT_ROBOT_PATH = Path(os.environ.get(
        "ROBOT_XML",
        "/home/paul/Documents/quad-locomotion/mujoco_menagerie/unitree_go2/go2.xml",
    ))

    if DEFAULT_ROBOT_PATH.exists():
        robot_path = DEFAULT_ROBOT_PATH
        print(f"   Using robot model at: {robot_path}")
    else:
        print(f"   Error: No robot XML found at {DEFAULT_ROBOT_PATH}.")
        print("   Set the ROBOT_XML environment variable to a valid robot XML or place the model at the default path.")
        return
    
    # Step 3: Create simulator
    print("\n3. Creating simulator...")
    sim_config = SimulatorConfig(
        robot_xml_path=str(robot_path.absolute()),
        terrain_xml_path=str(terrain_path.absolute()),
        timestep=0.002,
        n_substeps=10,  # 20ms control timestep
        max_episode_steps=500,
        render_config=RenderConfig(
            width=640,
            height=480,
        ),
    )
    
    sim = MujocoSimulator(sim_config)
    print(f"   Created simulator:")
    print(f"   - Observation dim: {sim._obs_dim}")
    print(f"   - Action dim: {sim.n_actuators}")
    print(f"   - Control dt: {sim.dt:.4f}s")
    
    # Step 4: Run simulation
    print("\n4. Running simulation...")
    obs, info = sim.reset(seed=0)
    print(f"   Initial obs shape: {obs.shape}")
    
    total_reward = 0
    for step in range(100):
        # Random action
        action = np.random.uniform(-1, 1, sim.n_actuators)
        obs, reward, done, truncated, info = sim.step(action)
        total_reward += reward
        
        if done:
            print(f"   Episode terminated at step {step}")
            break
    
    print(f"   Completed {step + 1} steps")
    print(f"   Total reward: {total_reward:.2f}")
    
    # Step 5: Render
    print("\n5. Rendering frame...")
    image = sim.render()
    print(f"   Rendered image shape: {image.shape}")
    
    # Save image
    try:
        from PIL import Image
        img = Image.fromarray(image)
        img.save("generated_scenes/demo_frame.png")
        print(f"   Saved frame to: generated_scenes/demo_frame.png")
    except ImportError:
        print("   (PIL not available, skipping image save)")
    
    sim.close()
    print("\n   Simulation closed.")


def demo_vectorized_env():
    """Demonstrate vectorized environment simulation."""
    print("\n" + "=" * 60)
    print("Demo: Vectorized Environment (4 parallel envs)")
    print("=" * 60)
    
    # Create terrain and select robot model
    terrain_path = Path("generated_scenes/demo_terrain.xml")
    DEFAULT_ROBOT_PATH = Path(os.environ.get(
        "ROBOT_XML",
        "/home/paul/Documents/quad-locomotion/mujoco_menagerie/unitree_go2/go2.xml",
    ))
    if not DEFAULT_ROBOT_PATH.exists():
        print(f"   Error: No robot XML found at {DEFAULT_ROBOT_PATH}.")
        print("   Set the ROBOT_XML environment variable to a valid robot XML.")
        return
    robot_path = DEFAULT_ROBOT_PATH
    if not terrain_path.exists():
        print("   Run the single env demo first to create terrain files")
        return
    
    # Create configs for 4 environments with different seeds
    print("\n1. Creating 4 environment configs...")
    configs = []
    for i in range(4):
        # Each env gets different terrain seed
        terrain_config = BoxesTerrainConfig(
            arena_size=(6.0, 6.0),
            box_height=(0.02, 0.12),
            spawn_area_size=(1.5, 1.5),
            seed=42 + i,  # Different terrain per env
        )
        scene_config = SceneConfig(terrain=terrain_config)
        builder = SceneBuilder(scene_config)
        terrain_file = f"generated_scenes/vec_terrain_{i}.xml"
        builder.save(terrain_file)
        
        sim_config = SimulatorConfig(
            robot_xml_path=str(robot_path.absolute()),
            terrain_xml_path=str(Path(terrain_file).absolute()),
            timestep=0.002,
            n_substeps=10,
            max_episode_steps=200,
        )
        configs.append(sim_config)
    
    # Create vectorized environment
    print("\n2. Creating vectorized simulator...")
    vec_sim = VecSimulator(configs, auto_reset=True)
    print(f"   Num envs: {vec_sim.num_envs}")
    print(f"   Obs shape: {vec_sim.observation_shape}")
    print(f"   Action shape: {vec_sim.action_shape}")
    
    # Run simulation
    print("\n3. Running 100 parallel steps...")
    obs, infos = vec_sim.reset(seed=0)
    
    total_rewards = np.zeros(vec_sim.num_envs)
    episode_counts = np.zeros(vec_sim.num_envs)
    
    for step in range(100):
        # Random actions for all envs
        actions = np.random.uniform(-1, 1, (vec_sim.num_envs,) + vec_sim.action_shape)
        obs, rewards, dones, truncateds, infos = vec_sim.step(actions)
        
        total_rewards += rewards
        episode_counts += (dones | truncateds).astype(int)
    
    print(f"   Total rewards per env: {total_rewards}")
    print(f"   Episodes completed per env: {episode_counts}")
    
    # Render from first environment
    print("\n4. Rendering from first environment...")
    images = vec_sim.render(indices=[0])
    print(f"   Rendered images shape: {images.shape}")
    
    vec_sim.close()
    print("\n   Vectorized simulator closed.")


def demo_rendering():
    """Demonstrate various rendering options."""
    print("\n" + "=" * 60)
    print("Demo: Rendering Options")
    print("=" * 60)
    
    terrain_path = Path("generated_scenes/demo_terrain.xml")
    DEFAULT_ROBOT_PATH = Path(os.environ.get(
        "ROBOT_XML",
        "/home/paul/Documents/quad-locomotion/mujoco_menagerie/unitree_go2/go2.xml",
    ))
    if not DEFAULT_ROBOT_PATH.exists():
        print(f"   Error: No robot XML found at {DEFAULT_ROBOT_PATH}.")
        print("   Set the ROBOT_XML environment variable to a valid robot XML.")
        return
    robot_path = DEFAULT_ROBOT_PATH
    if not terrain_path.exists():
        print("   Run the single env demo first to create terrain files")
        return
    
    sim_config = SimulatorConfig(
        robot_xml_path=str(robot_path.absolute()),
        terrain_xml_path=str(terrain_path.absolute()),
        render_config=RenderConfig(
            width=640,
            height=480,
            distance=4.0,
            elevation=-30,
            azimuth=45,
        ),
    )
    
    sim = MujocoSimulator(sim_config)
    obs, _ = sim.reset()
    
    # Take a few steps
    for _ in range(50):
        action = np.random.uniform(-1, 1, sim.n_actuators)
        sim.step(action)
    
    print("\n1. Default render...")
    img1 = sim.render()
    print(f"   Shape: {img1.shape}")
    
    print("\n2. Different resolution...")
    img2 = sim.render(width=320, height=240)
    print(f"   Shape: {img2.shape}")
    
    print("\n3. Depth rendering...")
    depth = sim.render_depth()
    print(f"   Depth shape: {depth.shape}")
    print(f"   Depth range: [{depth.min():.2f}, {depth.max():.2f}]")
    
    # Save renders
    try:
        from PIL import Image
        Image.fromarray(img1).save("generated_scenes/render_default.png")
        Image.fromarray(img2).save("generated_scenes/render_small.png")
        
        # Normalize depth for visualization (handle NaNs safely)
        if np.isnan(depth).all():
            print("\n   Warning: depth render produced only NaNs; skipping depth save")
        else:
            # Replace NaNs with the minimum finite depth for visualization
            if np.isnan(depth).any():
                finite_mask = np.isfinite(depth)
                if finite_mask.any():
                    depth = np.where(np.isfinite(depth), depth, np.nanmin(depth[finite_mask]))
                else:
                    depth = np.nan_to_num(depth)

            dmin = np.nanmin(depth)
            dmax = np.nanmax(depth)
            depth_norm = ((depth - dmin) / (dmax - dmin + 1e-6) * 255).astype(np.uint8)
            Image.fromarray(depth_norm).save("generated_scenes/render_depth.png")
        
        print("\n   Saved renders to generated_scenes/")
    except ImportError:
        pass
    
    sim.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        demo = sys.argv[1]
        if demo == "single":
            demo_single_env()
        elif demo == "vec":
            demo_vectorized_env()
        elif demo == "render":
            demo_rendering()
        else:
            print(f"Unknown demo: {demo}")
            print("Available: single, vec, render")
    else:
        # Run all demos
        demo_single_env()
        demo_vectorized_env()
        demo_rendering()
        
        print("\n" + "=" * 60)
        print("All demos completed!")
        print("=" * 60)
