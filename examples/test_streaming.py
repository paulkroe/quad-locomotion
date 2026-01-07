#!/usr/bin/env python3
"""
Test script for the streaming simulator.

This script:
1. Generates a terrain
2. Loads a robot XML (prefers Unitree GO2 by default)
3. Runs simulation with streaming to browser
4. You can view at: file://path/to/viewer.html

Usage:
    python test_streaming.py
    
Then open viewer.html in a browser to see the simulation.
"""

import os
# Force headless rendering
os.environ["MUJOCO_GL"] = "egl"

import sys
import time
import numpy as np
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from env_backend.configs.env_config import BoxesTerrainConfig, SceneConfig
from env_backend.terrain import SceneBuilder
from env_backend.simulator import SimulatorConfig, RenderConfig
from env_backend.streamer import StreamingSimulator, StreamConfig





def main():
    print("=" * 60)
    print("Streaming Simulator Test")
    print("=" * 60)
    
    # Create output directory
    output_dir = Path("generated_scenes")
    output_dir.mkdir(exist_ok=True)
    
    # Step 1: Generate terrain
    print("\n1. Generating terrain...")
    terrain_config = BoxesTerrainConfig(
        arena_size=(6.0, 6.0),
        box_height=(0.02, 0.10),
        spawn_area_size=(0.5, 0.5),
        seed=42,
    )
    scene_config = SceneConfig(terrain=terrain_config)
    builder = SceneBuilder(scene_config)
    terrain_path = output_dir / "stream_test_terrain.xml"
    builder.save(terrain_path)
    print(f"   Terrain saved to: {terrain_path}")
    
    # Step 2: Select robot XML (prefer GO2 or use ROBOT_XML env var)
    print("\n2. Selecting robot model...")
    DEFAULT_ROBOT_PATH = Path(os.environ.get(
        "ROBOT_XML",
        "/home/paul/Documents/quad-locomotion/mujoco_menagerie/unitree_go2/go2.xml",
    ))
    if not DEFAULT_ROBOT_PATH.exists():
        print(f"   Error: No robot XML found at {DEFAULT_ROBOT_PATH}.")
        print("   Set the ROBOT_XML environment variable to a valid robot XML file.")
        return
    robot_path = DEFAULT_ROBOT_PATH
    print(f"   Using robot model at: {robot_path}")
    
    # Step 3: Configure simulator
    print("\n3. Configuring simulator...")
    sim_config = SimulatorConfig(
        robot_xml_path=str(robot_path.absolute()),
        terrain_xml_path=str(terrain_path.absolute()),
        timestep=0.002,
        n_substeps=10,  # 20ms control dt
        max_episode_steps=1000,
        render_config=RenderConfig(
            width=640,
            height=480,
            distance=2.5,
            elevation=-25,
            azimuth=135,
        ),
    )
    
    stream_config = StreamConfig(
        host="localhost",
        port=8765,
        max_fps=30,
        jpeg_quality=85,
    )
    
    # Step 4: Create and start streaming simulator
    print("\n4. Starting streaming simulator...")
    sim = StreamingSimulator(
        sim_config,
        stream_config=stream_config,
        render_every=2,  # Render every 2 steps for performance
        track_body="torso",
    )
    
    sim.start_streaming()
    
    print("\n" + "=" * 60)
    print("STREAMING ACTIVE!")
    print("=" * 60)
    print(f"\nOpen 'viewer.html' in a browser to see the simulation")
    print(f"WebSocket URL: ws://{stream_config.host}:{stream_config.port}")
    print("\nPress Ctrl+C to stop\n")
    
    # Step 5: Run simulation loop
    try:
        episode = 0
        while True:
            episode += 1
            print(f"Episode {episode}")
            
            obs, info = sim.reset(seed=episode)
            total_reward = 0
            step = 0
            
            done = False
            while not done:
                # Simple oscillating action for visual effect
                t = step * sim.dt
                action = np.zeros(sim.n_actuators)
                
                # Oscillate hip joints
                action[0] = 0.5 * np.sin(2 * np.pi * t * 2)  # FL hip
                action[2] = 0.5 * np.sin(2 * np.pi * t * 2 + np.pi)  # FR hip
                action[4] = 0.5 * np.sin(2 * np.pi * t * 2 + np.pi)  # RL hip
                action[6] = 0.5 * np.sin(2 * np.pi * t * 2)  # RR hip
                
                # Oscillate knee joints
                action[1] = -0.5 + 0.3 * np.sin(2 * np.pi * t * 2)  # FL knee
                action[3] = -0.5 + 0.3 * np.sin(2 * np.pi * t * 2 + np.pi)  # FR knee
                action[5] = -0.5 + 0.3 * np.sin(2 * np.pi * t * 2 + np.pi)  # RL knee
                action[7] = -0.5 + 0.3 * np.sin(2 * np.pi * t * 2)  # RR knee
                
                obs, reward, terminated, truncated, info = sim.step(action)
                total_reward += reward
                step += 1
                done = terminated or truncated
                
                # Control simulation speed
                time.sleep(0.01)
            
            print(f"  Steps: {step}, Reward: {total_reward:.2f}")
            time.sleep(0.5)  # Pause between episodes
            
    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        sim.close()
        print("Simulator closed.")


if __name__ == "__main__":
    main()
