"""
Vectorized simulator for parallel environment execution.

Provides:
- VecSimulator: Simple vectorized wrapper (sequential)
- SubprocVecSimulator: True parallel execution via subprocesses
"""

from typing import Optional, Tuple, Dict, Any, List, Callable, Union
from abc import ABC, abstractmethod
import numpy as np
import multiprocessing as mp
from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection
import cloudpickle

from .config import SimulatorConfig
from .core import MujocoSimulator


class BaseVecSimulator(ABC):
    """
    Abstract base class for vectorized simulators.
    
    Provides a unified interface for running multiple environments
    in parallel for RL training.
    """
    
    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self._closed = False
    
    @abstractmethod
    def reset(
        self,
        seed: Optional[Union[int, List[int]]] = None,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Reset all environments."""
        pass
    
    @abstractmethod
    def step(
        self,
        actions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """Step all environments."""
        pass
    
    @abstractmethod
    def render(self, indices: Optional[List[int]] = None) -> np.ndarray:
        """Render specified environments."""
        pass
    
    @abstractmethod
    def close(self):
        """Clean up all environments."""
        pass
    
    @property
    @abstractmethod
    def observation_shape(self) -> Tuple[int, ...]:
        """Shape of observations."""
        pass
    
    @property
    @abstractmethod
    def action_shape(self) -> Tuple[int, ...]:
        """Shape of actions."""
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class VecSimulator(BaseVecSimulator):
    """
    Simple vectorized simulator using sequential execution.
    
    Good for debugging and when true parallelism isn't needed.
    Uses the same process for all environments.
    
    Usage:
        configs = [SimulatorConfig(...) for _ in range(4)]
        vec_sim = VecSimulator(configs)
        
        obs, infos = vec_sim.reset()
        for _ in range(1000):
            actions = policy(obs)  # (num_envs, action_dim)
            obs, rewards, dones, truncateds, infos = vec_sim.step(actions)
        
        vec_sim.close()
    """
    
    def __init__(
        self,
        configs: List[SimulatorConfig],
        auto_reset: bool = True,
    ):
        """
        Initialize vectorized simulator.
        
        Args:
            configs: List of configurations, one per environment
            auto_reset: Whether to automatically reset finished episodes
        """
        super().__init__(len(configs))
        
        self.configs = configs
        self.auto_reset = auto_reset
        
        # Create all simulators
        self._sims = [MujocoSimulator(cfg) for cfg in configs]
        
        # Cache shapes
        self._obs_shape = (self._sims[0]._obs_dim,)
        self._action_shape = (self._sims[0].n_actuators,)
        
        # Pre-allocate output arrays
        self._obs_buffer = np.zeros(
            (self.num_envs,) + self._obs_shape, dtype=np.float32
        )
        self._reward_buffer = np.zeros(self.num_envs, dtype=np.float32)
        self._done_buffer = np.zeros(self.num_envs, dtype=bool)
        self._truncated_buffer = np.zeros(self.num_envs, dtype=bool)
    
    @property
    def observation_shape(self) -> Tuple[int, ...]:
        return self._obs_shape
    
    @property
    def action_shape(self) -> Tuple[int, ...]:
        return self._action_shape
    
    def reset(
        self,
        seed: Optional[Union[int, List[int]]] = None,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Reset all environments."""
        infos = []
        
        for i, sim in enumerate(self._sims):
            env_seed = seed[i] if isinstance(seed, list) else (
                seed + i if seed is not None else None
            )
            obs, info = sim.reset(seed=env_seed)
            self._obs_buffer[i] = obs
            infos.append(info)
        
        return self._obs_buffer.copy(), infos
    
    def reset_at(
        self,
        index: int,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset a specific environment."""
        obs, info = self._sims[index].reset(seed=seed)
        self._obs_buffer[index] = obs
        return obs, info
    
    def step(
        self,
        actions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Step all environments.
        
        Args:
            actions: Actions for all envs, shape (num_envs, action_dim)
            
        Returns:
            observations: (num_envs, obs_dim)
            rewards: (num_envs,)
            terminateds: (num_envs,) bool
            truncateds: (num_envs,) bool
            infos: List of info dicts
        """
        infos = []
        
        for i, sim in enumerate(self._sims):
            obs, reward, done, truncated, info = sim.step(actions[i])
            
            self._obs_buffer[i] = obs
            self._reward_buffer[i] = reward
            self._done_buffer[i] = done
            self._truncated_buffer[i] = truncated
            
            # Auto-reset if episode ended
            if self.auto_reset and (done or truncated):
                final_obs = obs.copy()
                obs, reset_info = sim.reset()
                self._obs_buffer[i] = obs
                info["final_observation"] = final_obs
                info["final_info"] = info.copy()
            
            infos.append(info)
        
        return (
            self._obs_buffer.copy(),
            self._reward_buffer.copy(),
            self._done_buffer.copy(),
            self._truncated_buffer.copy(),
            infos,
        )
    
    def render(self, indices: Optional[List[int]] = None) -> np.ndarray:
        """
        Render specified environments.
        
        Args:
            indices: Which envs to render (default: [0])
            
        Returns:
            Stacked images, shape (n_indices, H, W, 3)
        """
        if indices is None:
            indices = [0]
        
        images = [self._sims[i].render() for i in indices]
        return np.stack(images)
    
    def get_attr(self, name: str, indices: Optional[List[int]] = None) -> List[Any]:
        """Get an attribute from specified environments."""
        if indices is None:
            indices = list(range(self.num_envs))
        return [getattr(self._sims[i], name) for i in indices]
    
    def set_attr(self, name: str, value: Any, indices: Optional[List[int]] = None):
        """Set an attribute on specified environments."""
        if indices is None:
            indices = list(range(self.num_envs))
        for i in indices:
            setattr(self._sims[i], name, value)
    
    def call(
        self,
        method_name: str,
        indices: Optional[List[int]] = None,
        *args,
        **kwargs,
    ) -> List[Any]:
        """Call a method on specified environments."""
        if indices is None:
            indices = list(range(self.num_envs))
        return [
            getattr(self._sims[i], method_name)(*args, **kwargs)
            for i in indices
        ]
    
    def close(self):
        """Clean up all environments."""
        if not self._closed:
            for sim in self._sims:
                sim.close()
            self._closed = True


# Worker commands for subprocess communication
_CMD_RESET = 0
_CMD_STEP = 1
_CMD_RENDER = 2
_CMD_CLOSE = 3
_CMD_GET_ATTR = 4
_CMD_SET_ATTR = 5
_CMD_CALL = 6
_CMD_GET_SPACES = 7


def _worker(
    remote: Connection,
    parent_remote: Connection,
    config_bytes: bytes,
):
    """Worker process for SubprocVecSimulator."""
    parent_remote.close()
    
    # Unpickle config and create simulator
    config = cloudpickle.loads(config_bytes)
    sim = MujocoSimulator(config)
    
    try:
        while True:
            cmd, data = remote.recv()
            
            if cmd == _CMD_RESET:
                result = sim.reset(seed=data)
                remote.send(result)
                
            elif cmd == _CMD_STEP:
                result = sim.step(data)
                remote.send(result)
                
            elif cmd == _CMD_RENDER:
                result = sim.render(**data) if data else sim.render()
                remote.send(result)
                
            elif cmd == _CMD_GET_ATTR:
                result = getattr(sim, data)
                remote.send(result)
                
            elif cmd == _CMD_SET_ATTR:
                name, value = data
                setattr(sim, name, value)
                remote.send(None)
                
            elif cmd == _CMD_CALL:
                method_name, args, kwargs = data
                result = getattr(sim, method_name)(*args, **kwargs)
                remote.send(result)
                
            elif cmd == _CMD_GET_SPACES:
                result = (sim._obs_dim, sim.n_actuators)
                remote.send(result)
                
            elif cmd == _CMD_CLOSE:
                sim.close()
                remote.close()
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        sim.close()


class SubprocVecSimulator(BaseVecSimulator):
    """
    Vectorized simulator using subprocesses for true parallelism.
    
    Each environment runs in its own process, enabling parallel
    stepping and rendering. Best for compute-heavy simulations.
    
    Usage:
        configs = [SimulatorConfig(...) for _ in range(8)]
        vec_sim = SubprocVecSimulator(configs)
        
        obs, infos = vec_sim.reset()
        for _ in range(1000):
            actions = policy(obs)
            obs, rewards, dones, truncateds, infos = vec_sim.step(actions)
        
        vec_sim.close()
    """
    
    def __init__(
        self,
        configs: List[SimulatorConfig],
        auto_reset: bool = True,
        start_method: str = "spawn",
    ):
        """
        Initialize subprocess vectorized simulator.
        
        Args:
            configs: List of configurations, one per environment
            auto_reset: Whether to automatically reset finished episodes
            start_method: Multiprocessing start method ('spawn', 'fork', 'forkserver')
        """
        super().__init__(len(configs))
        
        self.configs = configs
        self.auto_reset = auto_reset
        
        # Set up multiprocessing context
        ctx = mp.get_context(start_method)
        
        # Create communication pipes and workers
        self._remotes: List[Connection] = []
        self._work_remotes: List[Connection] = []
        self._processes: List[Process] = []
        
        for i, config in enumerate(configs):
            parent_remote, work_remote = ctx.Pipe()
            self._remotes.append(parent_remote)
            self._work_remotes.append(work_remote)
            
            # Serialize config
            config_bytes = cloudpickle.dumps(config)
            
            # Create worker process
            process = ctx.Process(
                target=_worker,
                args=(work_remote, parent_remote, config_bytes),
                daemon=True,
            )
            process.start()
            self._processes.append(process)
            work_remote.close()
        
        # Get space info from first worker
        self._remotes[0].send((_CMD_GET_SPACES, None))
        obs_dim, n_actuators = self._remotes[0].recv()
        
        self._obs_shape = (obs_dim,)
        self._action_shape = (n_actuators,)
        
        # Pre-allocate buffers
        self._obs_buffer = np.zeros(
            (self.num_envs,) + self._obs_shape, dtype=np.float32
        )
        self._reward_buffer = np.zeros(self.num_envs, dtype=np.float32)
        self._done_buffer = np.zeros(self.num_envs, dtype=bool)
        self._truncated_buffer = np.zeros(self.num_envs, dtype=bool)
    
    @property
    def observation_shape(self) -> Tuple[int, ...]:
        return self._obs_shape
    
    @property
    def action_shape(self) -> Tuple[int, ...]:
        return self._action_shape
    
    def reset(
        self,
        seed: Optional[Union[int, List[int]]] = None,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Reset all environments in parallel."""
        # Send reset commands
        for i, remote in enumerate(self._remotes):
            env_seed = seed[i] if isinstance(seed, list) else (
                seed + i if seed is not None else None
            )
            remote.send((_CMD_RESET, env_seed))
        
        # Collect results
        infos = []
        for i, remote in enumerate(self._remotes):
            obs, info = remote.recv()
            self._obs_buffer[i] = obs
            infos.append(info)
        
        return self._obs_buffer.copy(), infos
    
    def step(
        self,
        actions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """Step all environments in parallel."""
        # Send step commands
        for i, remote in enumerate(self._remotes):
            remote.send((_CMD_STEP, actions[i]))
        
        # Collect results
        infos = []
        for i, remote in enumerate(self._remotes):
            obs, reward, done, truncated, info = remote.recv()
            
            self._obs_buffer[i] = obs
            self._reward_buffer[i] = reward
            self._done_buffer[i] = done
            self._truncated_buffer[i] = truncated
            
            # Auto-reset if episode ended
            if self.auto_reset and (done or truncated):
                info["final_observation"] = obs.copy()
                info["final_info"] = info.copy()
                
                # Reset this environment
                remote.send((_CMD_RESET, None))
                obs, reset_info = remote.recv()
                self._obs_buffer[i] = obs
            
            infos.append(info)
        
        return (
            self._obs_buffer.copy(),
            self._reward_buffer.copy(),
            self._done_buffer.copy(),
            self._truncated_buffer.copy(),
            infos,
        )
    
    def render(self, indices: Optional[List[int]] = None) -> np.ndarray:
        """Render specified environments in parallel."""
        if indices is None:
            indices = [0]
        
        # Send render commands
        for i in indices:
            self._remotes[i].send((_CMD_RENDER, None))
        
        # Collect images
        images = [self._remotes[i].recv() for i in indices]
        return np.stack(images)
    
    def get_attr(self, name: str, indices: Optional[List[int]] = None) -> List[Any]:
        """Get an attribute from specified environments."""
        if indices is None:
            indices = list(range(self.num_envs))
        
        for i in indices:
            self._remotes[i].send((_CMD_GET_ATTR, name))
        
        return [self._remotes[i].recv() for i in indices]
    
    def set_attr(self, name: str, value: Any, indices: Optional[List[int]] = None):
        """Set an attribute on specified environments."""
        if indices is None:
            indices = list(range(self.num_envs))
        
        for i in indices:
            self._remotes[i].send((_CMD_SET_ATTR, (name, value)))
        
        for i in indices:
            self._remotes[i].recv()
    
    def call(
        self,
        method_name: str,
        indices: Optional[List[int]] = None,
        *args,
        **kwargs,
    ) -> List[Any]:
        """Call a method on specified environments."""
        if indices is None:
            indices = list(range(self.num_envs))
        
        for i in indices:
            self._remotes[i].send((_CMD_CALL, (method_name, args, kwargs)))
        
        return [self._remotes[i].recv() for i in indices]
    
    def close(self):
        """Clean up all worker processes."""
        if not self._closed:
            for remote in self._remotes:
                try:
                    remote.send((_CMD_CLOSE, None))
                except:
                    pass
            
            for process in self._processes:
                process.join(timeout=1)
                if process.is_alive():
                    process.terminate()
            
            self._closed = True


def make_vec_simulator(
    config: SimulatorConfig,
    num_envs: int,
    parallel: bool = True,
    **kwargs,
) -> BaseVecSimulator:
    """
    Convenience function to create a vectorized simulator.
    
    Args:
        config: Base configuration (will be copied for each env)
        num_envs: Number of environments
        parallel: Whether to use subprocess parallelism
        **kwargs: Additional arguments for the VecSimulator
        
    Returns:
        Vectorized simulator instance
    """
    # Create configs with different seeds
    configs = []
    for i in range(num_envs):
        cfg = SimulatorConfig(
            robot_xml_path=config.robot_xml_path,
            terrain_xml_path=config.terrain_xml_path,
            scene_xml_path=config.scene_xml_path,
            timestep=config.timestep,
            n_substeps=config.n_substeps,
            control_timestep=config.control_timestep,
            random_init=config.random_init,
            init_noise_scale=config.init_noise_scale,
            render_config=config.render_config,
            max_episode_steps=config.max_episode_steps,
            observation_keys=config.observation_keys,
        )
        configs.append(cfg)
    
    if parallel:
        return SubprocVecSimulator(configs, **kwargs)
    else:
        return VecSimulator(configs, **kwargs)
