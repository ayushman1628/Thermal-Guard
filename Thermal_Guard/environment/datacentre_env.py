import numpy as np
try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    try:
        import gym
        from gym import spaces
        GYM_AVAILABLE = True
    except ImportError:
        GYM_AVAILABLE = False
        # Minimal stubs so the file can be tested standalone
        class _Box:
            def __init__(self, low, high, dtype=np.float32):
                self.low = low; self.high = high; self.dtype = dtype
                self.shape = low.shape
            def sample(self):
                return (np.random.rand(*self.shape) * (self.high - self.low) + self.low).astype(self.dtype)
            def contains(self, x):
                return np.all(x >= self.low) and np.all(x <= self.high)
            def __repr__(self):
                return f"Box({self.low}, {self.high})"
        class spaces:
            Box = _Box
        class gym:
            class Env:
                def __init__(self): pass
                def reset(self, seed=None, options=None): pass
from typing import Optional, Tuple, Dict, Any
import sys
import os

# Add parent directory to path so we can import thermal_model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.thermal_model import ThermalModel, ServerLoadProfile, WeatherProfile


_BaseEnv = gym.Env

