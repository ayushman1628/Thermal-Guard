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

class DataCentreEnv(_BaseEnv):
    """
    Single-zone Data Centre Cooling Environment.

    The agent controls one CRAC unit's supply temperature setpoint.
    Goal: minimize PUE (energy waste) while keeping servers in safe
    temperature range (ASHRAE: 18–27°C).

    Attributes
    ----------
    observation_space : Box(5,)
        [server_temp, server_load_kw, outside_temp, hour_of_day, crac_setpoint]
    action_space : Box(1,)
        [crac_setpoint] in range [16.0, 24.0] °C
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        episode_length_hours: float = 24.0,
        dt_seconds: float = 60.0,
        initial_temp: Optional[float] = None,
        season: str = "summer",    
        render_mode: Optional[str] = None
    ):
        super().__init__()

        self.dt_seconds = dt_seconds
        self.max_steps = int(episode_length_hours * 3600 / dt_seconds)
        self.render_mode = render_mode
        self.season = season
        self._initial_temp = initial_temp

        # ── COMPONENTS ─────
        self.thermal_model = ThermalModel(thermal_mass=500.0)

        # Season determines outside temperature profile
        season_configs = {
            "summer": {"mean_temp": 28.0, "amplitude": 8.0},
            "winter": {"mean_temp": 5.0,  "amplitude": 5.0},
            "spring": {"mean_temp": 15.0, "amplitude": 6.0},
        }
        cfg = season_configs.get(season, season_configs["summer"])
        self.weather = WeatherProfile(**cfg)
        self.load_profile = ServerLoadProfile(base_load_kw=60.0, noise_std=2.0)

        # ── OBSERVATION SPACE ────────────────────────────────────────────
        # Each row: [min, max] for that observation
        #           [server_temp, server_load, outside_temp, hour_of_day, crac_setpoint]
        obs_low  = np.array([15.0,   0.0,  -15.0,  0.0,  16.0], dtype=np.float32)
        obs_high = np.array([45.0, 150.0,   45.0, 24.0,  24.0], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )

        # ── ACTION SPACE ─────────────────────────────────────────────────
        # Single continuous action: CRAC supply temperature setpoint
        self.action_space = spaces.Box(
            low=np.array([16.0], dtype=np.float32),
            high=np.array([24.0], dtype=np.float32),
            dtype=np.float32
        )

        # ── REWARD WEIGHTS ───────────────────────────────────────────────
        self.w_efficiency  = 1.0    # weight on PUE penalty
        self.w_violation   = 10.0   # weight on temperature violation
        self.w_smoothness  = 0.05   # weight on action changes (mechanical wear)

        # ── EPISODE STATE ─────────────────────────────────────────────────
        self.server_temp    = 22.0
        self.server_load    = 60.0
        self.outside_temp   = 20.0
        self.crac_setpoint  = 20.0
        self.timestep       = 0
        self._prev_action   = 20.0  # for smoothness penalty

        # ── LOGGING ──────────────────────────────────────────────────────
        # Stores episode data for analysis
        self._episode_log: list = []