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

    # ─────────────────────────────────────────────────────────────────────
    # CORE GYM METHODS
    # ─────────────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to start a new episode.

        Randomizes starting conditions to help the agent generalize.
        """
        if seed is not None: np.random.seed(seed)

        # Randomize starting temperature
        if self._initial_temp is not None:
            self.server_temp = self._initial_temp
        else:
            self.server_temp = np.random.uniform(19.0, 26.0)

        # Randomize starting load
        self.server_load = np.random.uniform(40.0, 80.0)

        # Start at a random time of day
        self._start_step = int(np.random.uniform(0, 1440))
        self.timestep = 0

        # Initial CRAC setpoint
        self.crac_setpoint = 20.0
        self._prev_action = 20.0

        # Get initial outside temp
        self.outside_temp = self.weather.get_outside_temp(
            self._start_step, self.dt_seconds
        )

        self._episode_log = []

        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Apply action and advance simulation by one timestep (1 minute).

        Parameters
        ----------
        action : np.ndarray shape (1,)
            CRAC supply temperature setpoint [16.0, 24.0] °C

        Returns
        -------
        observation : np.ndarray shape (5,)
        reward      : float
        terminated  : bool  (agent caused episode to end — critical overheating)
        truncated   : bool  (episode hit max length — 24 hours)
        info        : dict  (extra metrics for logging)
        """
        # Clip action to valid range 
        self.crac_setpoint = float(np.clip(action[0], 16.0, 24.0))

        # Current real timestep
        real_step = self._start_step + self.timestep

        # Update external conditions
        self.outside_temp = self.weather.get_outside_temp(real_step, self.dt_seconds)
        self.server_load  = self.load_profile.get_load(real_step, self.dt_seconds)

        # ── THERMAL SIMULATION ────────────────────────────────────────
        thermal_result = self.thermal_model.step(
            current_temp=self.server_temp,
            server_load_kw=self.server_load,
            crac_setpoint=self.crac_setpoint,
            outside_temp=self.outside_temp,
            dt_seconds=self.dt_seconds
        )
        self.server_temp = thermal_result["new_temp"]

        # ── REWARD ────────────────────────────────────────────────────
        reward, reward_components = self._calculate_reward(
            pue=thermal_result["pue"],
            server_temp=self.server_temp,
            action=self.crac_setpoint,
            prev_action=self._prev_action
        )

        # ── TERMINATION ───────────────────────────────────────────────
        # Critical overheating → terminate episode early (hardware damage!)
        terminated = self.server_temp >= self.thermal_model.TEMP_CRITICAL
        truncated  = self.timestep >= self.max_steps - 1

        # ── LOGGING ───────────────────────────────────────────────────
        step_log = {
            "timestep":       self.timestep,
            "server_temp":    self.server_temp,
            "server_load_kw": self.server_load,
            "outside_temp":   self.outside_temp,
            "crac_setpoint":  self.crac_setpoint,
            "crac_power_kw":  thermal_result["crac_power_kw"],
            "pue":            thermal_result["pue"],
            "cop":            thermal_result["cop"],
            "reward":         reward,
            **reward_components
        }
        self._episode_log.append(step_log)

        self._prev_action = self.crac_setpoint
        self.timestep += 1

        return self._get_obs(), reward, terminated, truncated, step_log
    

    # ─────────────────────────────────────────────────────────────────────
    # REWARD FUNCTION
    # ─────────────────────────────────────────────────────────────────────

    def _calculate_reward(
        self,
        pue: float,
        server_temp: float,
        action: float,
        prev_action: float
    ) -> Tuple[float, Dict]:
        """
            The safety weight (10x) is much higher than efficiency (1x)
            because temperature violations can damage hardware — we
            prioritize safety as a hard-ish constraint."

        Returns reward (float) and component breakdown (dict)
        """

        # ── COMPONENT 1: Efficiency (minimize PUE) ────────────────────
        # PUE of 1.0 is perfect. We normalize so reward ~ -0.5 to 0.0
        # at typical operating range (PUE 1.0–2.0)
        r_efficiency = -(pue - 1.0) * self.w_efficiency

        # ── COMPONENT 2: Safety (temperature violations) ───────────────
        violation = self.thermal_model.get_violation_magnitude(server_temp)
        r_safety = -violation * self.w_violation

        # ── COMPONENT 3: Smoothness (avoid rapid setpoint changes) ─────
        action_change = abs(action - prev_action)
        r_smoothness = -action_change * self.w_smoothness

        # Total reward
        total_reward = r_efficiency + r_safety + r_smoothness

        components = {
            "r_efficiency": r_efficiency,
            "r_safety":     r_safety,
            "r_smoothness": r_smoothness,
        }

        return total_reward, components
    
    # ─────────────────────────────────────────────────────────────────────
    # HELPER METHODS
    # ─────────────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        """
        Build observation vector from current state.
        """
        hour_of_day = ((self._start_step + self.timestep) * self.dt_seconds / 3600.0) % 24.0

        return np.array([
            self.server_temp,
            self.server_load,
            self.outside_temp,
            hour_of_day,
            self.crac_setpoint,
        ], dtype=np.float32)
    
    def get_episode_log(self) -> list:
        """Returns all step data from current episode for analysis."""
        return self._episode_log.copy()

    def render(self):
        """Simple terminal render for debugging."""
        if self.render_mode == "human":
            safe = "✅" if self.thermal_model.is_in_safe_zone(self.server_temp) else "⚠️ "
            print(
                f"Step {self.timestep:4d} | "
                f"Temp: {self.server_temp:.1f}°C {safe} | "
                f"Load: {self.server_load:.0f}kW | "
                f"CRAC: {self.crac_setpoint:.1f}°C | "
                f"Outside: {self.outside_temp:.1f}°C"
            )


# ─────────────────────────────────────────────────────────────────────────
# QUICK TEST — run this file directly to verify the environment
# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Testing DataCentreEnv")
    print("=" * 60)

    env = DataCentreEnv(render_mode="human")

    # Test 1: Environment resets correctly
    obs, info = env.reset(seed=42)
    print(f"\nInitial observation:")
    print(f"  server_temp:   {obs[0]:.2f}°C")
    print(f"  server_load:   {obs[1]:.2f} kW")
    print(f"  outside_temp:  {obs[2]:.2f}°C")
    print(f"  hour_of_day:   {obs[3]:.2f}")
    print(f"  crac_setpoint: {obs[4]:.2f}°C")
    print(f"  obs shape:     {obs.shape}  (should be (5,))")
    print(f"  obs dtype:     {obs.dtype}  (should be float32)")

    # Test 2: Step with a valid action
    print("\nRunning 5 steps with render:")
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        print(f"         → reward={reward:.4f}, PUE={info['pue']:.3f}")

    # Test 3: Full episode with random agent
    print("\nFull episode — random agent:")
    obs, _ = env.reset(seed=0)
    total_reward = 0
    pue_list = []
    violations = 0

    for step in range(1440):  # 24 hours
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        pue_list.append(info["pue"])
        if info["server_temp"] > 27 or info["server_temp"] < 18:
            violations += 1
        if terminated or truncated:
            break

    print(f"  Steps completed:      {step + 1}")
    print(f"  Total reward:         {total_reward:.2f}")
    print(f"  Mean PUE:             {np.mean(pue_list):.3f}")
    print(f"  Min PUE:              {np.min(pue_list):.3f}")
    print(f"  Max PUE:              {np.max(pue_list):.3f}")
    print(f"  Temperature violations: {violations} steps ({violations/14.4:.1f}% of day)")

    # Test 4: Observation and action space checks
    print(f"\nSpace checks:")
    print(f"  obs space:    {env.observation_space}")
    print(f"  action space: {env.action_space}")
    print(f"  obs in space: {env.observation_space.contains(obs)}")