import numpy as np
from typing import Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.datacenter_env import DataCentreEnv


class FixedSetpointAgent:
    """
    Always returns the same CRAC setpoint regardless of conditions.

    This is the absolute minimum baseline — even a thermostat beats this.
    We include it to show that naive approaches fail.

    """
    def __init__(self, setpoint: float = 19.0):
        self.setpoint = setpoint

    def predict(self, obs: np.ndarray) -> np.ndarray:
        return np.array([self.setpoint], dtype=np.float32)


class RuleBasedAgent:
    """
    Thermostat-style rule-based controller.

    Mimics how many data centres are actually controlled today:
    threshold-based rules written by engineers.

    Rules:
        - If server temp > 25°C → increase cooling aggressively
        - If server temp > 22°C → increase cooling moderately
        - If server temp < 19°C → reduce cooling (save energy)
        - Otherwise → maintain nominal setpoint

    Also adjusts for server load:
        - High load → more cooling headroom needed
        - Low load  → can afford to be less aggressive

    """

    def predict(self, obs: np.ndarray) -> np.ndarray:
        server_temp    = obs[0]
        server_load_kw = obs[1]

        # Load factor: high load → need more cooling margin
        load_factor = (server_load_kw - 60.0) / 40.0  # -0.5 to +1.0

        # Base setpoint from temperature rules
        if server_temp > 25.0:
            base_setpoint = 16.5   # aggressive cooling
        elif server_temp > 23.0:
            base_setpoint = 18.0   # moderate cooling
        elif server_temp < 19.0:
            base_setpoint = 22.0   # save energy
        elif server_temp < 21.0:
            base_setpoint = 21.0   # slightly less cooling
        else:
            base_setpoint = 19.5   # nominal

        # Adjust for load
        setpoint = base_setpoint - load_factor * 1.5

        return np.array([np.clip(setpoint, 16.0, 24.0)], dtype=np.float32)




class PIDAgent:
    """
    PID Controller — classical control theory baseline.

    PID = Proportional + Integral + Derivative control.

    P: responds to current error (how far from target temp)
    I: responds to accumulated error (corrects steady-state offset)
    D: responds to rate of change (dampens oscillations)

    This is the standard in industrial control systems.
    Many real HVAC systems use PID controllers.

    Parameters
    ----------
    target_temp : float
        Desired server inlet temperature (°C). 
        We target 21°C — middle of ASHRAE range (18–27°C).
    Kp, Ki, Kd : PID gains (tuned manually for this system)
    """

    def __init__(
        self,
        target_temp: float = 21.0,
        Kp: float = 0.5,
        Ki: float = 0.01,
        Kd: float = 0.1,
    ):
        self.target_temp = target_temp
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        # PID state
        self._integral = 0.0
        self._prev_error = 0.0

    def reset(self):
        """Call this at the start of each episode."""
        self._integral = 0.0
        self._prev_error = 0.0