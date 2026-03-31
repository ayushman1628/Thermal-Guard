import numpy as np
from typing import Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.datacenter_env import DataCentreEnv


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