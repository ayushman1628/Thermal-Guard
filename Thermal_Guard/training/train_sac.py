
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.datacenter_env import DataCentreEnv
from training.baselines import RuleBasedAgent, PIDAgent, evaluate_agent, print_results

try:
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import (
        EvalCallback,
        CheckpointCallback,
        BaseCallback,
    )
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.monitor import Monitor
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("⚠️  stable-baselines3 not installed.")
    print("    Install with: pip install stable-baselines3[extra]")
    print("    Running in demo mode — will show config only.\n")


# ─────────────────────────────────────────────────────────────────────────
# CUSTOM CALLBACK: logs PUE and temp violations during training
# ─────────────────────────────────────────────────────────────────────────

class DataCentreMetricsCallback(BaseCallback if SB3_AVAILABLE else object):
    """
    Custom callback that logs domain-specific metrics during training.

    stable-baselines3 callbacks let hook into the training loop.
    This one tracks PUE and violations alongside the default reward logging.

    """

    def __init__(self, verbose=0):
        if SB3_AVAILABLE:
            super().__init__(verbose)
        self.pue_history = []
        self.violation_history = []

    def _on_step(self) -> bool:
        # Extract info from environment at each step
        for info in self.locals.get("infos", []):
            if "pue" in info:
                self.pue_history.append(info["pue"])
            if "server_temp" in info:
                violation = info["server_temp"] > 27.0 or info["server_temp"] < 18.0
                self.violation_history.append(float(violation))
        return True  # return False to stop training early
