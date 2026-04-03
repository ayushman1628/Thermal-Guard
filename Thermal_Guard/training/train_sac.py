
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




# ─────────────────────────────────────────────────────────────────────────
# TRAINING CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────

CONFIG = {
    # Environment
    "season":           "summer",       # hardest season for cooling
    "episode_hours":    24.0,           # 24-hour episodes

    # Training
    "total_timesteps":  300_000,        # 300k steps (~208 episodes)
                                        # increase to 1M for better performance
    "eval_freq":        10_000,         # evaluate every 10k steps
    "n_eval_episodes":  5,              # episodes per evaluation

    # SAC Hyperparameters
    
    "learning_rate":    3e-4,           # Adam optimizer lr — standard for SAC
    "buffer_size":      100_000,        # replay buffer size (memory of past experience)
    "learning_starts":  2_000,          # steps before training begins (fill buffer first)
    "batch_size":       256,            # mini-batch size for gradient updates
    "tau":              0.005,          # soft update rate for target networks
    "gamma":            0.99,           # discount factor — how much to value future rewards
                                        # 0.99 = agent cares about next ~100 steps
    "train_freq":       1,              # update every step
    "gradient_steps":   1,              # gradient updates per env step

    # Paths
    "model_dir":        "models",
    "log_dir":          "results",
}



def make_env(seed: int = 0, monitor: bool = True):
    """Create and optionally wrap the environment."""
    env = DataCentreEnv(season=CONFIG["season"])
    if SB3_AVAILABLE and monitor:
        env = Monitor(env, filename=None)
    return env


def train_sac():
    """Main training function."""
    print("=" * 65)
    print("  Data Centre Cooling — SAC Training")
    print("=" * 65)
    print(f"  Timesteps: {CONFIG['total_timesteps']:,}")
    print(f"  Season:    {CONFIG['season']}")
    print(f"  Gamma:     {CONFIG['gamma']}  (discount factor)")
    print(f"  Buffer:    {CONFIG['buffer_size']:,}  (replay buffer size)")
    print()

    # Create directories
    os.makedirs(CONFIG["model_dir"], exist_ok=True)
    os.makedirs(CONFIG["log_dir"], exist_ok=True)

    if not SB3_AVAILABLE:
        print("stable-baselines3 not available. Showing config only.")
        print("\nTo train, install dependencies:")
        print("  pip install stable-baselines3[extra] gymnasium")
        print("\nSAC CONFIG that will be used:")
        for k, v in CONFIG.items():
            print(f"  {k}: {v}")
        return None