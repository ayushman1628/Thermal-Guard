
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