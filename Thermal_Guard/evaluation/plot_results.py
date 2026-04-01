"""
plot_results.py
===============
Generate all evaluation plots for portfolio and interviews.

THE 4 PLOTS THIS CREATES:
    1. Training curve      → shows the agent is learning
    2. PUE comparison      → headline result (RL vs baselines)
    3. Temperature timeline → shows safety constraint satisfaction
    4. Action analysis     → shows what the agent learned to do

Run AFTER training:
    python evaluation/plot_results.py
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment.datacenter_env import DataCentreEnv
from training.baselines import RuleBasedAgent, PIDAgent

