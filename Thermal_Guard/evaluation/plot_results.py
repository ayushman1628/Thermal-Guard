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

# ─────────────────────────────────────────────────────────────────────────
# PLOT 3: 24-HOUR TEMPERATURE TIMELINE
# ─────────────────────────────────────────────────────────────────────────

def plot_temperature_timeline(trained_model=None, save=True):
    """
    Compare temperature control over 24 hours: SAC vs Rule-Based vs PID.
    Shows whether agents keep temperature in ASHRAE safe zone (18–27°C).
    """
    env = DataCentreEnv(season="summer", initial_temp=22.0)
    agents = {
        "Rule-Based": (RuleBasedAgent(), COLORS["rule"]),
        "PID":        (PIDAgent(),       COLORS["pid"]),
    }

    if trained_model is not None:
        class SB3Wrapper:
            def _init_(self, m): self.m = m
            def predict(self, obs):
                a, _ = self.m.predict(obs, deterministic=True)
                return a
        agents["SAC (Ours)"] = (SB3Wrapper(trained_model), COLORS["sac"])

    # Run each agent for one episode
    trajectories = {}
    for name, (agent, color) in agents.items():
        if hasattr(agent, 'reset'):
            agent.reset()
        obs, _ = env.reset(seed=999)
        temps, loads, setpoints, pues = [], [], [], []
        for _ in range(1440):
            action = agent.predict(obs)
            obs, _, term, trunc, info = env.step(action)
            temps.append(info["server_temp"])
            loads.append(info["server_load_kw"])
            setpoints.append(info["crac_setpoint"])
            pues.append(info["pue"])
            if term or trunc:
                break
        trajectories[name] = {
            "temps": temps, "loads": loads,
            "setpoints": setpoints, "pues": pues,
            "color": color
        }

    # ── PLOT ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle("24-Hour Data Centre Control Comparison\n(Summer Day — Peak Cooling Challenge)",
                 fontweight="bold", color="white")

    gs = gridspec.GridSpec(3, 1, hspace=0.4)
    time_hours = np.arange(len(list(trajectories.values())[0]["temps"])) / 60.0

    # Panel 1: Server temperature
    ax1 = fig.add_subplot(gs[0])
    ax1.axhspan(18, 27, alpha=0.12, color="#00ff88", label="ASHRAE safe zone (18–27°C)")
    ax1.axhline(27, color="#00ff88", linestyle="--", linewidth=1, alpha=0.6)
    ax1.axhline(18, color="#00ff88", linestyle="--", linewidth=1, alpha=0.6)

    for name, traj in trajectories.items():
        ax1.plot(time_hours, traj["temps"], color=traj["color"],
                 linewidth=2 if "SAC" in name else 1.3,
                 alpha=1.0 if "SAC" in name else 0.75,
                 label=name, zorder=3 if "SAC" in name else 2)

    ax1.set_ylabel("Server Temp (°C)")
    ax1.set_title("Server Inlet Temperature")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True)
    ax1.set_xlim(0, 24)

    # Panel 2: CRAC setpoints
    ax2 = fig.add_subplot(gs[1])
    for name, traj in trajectories.items():
        ax2.plot(time_hours, traj["setpoints"], color=traj["color"],
                 linewidth=2 if "SAC" in name else 1.3,
                 alpha=1.0 if "SAC" in name else 0.75,
                 label=name)
    ax2.set_ylabel("CRAC Setpoint (°C)")
    ax2.set_title("CRAC Supply Temperature Setpoint (Agent Action)")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True)
    ax2.set_xlim(0, 24)
    ax2.set_ylim(15, 25)

    # Panel 3: PUE over the day
    ax3 = fig.add_subplot(gs[2])
    for name, traj in trajectories.items():
        ax3.plot(time_hours, traj["pues"], color=traj["color"],
                 linewidth=2 if "SAC" in name else 1.3,
                 alpha=1.0 if "SAC" in name else 0.75,
                 label=name)
    ax3.axhline(1.0, color="#00ff88", linestyle="--", linewidth=1, alpha=0.5, label="Ideal PUE = 1.0")
    ax3.set_ylabel("PUE")
    ax3.set_xlabel("Hour of Day")
    ax3.set_title("Power Usage Effectiveness (PUE)")
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(True)
    ax3.set_xlim(0, 24)

    # Add mean PUE annotations
    for i, (name, traj) in enumerate(trajectories.items()):
        mean_pue = np.mean(traj["pues"])
        violations = sum(1 for t in traj["temps"] if t > 27 or t < 18)
        ax3.text(0.5 + i * 7, ax3.get_ylim()[0] + 0.01,
                 f"{name}\nPUE={mean_pue:.3f}\nViol={violations}min",
                 fontsize=7, color=traj["color"], ha='center')

    plt.tight_layout()

    if save:
        path = os.path.join(OUTPUT_DIR, "03_temperature_timeline.png")
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
        print(f"  ✅ Saved: {path}")
    plt.close()

