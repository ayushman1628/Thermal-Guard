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
import environment.datacenter_env
from training.baselines import RuleBasedAgent, PIDAgent


# ── PLOT STYLE ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0f1117",
    "axes.facecolor":    "#1a1d27",
    "axes.edgecolor":    "#3a3d4a",
    "text.color":        "#e8eaf0",
    "axes.labelcolor":   "#b8bcc8",
    "xtick.color":       "#b8bcc8",
    "ytick.color":       "#b8bcc8",
    "grid.color":        "#2a2d3a",
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "font.family":       "monospace",
    "axes.titlesize":    11,
    "axes.labelsize":    9,
    "figure.titlesize":  13,
})

COLORS = {
    "sac":       "#00d4ff",   # cyan  — SAC agent
    "rule":      "#ff6b6b",   # red   — rule-based
    "pid":       "#ffa500",   # orange — PID
    "safe_zone": "#00ff8888", # green transparent — safe temp zone
    "danger":    "#ff4444",   # red   — violations
    "bg":        "#0f1117",
    "accent":    "#7c4dff",   # purple
}

OUTPUT_DIR = "results/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────
# PLOT 1: TRAINING CURVE (simulated — shows what to expect)
# ─────────────────────────────────────────────────────────────────────────

def plot_training_curve(save=True):
    """
    Plot episode reward over training timesteps.

    In real training, this comes from stable-baselines3's monitor logs.
    Here we simulate a realistic learning curve to show the expected shape.

    Shape to expect:
        - Initial plateau: agent exploring randomly
        - Rapid improvement: agent discovers good strategies
        - Convergence: policy stabilizes
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("SAC Training Progress", fontweight="bold", color="white", y=1.02)

    timesteps = np.linspace(0, 300_000, 500)

    # Simulate realistic SAC learning curve
    # Real curve will come from: results/evaluations.npz after training
    def learning_curve(x, noise_scale=0.15):
        # Exploration phase → rapid learning → convergence
        progress = 1 - np.exp(-x / 60_000)
        baseline_reward = -24000  # random agent level
        optimal_reward = -6000  # near-optimal level
        signal = baseline_reward + (optimal_reward - baseline_reward) * progress
        noise = np.random.normal(0, abs(signal) * noise_scale, size=len(x))
        # Smooth the noise
        window = 20
        smoothed = np.convolve(signal + noise, np.ones(window) / window, mode='same')
        return smoothed, signal

    np.random.seed(42)
    raw, smooth = learning_curve(timesteps)

    # Subplot 1: Episode reward
    ax1 = axes[0]
    ax1.plot(timesteps, raw, color=COLORS["sac"], alpha=0.25, linewidth=0.8, label="Per-episode")
    ax1.plot(timesteps, smooth, color=COLORS["sac"], linewidth=2.5, label="Smoothed (window=20)")
    ax1.axhline(y=-24000, color=COLORS["rule"], linestyle="--", linewidth=1.5, label="Random agent")
    ax1.axhline(y=-12000, color=COLORS["pid"], linestyle="--", linewidth=1.5, label="PID baseline")
    ax1.set_xlabel("Training Timesteps")
    ax1.set_ylabel("Episode Reward")
    ax1.set_title("Episode Reward vs Training Steps")
    ax1.legend(fontsize=8)
    ax1.grid(True)
    ax1.set_xlim(0, 300_000)

    # Subplot 2: PUE over training
    ax2 = axes[1]
    # PUE improves as training progresses
    pue_curve = 1.22 - 0.10 * (1 - np.exp(-timesteps / 70_000))
    pue_noise = pue_curve + np.random.normal(0, 0.008, size=len(timesteps))
    window = 20
    pue_smooth = np.convolve(pue_noise, np.ones(window) / window, mode='same')

    ax2.plot(timesteps, pue_noise, color=COLORS["sac"], alpha=0.25, linewidth=0.8)
    ax2.plot(timesteps, pue_smooth, color=COLORS["sac"], linewidth=2.5, label="SAC (smoothed)")
    ax2.axhline(y=1.21, color=COLORS["rule"], linestyle="--", linewidth=1.5, label="Rule-Based (1.2107)")
    ax2.axhline(y=1.21, color=COLORS["pid"], linestyle=":", linewidth=1.5, label="PID (1.2094)")
    ax2.axhline(y=1.0, color="#00ff88", linestyle="--", linewidth=1, label="Theoretical min (1.0)", alpha=0.6)
    ax2.set_xlabel("Training Timesteps")
    ax2.set_ylabel("Mean PUE")
    ax2.set_title("PUE vs Training Steps\n(lower = more energy efficient)")
    ax2.legend(fontsize=8)
    ax2.grid(True)
    ax2.set_ylim(1.05, 1.30)

    plt.tight_layout()

    note = ("Note: This shows the expected learning curve shape.\n"
            "After training, replace with actual data from results/evaluations.npz")
    fig.text(0.5, -0.04, note, ha='center', fontsize=7, color='#888888', style='italic')

    if save:
        path = os.path.join(OUTPUT_DIR, "01_training_curve.png")
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
        print(f"  ✅ Saved: {path}")
    plt.close()

