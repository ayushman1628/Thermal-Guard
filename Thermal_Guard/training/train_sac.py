
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

    # ── ENVIRONMENT SETUP ────────────────────────────────────────────
    print("Step 1: Validating environment...")
    env = make_env(seed=0)
    check_env(env.env if hasattr(env, 'env') else env)
    print("  ✅ Environment valid\n")

    eval_env = DataCentreEnv(season=CONFIG["season"])

    # ── CALLBACKS ────────────────────────────────────────────────────
    print("Step 2: Setting up callbacks...")

    # EvalCallback: evaluates agent periodically, saves best model
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=CONFIG["model_dir"],
        log_path=CONFIG["log_dir"],
        eval_freq=CONFIG["eval_freq"],
        n_eval_episodes=CONFIG["n_eval_episodes"],
        deterministic=True,     # no exploration noise during eval
        render=False,
        verbose=1,
    )

    # CheckpointCallback: saves model every N steps (resume if crash)
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=CONFIG["model_dir"],
        name_prefix="sac_checkpoint",
        verbose=0,
    )

    # Custom metrics callback
    metrics_callback = DataCentreMetricsCallback(verbose=0)

    from stable_baselines3.common.callbacks import CallbackList
    all_callbacks = CallbackList([
        eval_callback,
        checkpoint_callback,
        metrics_callback,
    ])
    print("  ✅ Callbacks ready\n")

    # ── MODEL CREATION ───────────────────────────────────────────────
    print("Step 3: Creating SAC model...")
    model = SAC(
        policy="MlpPolicy",         # Multi-Layer Perceptron policy
                                    # (neural network for both actor + critic)
        env=env,
        learning_rate=CONFIG["learning_rate"],
        buffer_size=CONFIG["buffer_size"],
        learning_starts=CONFIG["learning_starts"],
        batch_size=CONFIG["batch_size"],
        tau=CONFIG["tau"],
        gamma=CONFIG["gamma"],
        train_freq=CONFIG["train_freq"],
        gradient_steps=CONFIG["gradient_steps"],
        verbose=1,
        tensorboard_log=os.path.join(CONFIG["log_dir"], "tensorboard"),
        seed=42,
    )

    # Print network architecture
    print(f"\n  Actor network:  {model.actor}")
    print(f"\n  Critic network: {model.critic}\n")

    # ── TRAINING ─────────────────────────────────────────────────────
    print("Step 4: Training...\n")
    start_time = datetime.now()

    model.learn(
        total_timesteps=CONFIG["total_timesteps"],
        callback=all_callbacks,
        log_interval=4,         # log every 4 episodes
        progress_bar=True,
    )

    duration = datetime.now() - start_time
    print(f"\n  ✅ Training complete in {duration}")

    # ── SAVE FINAL MODEL ─────────────────────────────────────────────
    final_path = os.path.join(CONFIG["model_dir"], "sac_final")
    model.save(final_path)
    print(f"  ✅ Final model saved to {final_path}.zip")

    # Save config alongside model
    with open(os.path.join(CONFIG["model_dir"], "sac_config.json"), "w") as f:
        json.dump(CONFIG, f, indent=2)

    return model, metrics_callback


def evaluate_and_compare(model=None):
    """
    Compare SAC agent against all baselines.
    This produces the results table for your portfolio/interviews.
    """
    print("\n" + "=" * 65)
    print("  Final Evaluation: SAC vs Baselines")
    print("=" * 65)

    env = DataCentreEnv(season=CONFIG["season"])
    N_EVAL = 10

    # Evaluate baselines
    agents = {
        "Rule-Based Controller":  RuleBasedAgent(),
        "PID Controller":         PIDAgent(target_temp=21.0),
    }

    all_results = {}
    for name, agent in agents.items():
        print(f"\nEvaluating: {name}...")
        results = evaluate_agent(agent, env, n_episodes=N_EVAL, seed=100)
        all_results[name] = results
        print_results(name, results)

    # Evaluate SAC (if available)
    if model is not None:
        class SB3Wrapper:
            """Wrapper so SB3 model works with our evaluate_agent function."""
            def __init__(self, m): self.m = m
            def predict(self, obs):
                action, _ = self.m.predict(obs, deterministic=True)
                return action

        print(f"\nEvaluating: SAC Agent...")
        sac_results = evaluate_agent(SB3Wrapper(model), env, n_episodes=N_EVAL, seed=100)
        all_results["SAC Agent (Ours)"] = sac_results
        print_results("SAC Agent (Ours)", sac_results)
    else:
        print("\n  (SAC not trained — load model to evaluate)")

    # ── RESULTS TABLE ────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  FINAL RESULTS TABLE")
    print(f"{'='*65}")
    print(f"  {'Method':<28} {'Mean PUE':>10} {'Violations':>12} {'Reward':>12}")
    print(f"  {'-'*63}")

    for name, r in all_results.items():
        marker = " ← ours" if "SAC" in name else ""
        print(
            f"  {name:<28} "
            f"{r['mean_pue']:>10.4f} "
            f"{r['violation_rate_pct']:>10.2f}% "
            f"{r['mean_episode_reward']:>10.1f}"
            f"{marker}"
        )

    if "SAC Agent (Ours)" in all_results and "Rule-Based Controller" in all_results:
        sac_pue  = all_results["SAC Agent (Ours)"]["mean_pue"]
        base_pue = all_results["Rule-Based Controller"]["mean_pue"]
        improvement = (base_pue - sac_pue) / base_pue * 100
        print(f"\n  PUE improvement over rule-based: {improvement:.1f}%")

    return all_results

# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = train_sac()

    if result is not None:
        model, metrics_cb = result
        all_results = evaluate_and_compare(model)

        # Save results to JSON
        results_path = os.path.join(CONFIG["log_dir"], "final_results.json")
        serializable = {k: {mk: float(mv) for mk, mv in v.items()}
                        for k, v in all_results.items()}
        with open(results_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"\n  Results saved to {results_path}")
        print("\n  NEXT STEP: python evaluation/plot_results.py")
    else:
        # Demo mode — run comparison without trained RL
        evaluate_and_compare(model=None)
