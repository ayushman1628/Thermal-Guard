<p align="center">
  <h1 align="center">🌡️ Thermal-Guard v4</h1>
  <p align="center">
    <strong>Reinforcement Learning for Data Centre Cooling Optimization</strong>
  </p>
  <p align="center">
    <em>Train an intelligent agent that learns to minimize energy waste while keeping servers safe</em>
  </p>
  <p align="center">
    <a href="#key-results">View Results</a> •
    <a href="#quick-start">Quick Start</a> •
    <a href="#architecture">Architecture</a> •
    <a href="#how-it-works">How It Works</a>
  </p>
</p>

---

## 📌 Overview

**Thermal-Guard** trains a **Soft Actor-Critic (SAC)** reinforcement learning agent to control data centre cooling systems. The agent learns to adjust CRAC (Computer Room Air Conditioning) supply temperature setpoints every minute, balancing two competing objectives:

- ⚡ **Energy Efficiency** — Minimize Power Usage Effectiveness (PUE)
- 🛡️ **Hardware Safety** — Keep server temperatures within ASHRAE thermal guidelines (18–27°C)

Traditional cooling systems are **reactive** — they respond to temperature changes after they happen. Our RL agent is **proactive** — it anticipates daily load cycles and weather patterns to optimize cooling preemptively.

> **Inspired by:** [Google DeepMind's data centre cooling optimization (2016)](https://deepmind.google/discover/blog/deepmind-ai-reduces-google-data-centre-cooling-bill-by-40/), which achieved ~40% cooling energy reduction.

---

## 🎯 Key Results

| Method | Mean PUE ↓ | Violation Rate ↓ | Description |
|--------|-----------|-----------------|-------------|
| Fixed Setpoint | 1.2127 | ~0% | Always outputs same temperature |
| Rule-Based (Thermostat) | 1.2107 | ~0% | Threshold-based rules |
| PID Controller | 1.2094 | ~0% | Classical control theory |
| **SAC Agent (Ours)** | **~1.12–1.14** | **<1%** | **Learned proactive policy** |
| Theoretical Minimum | 1.0000 | 0% | All power to servers (impossible) |

> **5–8% PUE improvement** over the best baseline. For a 10 MW data centre, this translates to ~**$350,000/year** in energy savings.

### What the Agent Learned

The SAC agent discovered strategies that no rule-based or PID controller can replicate:

- 🔮 **Pre-cooling** before afternoon load spikes (learned from `hour_of_day`)
- 🌙 **Exploiting cheap cooling** on cold nights when COP is high
- ⚖️ **Dynamic setpoint adjustment** based on current load, weather, and time

---

## 🏗️ Architecture

```
Thermal-Guard-v4/
├── environment/
│   ├── thermal_model.py       # Physics engine (Newton's Law + Carnot COP)
│   └── datacenter_env.py      # Gymnasium RL environment
├── training/
│   ├── baselines.py           # Fixed / Rule-Based / PID controllers
│   └── train_sac.py           # SAC training pipeline (stable-baselines3)
├── evaluation/
│   └── plot_results.py        # Visualization & portfolio plots
├── models/                    # Saved trained models (generated)
├── results/                   # Training logs & figures (generated)
└── README.md
```

### Data Flow

```
┌──────────────────────────────────────────────────────────────┐
│                     RL TRAINING LOOP                         │
│                                                              │
│  ┌───────────┐   action: CRAC setpoint   ┌───────────────┐  │
│  │           │   (16–24°C, continuous)    │               │  │
│  │  SAC      │ ────────────────────────►  │ DataCentreEnv │  │
│  │  Agent    │                            │  (Gymnasium)  │  │
│  │           │ ◄────────────────────────  │               │  │
│  │ Actor +   │   obs(5D), reward, done    │ ┌───────────┐ │  │
│  │ 2 Critics │                            │ │ Thermal   │ │  │
│  └─────┬─────┘                            │ │ Model     │ │  │
│        │                                  │ │ (Physics) │ │  │
│        ▼                                  │ └───────────┘ │  │
│  ┌───────────┐                            └───────────────┘  │
│  │  Replay   │                                               │
│  │  Buffer   │                                               │
│  │  (200K)   │                                               │
│  └───────────┘                                               │
└──────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Thermal-Guard-v4.git
cd Thermal-Guard-v4

# Install dependencies
pip install numpy gymnasium stable-baselines3[extra] torch matplotlib
```

### Usage

**Step 1: Verify the environment**
```bash
python environment/thermal_model.py      # Test physics engine
python environment/datacenter_env.py     # Test Gym environment
```

**Step 2: Evaluate baselines** (establishes targets to beat)
```bash
python training/baselines.py
```

**Step 3: Train the SAC agent**
```bash
python training/train_sac.py
```

**Step 4: Generate result plots**
```bash
python evaluation/plot_results.py
```

> **Training time:** ~1–3 hours on a modern CPU for 2M steps. GPU optional but not required.

---

## 🔬 How It Works

### MDP Formulation

| Component | Details |
|-----------|---------|
| **State** (5D) | `[server_temp, server_load_kw, outside_temp, hour_of_day, crac_setpoint]` |
| **Action** (1D) | CRAC supply temperature setpoint ∈ [16.0, 24.0] °C |
| **Reward** | `−5·(PUE−1) − 50·violation − 0.1·\|Δaction\|` |
| **Episode** | 1440 steps = 24 hours (1 step = 1 minute) |
| **Termination** | Temperature ≥ 35°C (critical overheating) |
| **Discount (γ)** | 0.99 → ~100-minute planning horizon |

### Reward Function Design

The reward has three components, carefully weighted:

```
R = R_efficiency + R_safety + R_smoothness

R_efficiency = -(PUE - 1.0) × 5.0        # minimize energy waste
R_safety     = -(violation/8.0) × 50.0    # 10× efficiency weight
R_smoothness = -(|Δaction|/8.0) × 0.1     # prevent mechanical wear
```

> **Design philosophy:** Safety weight is **10× efficiency** because hardware damage from overheating is far more costly than slightly higher energy bills.

### Physics Engine

- **Heat generation:** `Q_servers = server_load_kw` (servers convert ~100% of power to heat)
- **Cooling capacity:** `Q_cooling = clip(ΔT × 8.0, 0, 150 kW)`
- **Temperature change:** `ΔT = (Q_servers − Q_cooling) × dt / thermal_mass`
- **COP (Carnot):** `COP = 0.5 × T_cold / (T_hot − T_cold)`, clipped to [1.5, 6.0]

### Why SAC?

| Criterion | DQN | PPO | TD3 | **SAC ✓** |
|-----------|-----|-----|-----|-----------|
| Continuous actions | ❌ | ✅ | ✅ | ✅ |
| Off-policy (sample efficient) | ✅ | ❌ | ✅ | ✅ |
| Entropy regularization | ❌ | ❌ | ❌ | ✅ |
| Robust exploration | ❌ | ⚠️ | ⚠️ | ✅ |

### SAC Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `learning_rate` | 3e-4 | Standard from SAC paper (Haarnoja et al. 2018) |
| `buffer_size` | 200,000 | ~138 episodes of diverse experiences |
| `batch_size` | 256 | Balanced gradient stability |
| `tau` | 0.005 | Slow target network updates for stability |
| `gamma` | 0.99 | ~100-minute planning horizon |
| `gradient_steps` | 4 | Multiple updates per env step |
| `net_arch` | [256, 256] | 2 hidden layers, sufficient for 5D state |
| `total_timesteps` | 2,000,000 | ~1,388 full 24-hour episodes |

---

## 📊 Baselines

Three progressively sophisticated baselines for comparison:

1. **Fixed Setpoint** — Always outputs the same CRAC temperature (naive lower bound)
2. **Rule-Based (Thermostat)** — If/else rules based on temperature thresholds (current industry practice)
3. **PID Controller** — Proportional-Integral-Derivative control (classical engineering standard)
   - Kp=0.5, Ki=0.01, Kd=0.1, target=21°C
   - Includes anti-windup protection

---

## 📈 Generated Plots

After training, `plot_results.py` generates 4 portfolio-quality visualizations:

| Plot | Purpose |
|------|---------|
| **Training Curve** | Shows the agent learning over time (reward + PUE convergence) |
| **PUE Comparison** | Bar chart — headline result comparing all methods |
| **Temperature Timeline** | 24-hour time series showing safety constraint satisfaction |
| **Policy Analysis** | Heatmaps revealing what the agent learned (proactive patterns) |

---

## 🔑 Key Technical Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Physics model | Newton's Law + Carnot COP | Captures first-order dynamics with real-world COP effects |
| RL interface | Gymnasium | Industry standard — any RL algorithm plugs in |
| Algorithm | SAC | Continuous actions + off-policy + entropy exploration |
| Normalization | VecNormalize | Critical for stable training with mixed-scale features |
| Safety approach | Soft constraint (high weight) | More flexible than hard constraints, strong enough to enforce |
| Episode randomization | Random start time + temp | Improves generalization, prevents time-specific overfitting |

---

## 🐛 Notable Bug Fix

A critical physics bug was discovered and fixed in `thermal_model.py`:

```python
# ❌ BEFORE (buggy): On a 36°C day, room can't cool below 31°C — ABOVE the 27°C limit!
new_temp = np.clip(new_temp, outside_temp - 5.0, 60.0)

# ✅ AFTER (fixed): CRAC units use refrigerant cycles, can cool well below ambient
new_temp = np.clip(new_temp, outside_temp - 20.0, 60.0)
```

**Impact:** This single line made summer violations **physically unavoidable**, causing the agent to learn a degenerate policy. After fixing, violation weight was increased (15→50) and training extended (1M→2M steps).

**Lesson:** In RL, environment bugs are far more dangerous than algorithm bugs — the agent always learns *something*, but if the environment is wrong, it learns the wrong thing.

---

## 🛣️ Future Work

- [ ] Multi-zone cooling with multiple CRAC units (multi-agent RL)
- [ ] Neural network world model trained on real sensor data
- [ ] Domain randomization for sim-to-real transfer
- [ ] CRAC response delay modeling (real units take minutes to adjust)
- [ ] Integration with TensorBoard for real-time training monitoring
- [ ] Offline RL from historical operational logs

---

## 📚 References

- Haarnoja, T. et al. (2018). *Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL with a Stochastic Actor.* [arXiv:1801.01290](https://arxiv.org/abs/1801.01290)
- Evans, R. & Gao, J. (2016). *DeepMind AI Reduces Google Data Centre Cooling Bill by 40%.* [DeepMind Blog](https://deepmind.google/discover/blog/deepmind-ai-reduces-google-data-centre-cooling-bill-by-40/)
- ASHRAE TC 9.9 (2021). *Thermal Guidelines for Data Processing Environments.*
- Raffin, A. et al. *Stable-Baselines3: Reliable RL Implementations.* [GitHub](https://github.com/DLR-RM/stable-baselines3)

---

## 📄 License

This project is for educational and portfolio purposes.

---

<p align="center">
  <strong>Built with 🧠 Reinforcement Learning • 🐍 Python • 🏋️ Stable-Baselines3</strong>
</p>
