# Joint Dynamic Pricing & Anti-Scalping RL (IPL Ticketing Simulation)

Reinforcement learning agents learn to **jointly adjust ticket prices and per-user purchase limits** in a simulated high-demand ticketing market, while a **behavioral suspicion score** (heuristic + optional Isolation Forest) feeds the state so policies can react to scalper-like traffic.

This repository accompanies the academic report: *Reinforcement Learning for Joint Dynamic Pricing and Anti-Scalping Control in IPL Ticketing Systems*.

---

## Overview

- **Environment**: Custom `gymnasium` MDP with launch/mid/closing demand, genuine vs bot users, optional adaptive scalpers, and multi-objective reward (revenue, fairness, scalper penalty, dissatisfaction, plus inventory urgency).
- **Agents**: Rainbow-style **DQN** (dueling + double + PER + n-step + NoisyNet) and recurrent **PPO** (LSTM actor-critic, GAE).
- **Baselines**: Static pricing and a rule-based dynamic controller.
- **Artifacts**: Training dashboards, evaluation tables, action heatmaps, match-day logs, and a small Flask gallery.

For a **line-by-line, viva-oriented explanation** of every module, see **[VIVA_README.md](./VIVA_README.md)**.

---

## Features

- 6D observation: inventory, time, demand, suspicion \(S'\), normalized purchase limit, normalized price.
- 9 discrete joint actions: 3 price moves × 3 limit moves.
- Hybrid suspicion scoring: interpretable weights + unsupervised anomaly detection (`scikit-learn` Isolation Forest).
- Evaluation vs static and rule-based policies; optional hyperparameter sweep and DQN ablation script.

---

## Requirements

- Python 3.10+ recommended  
- Install dependencies (adjust for your CUDA/CPU PyTorch build):

```bash
pip install gymnasium numpy matplotlib torch scipy scikit-learn flask
```

On some systems, PyTorch + NumPy/OpenMP may need:

```bash
# Linux / macOS — avoids occasional BLAS thread conflicts
export OMP_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE
```

---

## Quick start

### 1. Train agents

```bash
python train.py --agent both
```

Optional: `--episodes N`, `--seeds K` for multiple seeds (confidence bands on curves).

Checkpoints and plots are written under `results/`.

### 2. Evaluate and compare

```bash
python evaluate.py --agent both --n_eval 100
```

Produces `results/eval_summary.txt`, `comparison_bar_chart.png`, and action heatmaps for trained agents.

### 3. Match-day simulation (demo logs)

```bash
python simulate_matchday.py --agent all
```

### 4. Hyperparameter sweep (DQN-focused)

```bash
python evaluate.py --hparam
```

### 5. Suspicion module demo

```bash
python suspicion_score.py
```

### 6. Optional ablation (DQN variants)

```bash
python ablation_study.py --episodes 400
```

### 7. Results dashboard (local)

```bash
python app.py
```

Open **http://127.0.0.1:5000** — serves `eval_summary` and PNGs from `results/`.

---

## Project structure

| Path | Description |
|------|-------------|
| `environment.py` | `IPLTicketingEnv` — transitions, reward, suspicion EMA, adaptive bots |
| `suspicion_score.py` | `UserBehavior`, heuristic scorer, Isolation Forest, ensemble |
| `agent_dqn.py` | Rainbow-style DQN agent |
| `agent_ppo.py` | Recurrent PPO (LSTM) agent |
| `baseline_agents.py` | `StaticAgent`, `RuleBasedAgent` |
| `train.py` | Training loops, dashboards |
| `evaluate.py` | Benchmarking, plots, optional `--hparam` |
| `simulate_matchday.py` | Narrative simulation + analytics PNG |
| `ablation_study.py` | DQN ablation experiment |
| `app.py` | Flask app |
| `templates/index.html` | Dashboard template |
| `results/` | Checkpoints, logs, figures (generated) |
| `VIVA_README.md` | Detailed theory + file map for oral exams |

---

## Method (short)

**State**: normalized inventory, time, demand sample, suspicion score, limit level, price.  
**Action**: discretized joint price and purchase-limit control.  
**Reward**: weighted sum of revenue, fairness (tickets to genuine users), scalper penalty, price dissatisfaction penalty, and late unsold-inventory penalty.

Full MDP and implementation notes: **[VIVA_README.md](./VIVA_README.md)**.

---

## Results

After `evaluate.py`, see `results/eval_summary.txt` and `comparison_bar_chart.png`. Metrics include total reward, revenue, **fairness index** (share of tickets to genuine users), **scalper rate**, and average price. Interpret all numbers in the context of this **simulator**, not live IPL data.

---

## License

No license file is included in this repository. Add a `LICENSE` if you plan to publish publicly.

---