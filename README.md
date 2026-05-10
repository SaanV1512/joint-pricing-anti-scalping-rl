# IPL RL Ticketing System — Run Guide

## Quick Start (Copy-Paste These Commands)

### Step 1: Train Both Agents
```bash
cd /Users/shivanshshah/Documents/rl
OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE python3 train.py --agent both
```

### Step 2: Evaluate & Compare All 4 Agents
```bash
OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE python3 evaluate.py --agent both
```

### Step 3: Run Live Match Day Simulation
```bash
OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE python3 simulate_matchday.py --agent all
```

### Step 4: Hyperparameter Analysis
```bash
OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE python3 evaluate.py --hparam
```

### Step 5: Test Suspicion Score Module
```bash
python3 suspicion_score.py
```

---

## Project Structure

```
rl/
├── environment.py          ← Saanvi: Gym environment, demand spikes, user simulation
├── suspicion_score.py      ← Veda:   Exact S = w1f1+w2f2+w3f3+w4f4 formulation
├── baseline_agents.py      ← Veda:   Static + Rule-Based baselines for comparison
├── agent_dqn.py            ← Shivansh: DQN with replay buffer + target network
├── agent_ppo.py            ← Shivansh: PPO with GAE + clipped surrogate
├── train.py                ← Shivansh: Training loops for DQN and PPO
├── evaluate.py             ← Veda:   4-agent comparison + metrics + plots
├── simulate_matchday.py    ← Saanvi: Vivid match-day log + analytics dashboard
├── PAPER_ALL_SECTIONS.md   ← All team: Complete paper sections II–IX
└── results/
    ├── dqn_checkpoint.pth
    ├── ppo_checkpoint.pth
    ├── dqn_training_curves.png
    ├── ppo_training_curves.png
    ├── comparison_bar_chart.png
    ├── dqn_action_heatmap.png
    ├── ppo_action_heatmap.png
    ├── matchday_analytics_dqn.png
    ├── matchday_analytics_ppo.png
    ├── matchday_analytics_static.png
    ├── matchday_analytics_rule_based.png
    ├── matchday_log_dqn.txt
    ├── matchday_log_ppo.txt
    └── eval_summary.txt
```

---

## Who Wrote What

| File | Owner |
|------|-------|
| `environment.py` | Saanvi |
| `suspicion_score.py` | Veda |
| `baseline_agents.py` | Veda |
| `evaluate.py` | Veda |
| `agent_dqn.py` | Shivansh |
| `agent_ppo.py` | Shivansh |
| `train.py` | Shivansh |
| `simulate_matchday.py` | Saanvi |
| Sections III & IV of paper | Rihan & Shivansh |
| Sections II, VI, VII of paper | Veda |
| Section V of paper | Saanvi |
