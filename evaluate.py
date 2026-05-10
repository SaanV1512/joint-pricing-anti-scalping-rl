from __future__ import annotations

import argparse
import os
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from environment     import IPLTicketingEnv, N_ACTIONS
from agent_dqn       import DQNAgent
from agent_ppo       import PPOAgent
from baseline_agents import StaticAgent, RuleBasedAgent

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Episode Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_episode(env: IPLTicketingEnv, policy, agent_type: str) -> Dict:
    state, _ = env.reset()
    if agent_type == "ppo":
        policy.reset_hidden()
    total_reward = total_revenue = fair_tickets = scalper_tickets = 0.0
    action_counts = np.zeros(N_ACTIONS, dtype=int)
    price_history = []
    susp_history  = []

    done = False
    while not done:
        if agent_type == "dqn":
            action = policy.select_action(state, training=False)
        elif agent_type == "ppo":
            action, _, _ = policy.collect_step(state)
        elif agent_type in ("static", "rule_based"):
            action = policy.select_action(state)
        else:
            raise ValueError(f"Unknown agent_type: {agent_type}")

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_reward    += reward
        total_revenue   += info["revenue"]
        fair_tickets    += info["fair_tickets"]
        scalper_tickets += info["scalper_tickets"]
        action_counts[action] += 1
        price_history.append(info["price"])
        susp_history.append(info["suspicion"])
        state = next_state

    total_sold = fair_tickets + scalper_tickets + 1e-8
    return {
        "total_reward":    total_reward,
        "total_revenue":   total_revenue,
        "fair_tickets":    fair_tickets,
        "scalper_tickets": scalper_tickets,
        "scalper_rate":    scalper_tickets / total_sold,
        "fairness_index":  fair_tickets    / total_sold,
        "mean_price":      float(np.mean(price_history)),
        "mean_suspicion":  float(np.mean(susp_history)),
        "action_counts":   action_counts,
    }


def evaluate_agent(policy, agent_type: str, n_eval: int = 50, seed_offset: int = 100) -> Dict:
    all_m: List[Dict] = []
    for i in range(n_eval):
        env = IPLTicketingEnv(seed=seed_offset + i)
        all_m.append(run_episode(env, policy, agent_type))

    keys = ["total_reward", "total_revenue", "fair_tickets", "scalper_tickets",
            "scalper_rate", "fairness_index", "mean_price", "mean_suspicion"]
    summary = {k: float(np.mean([m[k] for m in all_m])) for k in keys}
    summary["std_reward"] = float(np.std([m["total_reward"] for m in all_m]))

    agg = np.zeros(N_ACTIONS, dtype=float)
    for m in all_m:
        agg += m["action_counts"]
    summary["action_dist"] = agg / agg.sum()
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison_bar(results: Dict[str, Dict]):
    """Grouped bar chart comparing all agents on 4 key metrics."""
    agents  = list(results.keys())
    metrics = {
        "Total Reward":   [results[a]["total_reward"]    for a in agents],
        "Fairness Index": [results[a]["fairness_index"]  for a in agents],
        "Scalper Rate":   [results[a]["scalper_rate"]    for a in agents],
        "Revenue (k₹)":  [results[a]["total_revenue"] / 1000 for a in agents],
    }

    palette = ["#BDC3C7", "#E67E22", "#3498DB", "#2ECC71"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("3-Way Agent Comparison (50 Episodes Each)", fontsize=14, fontweight="bold")

    for ax, (metric_name, vals), colour in zip(axes, metrics.items(), palette):
        bars = ax.bar(agents, vals, color=colour, edgecolor="white", linewidth=1.2)
        ax.set_title(metric_name, fontweight="bold")
        ax.set_xticklabels(agents, rotation=20, ha="right", fontsize=9)
        ax.grid(axis="y", alpha=0.35)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01 * max(vals),
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "comparison_bar_chart.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Comparison chart  → {out}")


def plot_action_heatmap(action_dist: np.ndarray, label: str):
    grid = action_dist.reshape(3, 3)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(grid, cmap="YlOrRd", vmin=0, vmax=grid.max())
    plt.colorbar(im, ax=ax, label="Action probability")

    price_labels = ["Increase price", "Decrease price", "Keep price"]
    limit_labels = ["Tighten limit",  "Relax limit",    "Maintain limit"]
    ax.set_xticks(range(3)); ax.set_xticklabels(limit_labels, rotation=20, ha="right")
    ax.set_yticks(range(3)); ax.set_yticklabels(price_labels)

    for r in range(3):
        for c in range(3):
            ax.text(c, r, f"{grid[r,c]:.2f}", ha="center", va="center",
                    color="black" if grid[r,c] < grid.max() * 0.7 else "white", fontsize=11)

    ax.set_title(f"{label} — Joint Action Distribution", fontweight="bold")
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, f"{label.lower().replace(' ', '_')}_action_heatmap.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Action heatmap    → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_table(results: Dict[str, Dict]):
    header = (
        f"{'Agent':<22} {'Reward':>9} {'Revenue':>13} "
        f"{'FairnessIdx':>12} {'ScalperRate':>12} {'AvgPrice':>10}"
    )
    sep = "─" * len(header)
    lines = ["\n" + sep, header, sep]
    for name, m in results.items():
        lines.append(
            f"{name:<22} "
            f"{m['total_reward']:>9.3f} "
            f"₹{m['total_revenue']:>12,.1f} "
            f"{m['fairness_index']:>12.3f} "
            f"{m['scalper_rate']:>12.3f} "
            f"₹{m['mean_price']:>9.1f}"
        )
    lines.append(sep)
    text = "\n".join(lines)
    print(text)

    out = os.path.join(RESULTS_DIR, "eval_summary.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\n  Summary saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameter sensitivity sweep (Veda — Section VII)
# ─────────────────────────────────────────────────────────────────────────────

def hyperparameter_sensitivity():
    configs = {
        "lr=1e-4":           {"lr": 1e-4},
        "lr=1e-3 (default)": {"lr": 1e-3},
        "lr=5e-3":           {"lr": 5e-3},
        "γ=0.90":            {"gamma": 0.90},
        "γ=0.99 (default)":  {"gamma": 0.99},
        "batch=32":          {"batch_size": 32},
        "batch=128":         {"batch_size": 128},
        "ε-decay=5k":        {"eps_decay": 5_000},
        "ε-decay=20k (def)": {"eps_decay": 20_000},
    }

    print("\n  Running DQN hyperparameter sensitivity sweep (100 episodes each)…")
    results = {}
    for name, kwargs in configs.items():
        env   = IPLTicketingEnv(seed=0)
        agent = DQNAgent(**kwargs)
        rewards = []
        state, _ = env.reset()
        ep_r, ep = 0.0, 0
        while ep < 100:
            a = agent.select_action(state, training=True)
            ns, r, term, trunc, _ = env.step(a)
            done = term or trunc
            agent.store(state, a, r, ns, done)
            agent.update()
            ep_r += r
            state = ns
            if done:
                rewards.append(ep_r)
                ep_r = 0.0
                ep  += 1
                state, _ = env.reset()
        results[name] = float(np.mean(rewards[-20:]))
        print(f"    {name:<25} → mean_reward = {results[name]:+.4f}")

    names  = list(results.keys())
    values = [results[n] for n in names]
    colors = ["#3498DB" if "default" in n or "def)" in n else "#E74C3C" for n in names]
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.barh(names, values, color=colors, edgecolor="white")
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean Episode Reward (last 20 eps)")
    ax.set_title("DQN Hyperparameter Sensitivity (Blue = Chosen Default)", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "hyperparameter_analysis.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\n  Hyperparameter plot → {out}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent",  choices=["dqn", "ppo", "both"], default="both")
    parser.add_argument("--n_eval", type=int, default=100)
    parser.add_argument("--hparam", action="store_true")
    args = parser.parse_args()

    all_results: Dict[str, Dict] = {}

    # Baseline 1: Static
    print("\nEvaluating Static baseline …")
    all_results["Static"] = evaluate_agent(StaticAgent(), "static", n_eval=args.n_eval)

    # Baseline 2: Rule-based
    print("Evaluating Rule-Based baseline …")
    all_results["Rule-Based"] = evaluate_agent(RuleBasedAgent(), "rule_based", n_eval=args.n_eval)

    # DQN
    if args.agent in ("dqn", "both"):
        ckpt = os.path.join(RESULTS_DIR, "dqn_checkpoint.pth")
        if not os.path.exists(ckpt):
            print(f"[WARN] DQN checkpoint not found. Run train.py first.")
        else:
            print("Evaluating DQN …")
            dqn = DQNAgent(); dqn.load(ckpt)
            all_results["DQN"] = evaluate_agent(dqn, "dqn", n_eval=args.n_eval)
            plot_action_heatmap(all_results["DQN"]["action_dist"], "DQN")

    # PPO
    if args.agent in ("ppo", "both"):
        ckpt = os.path.join(RESULTS_DIR, "ppo_checkpoint.pth")
        if not os.path.exists(ckpt):
            print(f"[WARN] PPO checkpoint not found. Run train.py first.")
        else:
            print("Evaluating PPO …")
            ppo = PPOAgent(); ppo.load(ckpt)
            all_results["PPO"] = evaluate_agent(ppo, "ppo", n_eval=args.n_eval)
            plot_action_heatmap(all_results["PPO"]["action_dist"], "PPO")

    print_table(all_results)
    plot_comparison_bar(all_results)

    if args.hparam:
        hyperparameter_sensitivity()
