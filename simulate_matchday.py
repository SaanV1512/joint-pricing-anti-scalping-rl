from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from environment      import IPLTicketingEnv, decode_action
from baseline_agents  import StaticAgent, RuleBasedAgent

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Colour helpers for terminal output
# ─────────────────────────────────────────────────────────────────────────────

class C:
    RED    = "\033[91m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    BLUE   = "\033[94m"
    CYAN   = "\033[96m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"


def clr(text: str, colour: str) -> str:
    return f"{colour}{text}{C.RESET}"


# ─────────────────────────────────────────────────────────────────────────────
# Main simulation
# ─────────────────────────────────────────────────────────────────────────────

def simulate(agent_label: str, policy, agent_type: str, seed: int = 99):
    env = IPLTicketingEnv(total_inventory=2000, match_duration=100,
                          base_price=1000.0, scalper_ratio=0.30, seed=seed)
    state, _ = env.reset()

    log_lines  = []
    price_hist = []
    susp_hist  = []
    inv_hist   = []
    rew_hist   = []

    banner = (
        f"\n{'='*65}\n"
        f"  IPL MATCH DAY SIMULATION  |  Agent: {agent_label.upper()}\n"
        f"{'='*65}\n"
        f"  Total Inventory : {env.total_inventory} tickets\n"
        f"  Base Price      : ₹{env.base_price:,.0f}\n"
        f"  Match Duration  : {env.match_duration} time-steps\n"
        f"  Scalper Ratio   : {env.scalper_ratio*100:.0f}% of traffic\n"
        f"{'='*65}"
    )
    print(banner)
    log_lines.append(banner)

    total_revenue  = 0.0
    total_fair     = 0.0
    total_scalpers = 0.0
    done           = False
    step           = 0

    while not done:
        # Get action
        if agent_type == "dqn":
            action = policy.select_action(state, training=False)
        elif agent_type == "ppo":
            action, _, _ = policy.net.act(state)
        elif agent_type == "static":
            action = policy.select_action(state)
        elif agent_type == "rule_based":
            action = policy.select_action(state)

        price_act, limit_act = decode_action(action)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Accumulate
        total_revenue  += info["revenue"]
        total_fair     += info["fair_tickets"]
        total_scalpers += info["scalper_tickets"]
        price_hist.append(info["price"])
        susp_hist.append(info["suspicion"])
        inv_hist.append(info["inventory"])
        rew_hist.append(reward)

        # Format log line
        time_pct   = (step / env.match_duration) * 100
        phase      = "LAUNCH RUSH" if time_pct < 15 else ("MID-SALE" if time_pct < 70 else "CLOSING")
        susp_level = "🔴 HIGH" if info["suspicion"] > 0.6 else ("🟡 MED" if info["suspicion"] > 0.3 else "🟢 LOW")

        line = (
            f"  [{step+1:>3d}/{env.match_duration}] {phase:<12} | "
            f"Price: ₹{info['price']:>6,.0f} ({price_act:>8}) | "
            f"Limit: {info['limit']}  ({limit_act:>8}) | "
            f"Suspicion: {info['suspicion']:.2f} {susp_level:<10} | "
            f"Inv: {info['inventory']:>5} | "
            f"R: {reward:+.3f}"
        )

        # Add alerts
        if info["suspicion"] > 0.65:
            line += "  ⚠️  BOT ATTACK DETECTED"
        if info["price"] > env.base_price * 1.8:
            line += "  💰 SURGE PRICING ACTIVE"
        if info["inventory"] < env.total_inventory * 0.10:
            line += "  🔥 NEARLY SOLD OUT"

        print(line)
        log_lines.append(line)
        state = next_state
        step += 1

    # Summary
    total_sold = total_fair + total_scalpers
    fairness   = total_fair / max(total_sold, 1)
    scalp_rate = total_scalpers / max(total_sold, 1)

    summary = (
        f"\n{'='*65}\n"
        f"  FINAL MATCH DAY REPORT\n"
        f"{'='*65}\n"
        f"  Total Revenue       : ₹{total_revenue:>12,.2f}\n"
        f"  Tickets to Genuine  : {int(total_fair):>6}\n"
        f"  Tickets to Scalpers : {int(total_scalpers):>6}\n"
        f"  Fairness Index      : {fairness:.3f}  (higher = better)\n"
        f"  Scalper Rate        : {scalp_rate:.3f}  (lower = better)\n"
        f"  Remaining Inventory : {env.inventory}\n"
        f"{'='*65}"
    )
    print(summary)
    log_lines.append(summary)

    # Save log
    log_path = os.path.join(RESULTS_DIR, f"matchday_log_{agent_type}.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    print(f"\n  Full log saved → {log_path}")

    # ── Plots ────────────────────────────────────────────────────────────────
    steps = list(range(len(price_hist)))
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f"IPL Match Day Analytics — {agent_label.upper()}",
        fontsize=16, fontweight="bold"
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.35)

    # Price & Suspicion
    ax1 = fig.add_subplot(gs[0, :])
    c1 = "#E74C3C"
    c2 = "#3498DB"
    ax1.plot(steps, price_hist, color=c1, linewidth=2, label="Ticket Price (₹)")
    ax1.set_ylabel("Ticket Price (₹)", color=c1, fontsize=11)
    ax1.tick_params(axis="y", labelcolor=c1)
    ax1b = ax1.twinx()
    ax1b.plot(steps, susp_hist, color=c2, linewidth=2, linestyle="--", label="Suspicion Score S'")
    ax1b.set_ylabel("Suspicion Score S'", color=c2, fontsize=11)
    ax1b.tick_params(axis="y", labelcolor=c2)
    ax1.axvline(x=env.match_duration * 0.15, color="orange", linestyle=":", linewidth=1.5, label="Launch Window End")
    ax1.set_xlabel("Time Step")
    ax1.set_title("Dynamic Pricing vs Suspicion Score Over Time")
    ax1.grid(alpha=0.3)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    # Inventory depletion
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.fill_between(steps, inv_hist, alpha=0.3, color="#2ECC71")
    ax2.plot(steps, inv_hist, color="#27AE60", linewidth=2)
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Remaining Inventory")
    ax2.set_title("Inventory Depletion Curve")
    ax2.grid(alpha=0.3)

    # Step reward
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.bar(steps, rew_hist,
            color=["#E74C3C" if r < 0 else "#2ECC71" for r in rew_hist],
            alpha=0.7)
    ax3.axhline(y=0, color="black", linewidth=0.8)
    ax3.set_xlabel("Time Step")
    ax3.set_ylabel("Step Reward")
    ax3.set_title("Step-wise Reward (Green=Positive, Red=Penalty)")
    ax3.grid(axis="y", alpha=0.3)

    out = os.path.join(RESULTS_DIR, f"matchday_analytics_{agent_type}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Analytics plot  → {out}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IPL Match Day Simulator")
    parser.add_argument("--agent", choices=["dqn", "ppo", "static", "rule_based", "all"],
                        default="all")
    parser.add_argument("--seed", type=int, default=99)
    args = parser.parse_args()

    agents_to_run = (
        ["dqn", "ppo", "static", "rule_based"] if args.agent == "all" else [args.agent]
    )

    for agent_type in agents_to_run:
        if agent_type == "dqn":
            from agent_dqn import DQNAgent
            ckpt = os.path.join(RESULTS_DIR, "dqn_checkpoint.pth")
            if not os.path.exists(ckpt):
                print(f"[SKIP] DQN checkpoint not found. Run train.py first.")
                continue
            policy = DQNAgent(); policy.load(ckpt)
            simulate("DQN Agent", policy, "dqn", seed=args.seed)

        elif agent_type == "ppo":
            from agent_ppo import PPOAgent
            ckpt = os.path.join(RESULTS_DIR, "ppo_checkpoint.pth")
            if not os.path.exists(ckpt):
                print(f"[SKIP] PPO checkpoint not found. Run train.py first.")
                continue
            policy = PPOAgent(); policy.load(ckpt)
            simulate("PPO Agent", policy, "ppo", seed=args.seed)

        elif agent_type == "static":
            simulate("Static Baseline", StaticAgent(), "static", seed=args.seed)

        elif agent_type == "rule_based":
            simulate("Rule-Based Agent", RuleBasedAgent(), "rule_based", seed=args.seed)
