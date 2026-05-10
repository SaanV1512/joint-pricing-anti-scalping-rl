from __future__ import annotations

import argparse
import os
import time
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from environment import IPLTicketingEnv
from agent_dqn   import DQNAgent
from agent_ppo   import PPOAgent

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def rolling(x: np.ndarray, w: int = 30) -> np.ndarray:
    out = np.zeros(len(x), dtype=float)
    for i in range(len(x)):
        lo = max(0, i - w + 1)
        out[i] = x[lo:i+1].mean()
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Rainbow DQN Training
# ─────────────────────────────────────────────────────────────────────────────

def _train_rainbow_once(n_episodes: int, seed: int, verbose_every: int = 50) -> Dict:
    env   = IPLTicketingEnv(seed=seed, adaptive_bots=True)
    agent = DQNAgent(
        state_dim          = 6,
        n_actions          = 9,
        gamma              = 0.99,
        n_step             = 3,
        lr                 = 3e-4,
        batch_size         = 256,
        buffer_capacity    = 200_000,
        target_update_freq = 150,
        per_alpha          = 0.6,
        per_beta_start     = 0.4,
        per_beta_steps     = 80_000,
        hidden             = 256,
        sigma_init         = 0.5,
    )

    rewards:  List[float] = []
    revenues: List[float] = []
    scalpers: List[float] = []

    for ep in range(1, n_episodes + 1):
        state, _ = env.reset()
        ep_r = ep_rev = ep_sc = ep_tot = 0.0
        done = False
        while not done:
            a = agent.select_action(state, training=True)
            ns, r, term, trunc, info = env.step(a)
            done = term or trunc
            agent.store(state, a, r, ns, done)
            agent.update()
            state = ns
            ep_r   += r
            ep_rev += info["revenue"]
            ep_sc  += info["scalper_tickets"]
            ep_tot += info["fair_tickets"] + info["scalper_tickets"]

        rewards.append(ep_r)
        revenues.append(ep_rev)
        scalpers.append(ep_sc / max(ep_tot, 1))

        if ep % verbose_every == 0:
            avg_loss = np.mean(agent.loss_history[-500:]) if agent.loss_history else 0
            avg_q    = np.mean(agent.q_value_history[-500:]) if agent.q_value_history else 0
            print(
                f"  [Rainbow|s={seed}] Ep {ep:>4}/{n_episodes} | "
                f"Reward: {np.mean(rewards[-50:]):+.3f} | "
                f"Revenue: ₹{np.mean(revenues[-50:]):>12,.0f} | "
                f"ScalperRate: {np.mean(scalpers[-50:]):.3f} | "
                f"Loss: {avg_loss:.5f} | Q̄: {avg_q:.3f} | "
                f"Buffer: {len(agent.memory):>7}"
            )

    return dict(agent=agent,
                rewards=np.array(rewards),
                revenues=np.array(revenues),
                scalpers=np.array(scalpers),
                losses=np.array(agent.loss_history),
                q_values=np.array(agent.q_value_history))


def train_dqn(n_episodes: int = 500, n_seeds: int = 1) -> DQNAgent:
    print(f"\n{'='*70}")
    print(f"  Training Rainbow DQN — {n_episodes} episodes × {n_seeds} seeds")
    print(f"{'='*70}")
    t0          = time.time()
    all_rewards = []
    best_agent  = None
    best_score  = -np.inf

    for s in range(n_seeds):
        res = _train_rainbow_once(n_episodes, seed=s)
        all_rewards.append(res["rewards"])
        score = res["rewards"][-50:].mean()
        if score > best_score:
            best_score = score
            best_agent = res["agent"]
        if s == 0:
            np.save(f"{RESULTS_DIR}/dqn_losses.npy",   res["losses"])
            np.save(f"{RESULTS_DIR}/dqn_q_values.npy", res["q_values"])

    rewards_arr = np.array(all_rewards)
    np.save(f"{RESULTS_DIR}/dqn_rewards.npy", rewards_arr)
    best_agent.save(f"{RESULTS_DIR}/dqn_checkpoint.pth")
    _plot_dashboard("Rainbow DQN", rewards_arr,
                    np.load(f"{RESULTS_DIR}/dqn_losses.npy"),
                    np.load(f"{RESULTS_DIR}/dqn_q_values.npy"))
    print(f"\n  Done in {time.time()-t0:.1f}s  |  Best avg: {best_score:+.3f}\n")
    return best_agent


# ─────────────────────────────────────────────────────────────────────────────
# Recurrent PPO Training
# ─────────────────────────────────────────────────────────────────────────────

def _train_rppo_once(n_episodes: int, seed: int, verbose_every: int = 20) -> Dict:
    import torch
    env   = IPLTicketingEnv(seed=seed, adaptive_bots=True)
    agent = PPOAgent(
        state_dim     = 6,
        n_actions     = 9,
        gamma         = 0.99,
        gae_lambda    = 0.95,
        clip_epsilon  = 0.2,
        vf_clip       = 0.2,
        c1            = 0.5,
        c2            = 0.01,
        lr_actor      = 3e-4,
        lr_critic     = 1e-3,
        n_epochs      = 10,
        rollout_steps = 512,
        mini_batch    = 64,
        max_kl        = 0.015,
        lstm_hidden   = 256,
    )

    rewards:  List[float] = []
    revenues: List[float] = []

    state, _ = env.reset()
    agent.reset_hidden()
    ep_r = ep_rev = 0.0
    ep = global_step = 0

    while ep < n_episodes:
        for _ in range(agent.rollout_steps):
            action, log_prob, value = agent.collect_step(state)
            ns, r, term, trunc, info = env.step(action)
            done = term or trunc
            agent.store(state, action, log_prob, r, value, done)
            state  = ns
            ep_r   += r
            ep_rev += info["revenue"]
            global_step += 1
            if done:
                ep += 1
                rewards.append(ep_r)
                revenues.append(ep_rev)
                if ep % verbose_every == 0:
                    avg_loss = np.mean(agent.loss_history[-10:]) if agent.loss_history else 0
                    avg_kl   = np.mean(agent.kl_history[-10:])   if agent.kl_history   else 0
                    print(
                        f"  [R-PPO|s={seed}] Ep {ep:>4}/{n_episodes} | "
                        f"Reward: {np.mean(rewards[-20:]):+.3f} | "
                        f"Revenue: ₹{np.mean(revenues[-20:]):>12,.0f} | "
                        f"Steps: {global_step:>7} | Loss: {avg_loss:.4f} | KL: {avg_kl:.5f}"
                    )
                ep_r = ep_rev = 0.0
                state, _ = env.reset()
                if ep >= n_episodes:
                    break

        with torch.no_grad():
            import torch as _t
            s_t  = _t.tensor(agent.obs_norm.normalize(state),
                             dtype=_t.float32).unsqueeze(0).unsqueeze(0)
            _, lv, _ = agent.net(s_t, agent._hx)
            last_val = float(lv.squeeze().item()) if not done else 0.0
        agent.update(last_value=last_val)

    return dict(agent=agent,
                rewards=np.array(rewards),
                revenues=np.array(revenues),
                losses=np.array(agent.loss_history))


def train_ppo(n_episodes: int = 300, n_seeds: int = 1) -> PPOAgent:
    print(f"\n{'='*70}")
    print(f"  Training Recurrent PPO — {n_episodes} episodes × {n_seeds} seeds")
    print(f"{'='*70}")
    t0          = time.time()
    all_rewards = []
    best_agent  = None
    best_score  = -np.inf

    for s in range(n_seeds):
        res = _train_rppo_once(n_episodes, seed=s)
        all_rewards.append(res["rewards"])
        score = res["rewards"][-20:].mean()
        if score > best_score:
            best_score = score
            best_agent = res["agent"]
        if s == 0:
            np.save(f"{RESULTS_DIR}/ppo_losses.npy", res["losses"])

    rewards_arr = np.array(all_rewards)
    np.save(f"{RESULTS_DIR}/ppo_rewards.npy", rewards_arr)
    best_agent.save(f"{RESULTS_DIR}/ppo_checkpoint.pth")
    _plot_dashboard("R-PPO", rewards_arr,
                    np.load(f"{RESULTS_DIR}/ppo_losses.npy"))
    print(f"\n  Done in {time.time()-t0:.1f}s  |  Best avg: {best_score:+.3f}\n")
    return best_agent


# ─────────────────────────────────────────────────────────────────────────────
# Publication-quality 4-panel training dashboard
# ─────────────────────────────────────────────────────────────────────────────

def _plot_dashboard(label: str, rewards_arr: np.ndarray,
                    losses: np.ndarray,
                    q_values: Optional[np.ndarray] = None):
    n_seeds, n_eps = rewards_arr.shape
    eps = np.arange(1, n_eps + 1)
    mean_r = rewards_arr.mean(0)
    std_r  = rewards_arr.std(0)

    BLUE  = "#1565C0"; LBLUE = "#90CAF9"
    RED   = "#C62828"; LRED  = "#EF9A9A"
    GREEN = "#2E7D32"
    AMBER = "#F57F17"

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"{label} — Training Dashboard (IPL Ticketing RL)",
                 fontsize=15, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)

    # Panel 1: Reward convergence + 95% CI
    ax1 = fig.add_subplot(gs[0, 0])
    rm  = rolling(mean_r, 30)
    rs  = rolling(std_r,  30)
    ax1.fill_between(eps, rm - 1.96*std_r, rm + 1.96*std_r,
                     alpha=0.10, color=LBLUE, label="95% CI across seeds")
    ax1.fill_between(eps, rm - rs, rm + rs,
                     alpha=0.25, color=LBLUE, label="Rolling ±1σ")
    ax1.plot(eps, mean_r, alpha=0.20, color=BLUE, linewidth=0.6)
    ax1.plot(eps, rm,     color=BLUE, linewidth=2.2, label="Rolling mean (w=30)")
    final_avg = rm[-30:].mean()
    ax1.axhline(final_avg, color=GREEN, linestyle="--", linewidth=1.5,
                label=f"Final avg: {final_avg:+.2f}")
    ax1.set_xlabel("Episode", fontsize=11)
    ax1.set_ylabel("Total Reward", fontsize=11)
    ax1.set_title("Reward Convergence" + (f" ({n_seeds} seeds)" if n_seeds > 1 else ""),
                  fontweight="bold")
    ax1.legend(fontsize=8, loc="lower right")
    ax1.grid(alpha=0.3)
    ax1.spines[["top", "right"]].set_visible(False)

    # Panel 2: Loss (log scale)
    ax2 = fig.add_subplot(gs[0, 1])
    if losses.size > 0:
        u = np.arange(len(losses))
        ax2.plot(u, losses, alpha=0.15, color=RED, linewidth=0.5)
        ax2.plot(u, rolling(losses, 200), color=RED, linewidth=2.0,
                 label="Rolling mean (w=200)")
        ax2.set_yscale("log")
        ax2.set_xlabel("Update Step", fontsize=11)
        ax2.set_ylabel("Loss (log scale)", fontsize=11)
        ax2.set_title("Training Loss", fontweight="bold")
        ax2.legend(fontsize=9)
        ax2.grid(alpha=0.3, which="both")
        ax2.spines[["top", "right"]].set_visible(False)

    # Panel 3: Q-value / Per-seed curves
    ax3 = fig.add_subplot(gs[1, 0])
    if q_values is not None and q_values.size > 0:
        qs = np.arange(len(q_values))
        ax3.plot(qs, q_values, alpha=0.15, color=AMBER, linewidth=0.5)
        ax3.plot(qs, rolling(q_values, 500), color=AMBER, linewidth=2.0,
                 label="Q̄ rolling mean (w=500)")
        ax3.set_xlabel("Environment Step", fontsize=11)
        ax3.set_ylabel("Mean Max Q-value", fontsize=11)
        ax3.set_title("Q-Value Estimation (NoisyNet-driven)", fontweight="bold")
        ax3.legend(fontsize=9)
        ax3.grid(alpha=0.3)
        ax3.spines[["top", "right"]].set_visible(False)
    else:
        palette = plt.cm.viridis(np.linspace(0.2, 0.9, n_seeds))
        for i, (row, c) in enumerate(zip(rewards_arr, palette)):
            ax3.plot(eps, rolling(row, 25), color=c, linewidth=1.4,
                     alpha=0.85, label=f"Seed {i}")
        ax3.set_xlabel("Episode", fontsize=11)
        ax3.set_ylabel("Reward (rolling w=25)", fontsize=11)
        ax3.set_title("Per-Seed Learning Curves", fontweight="bold")
        if n_seeds <= 5:
            ax3.legend(fontsize=9)
        ax3.grid(alpha=0.3)
        ax3.spines[["top", "right"]].set_visible(False)

    # Panel 4: Final reward histogram
    ax4 = fig.add_subplot(gs[1, 1])
    final_rews = rewards_arr[:, int(n_eps * 0.75):].flatten()
    n, bins, patches = ax4.hist(final_rews, bins=35, color=GREEN,
                                 edgecolor="white", alpha=0.85, density=True)
    mu, sigma = final_rews.mean(), final_rews.std()
    # Overlay Gaussian KDE
    from scipy.stats import norm as _norm
    try:
        x_kde = np.linspace(final_rews.min(), final_rews.max(), 200)
        ax4.plot(x_kde, _norm.pdf(x_kde, mu, sigma), "k--", linewidth=1.5,
                 label=f"N(μ={mu:.2f}, σ={sigma:.2f})")
    except Exception:
        pass
    ax4.axvline(mu, color="black", linewidth=2, label=f"μ = {mu:.2f}")
    ax4.axvline(mu - sigma, color="gray", linewidth=1, linestyle=":")
    ax4.axvline(mu + sigma, color="gray", linewidth=1, linestyle=":")
    ax4.set_xlabel("Episode Reward", fontsize=11)
    ax4.set_ylabel("Density", fontsize=11)
    ax4.set_title("Final Reward Distribution (last 25% training)", fontweight="bold")
    ax4.legend(fontsize=9)
    ax4.grid(alpha=0.3)
    ax4.spines[["top", "right"]].set_visible(False)

    name = label.lower().replace(" ", "_").replace("-", "_")
    out  = os.path.join(RESULTS_DIR, f"{name}_training_dashboard.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Dashboard → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent",    choices=["dqn", "ppo", "both"], default="both")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seeds",    type=int, default=3,
                        help="Number of random seeds (>1 gives confidence bands)")
    args = parser.parse_args()

    if args.agent in ("dqn", "both"):
        train_dqn(n_episodes=args.episodes or 800, n_seeds=args.seeds)
    if args.agent in ("ppo", "both"):
        train_ppo(n_episodes=args.episodes or 500, n_seeds=args.seeds)

    print("\nAll done. Results in ./results/")
