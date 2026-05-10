from __future__ import annotations

import argparse
import os
import time
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from environment import IPLTicketingEnv, N_ACTIONS

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Vanilla DQN  (for ablation only — minimal implementation)
# ─────────────────────────────────────────────────────────────────────────────

import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim


class _VanillaQNet(nn.Module):
    def __init__(self, state_dim=6, n_actions=N_ACTIONS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128),       nn.ReLU(),
            nn.Linear(128, n_actions),
        )
    def forward(self, x): return self.net(x)


class _VanillaDQN:
    """Vanilla DQN: uniform replay, ε-greedy, standard FC, vanilla target."""
    name = "Vanilla DQN"

    def __init__(self, double=False, dueling=False):
        self.double  = double
        self.dueling = dueling
        if dueling:
            from agent_dqn import DuelingQNetwork
            self.policy = DuelingQNetwork()
            self.target = DuelingQNetwork()
        else:
            self.policy = _VanillaQNet()
            self.target = _VanillaQNet()
        self.target.load_state_dict(self.policy.state_dict())
        self.opt    = optim.Adam(self.policy.parameters(), lr=1e-3)
        self.buf    = deque(maxlen=50_000)
        self.steps  = 0
        self.eps    = 1.0
        self.losses = []

    def select_action(self, s, training=True):
        if training:
            self.eps = max(0.05, 1.0 - self.steps / 20_000)
            self.steps += 1
        if training and random.random() < self.eps:
            return random.randrange(N_ACTIONS)
        with torch.no_grad():
            q = self.policy(torch.tensor(s, dtype=torch.float32).unsqueeze(0))
            return int(q.argmax().item())

    def store(self, s, a, r, ns, d):
        self.buf.append((s, a, r, ns, d))

    def update(self):
        if len(self.buf) < 64: return None
        batch = random.sample(self.buf, 64)
        s  = torch.tensor(np.array([t[0] for t in batch]), dtype=torch.float32)
        a  = torch.tensor([t[1] for t in batch], dtype=torch.long)
        r  = torch.tensor([t[2] for t in batch], dtype=torch.float32)
        ns = torch.tensor(np.array([t[3] for t in batch]), dtype=torch.float32)
        d  = torch.tensor([t[4] for t in batch], dtype=torch.float32)

        q_cur = self.policy(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            if self.double:
                best_a  = self.policy(ns).argmax(1, keepdim=True)
                q_next  = self.target(ns).gather(1, best_a).squeeze(1)
            else:
                q_next = self.target(ns).max(1).values
            tgt = r + 0.99 * q_next * (1 - d)

        loss = nn.functional.mse_loss(q_cur, tgt)
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        if self.steps % 500 == 0:
            self.target.load_state_dict(self.policy.state_dict())
        self.losses.append(loss.item())
        return loss.item()


# ─────────────────────────────────────────────────────────────────────────────
# Ablation runner
# ─────────────────────────────────────────────────────────────────────────────

def _run_agent(agent, n_episodes: int, seed: int = 7) -> np.ndarray:
    env = IPLTicketingEnv(seed=seed, adaptive_bots=True)
    rewards = []
    state, _ = env.reset()
    ep_r = 0.0
    ep   = 0
    done = False

    while ep < n_episodes:
        a = agent.select_action(state, training=True)
        ns, r, term, trunc, _ = env.step(a)
        done = term or trunc
        agent.store(state, a, r, ns, done)
        agent.update()
        ep_r  += r
        state  = ns
        if done:
            rewards.append(ep_r)
            ep_r   = 0.0
            ep    += 1
            state, _ = env.reset()

    return np.array(rewards)


def run_ablation(n_episodes: int = 300):
    print(f"\n{'='*60}")
    print(f"  ABLATION STUDY — {n_episodes} episodes per variant")
    print(f"{'='*60}\n")

    # Import Rainbow agent
    from agent_dqn import DQNAgent as RainbowAgent

    agents = {
        "Vanilla DQN":         lambda: _VanillaDQN(double=False, dueling=False),
        "Double DQN":          lambda: _VanillaDQN(double=True,  dueling=False),
        "Dueling + Double + PER": lambda: _VanillaDQN(double=True, dueling=True),
        "Rainbow DQN (Ours)":  lambda: RainbowAgent(),
    }

    colours = ["#78909C", "#EF9A9A", "#64B5F6", "#2ECC71"]
    results: Dict[str, np.ndarray] = {}

    for (name, factory), colour in zip(agents.items(), colours):
        print(f"  Training: {name}")
        t0     = time.time()
        agent  = factory()
        rews   = _run_agent(agent, n_episodes, seed=42)
        results[name] = rews
        print(f"    → Final 50-ep avg: {rews[-50:].mean():+.3f}  ({time.time()-t0:.1f}s)")

    np.save(os.path.join(RESULTS_DIR, "ablation_results.npy"),
            {k: v for k, v in results.items()})

    # ── Publication-quality ablation plot ─────────────────────────────────────
    def rm(x, w=20):
        out = np.zeros(len(x))
        for i in range(len(x)):
            lo = max(0, i - w + 1)
            out[i] = x[lo:i+1].mean()
        return out

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Ablation Study: DQN Variants (IPL Anti-Scalping)",
                 fontsize=14, fontweight="bold")

    eps = np.arange(1, n_episodes + 1)
    for (name, rews), colour in zip(results.items(), colours):
        lw   = 2.5 if "Rainbow" in name else 1.5
        ls   = "-"  if "Rainbow" in name else "--"
        axes[0].plot(eps, rm(rews, 30), label=name, color=colour,
                     linewidth=lw, linestyle=ls)

    axes[0].set_xlabel("Episode", fontsize=11)
    axes[0].set_ylabel("Rolling Reward (w=30)", fontsize=11)
    axes[0].set_title("Learning Curves", fontweight="bold")
    axes[0].legend(fontsize=9, loc="lower right")
    axes[0].grid(alpha=0.3)

    # Bar chart: final performance
    names   = list(results.keys())
    finals  = [results[n][-50:].mean() for n in names]
    stds    = [results[n][-50:].std()  for n in names]
    bars    = axes[1].bar(range(len(names)), finals, color=colours,
                          edgecolor="white", linewidth=1.5, yerr=stds,
                          capsize=5, error_kw={"linewidth": 1.5})

    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(
        [n.replace(" (Ours)", "\n(Ours)") for n in names],
        fontsize=9, ha="center"
    )
    axes[1].set_ylabel("Mean Reward (last 50 eps)", fontsize=11)
    axes[1].set_title("Final Performance Comparison", fontweight="bold")
    axes[1].grid(axis="y", alpha=0.3)

    for bar, v, s in zip(bars, finals, stds):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + s + 0.1,
                     f"{v:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "ablation_study.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Ablation plot saved → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=400)
    args = parser.parse_args()

    # Ensure DuelingQNetwork is importable for the ablation intermediate variant
    try:
        from agent_dqn import RainbowNetwork as DuelingQNetwork
        # Monkey-patch the local _VanillaDQN to use the real Dueling net
        import agent_dqn
        _VanillaDQN_orig_init = _VanillaDQN.__init__
        def _patched_init(self, double=False, dueling=False):
            _VanillaDQN_orig_init(self, double, dueling)
        _VanillaDQN.__init__ = _patched_init
    except Exception:
        pass

    run_ablation(n_episodes=args.episodes)
