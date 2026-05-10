from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from environment import N_ACTIONS


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Running Observation Normaliser (Welford's online algorithm)
# ══════════════════════════════════════════════════════════════════════════════

class RunningNormalizer:
    """
    Online mean and variance estimation using Welford's algorithm.
    Normalizes incoming observations to zero-mean, unit-variance.
    This is critical when features have different scales (e.g. inventory
    vs suspicion score vs price).
    """

    def __init__(self, shape: Tuple[int, ...], clip: float = 5.0):
        self.mean  = np.zeros(shape, dtype=np.float64)
        self.var   = np.ones(shape,  dtype=np.float64)
        self.count = 0
        self.clip  = clip

    def update(self, x: np.ndarray):
        """Update running stats with a new observation."""
        self.count += 1
        delta  = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.var += delta * delta2      # M2 accumulation

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Return normalized x using current running statistics."""
        if self.count < 2:
            return x.astype(np.float32)
        std  = np.sqrt(self.var / max(self.count - 1, 1) + 1e-8)
        norm = (x - self.mean) / std
        return np.clip(norm, -self.clip, self.clip).astype(np.float32)

    def state_dict(self):
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, d):
        self.mean, self.var, self.count = d["mean"], d["var"], d["count"]


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Recurrent Actor-Critic (LSTM backbone)
# ══════════════════════════════════════════════════════════════════════════════

class RecurrentActorCritic(nn.Module):
    """
    Actor-Critic with an LSTM backbone.

    The LSTM hidden state h_t is carried across steps within an episode,
    allowing the agent to learn temporal patterns — e.g. "prices have been
    rising for 3 steps, bots are probably about to give up."

    Architecture
    ------------
    Input embedding : FC(state_dim → 128) → LayerNorm → ReLU
    LSTM            : 1 layer, hidden_size = 256
    Actor head      : FC(256 → 128) → ReLU → FC(128 → n_actions)
    Critic head     : FC(256 → 128) → ReLU → FC(128 → 1)
    """

    def __init__(self, state_dim: int = 6, n_actions: int = N_ACTIONS,
                 lstm_hidden: int = 256):
        super().__init__()
        self.lstm_hidden = lstm_hidden

        # Input embedding
        self.embed = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )

        # LSTM recurrent core
        self.lstm = nn.LSTM(128, lstm_hidden, batch_first=True)

        # Actor head
        self.actor = nn.Sequential(
            nn.Linear(lstm_hidden, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(lstm_hidden, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param, gain=1.0)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)
        for m in list(self.actor.modules()) + list(self.critic.modules()):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def get_init_hidden(self, batch: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return zeroed LSTM hidden state."""
        return (
            torch.zeros(1, batch, self.lstm_hidden),
            torch.zeros(1, batch, self.lstm_hidden),
        )

    def forward(
        self,
        x:  torch.Tensor,                              # (B, T, state_dim)
        hx: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple]:
        """
        Returns: logits (B, T, A), values (B, T), new_hidden
        """
        emb        = self.embed(x)                     # (B, T, 128)
        out, new_h = self.lstm(emb, hx)                # (B, T, hidden)
        logits     = self.actor(out)                   # (B, T, A)
        values     = self.critic(out).squeeze(-1)      # (B, T)
        return logits, values, new_h

    def act(
        self,
        state: np.ndarray,
        hx:    Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[int, float, float, Tuple]:
        """Single-step inference. Returns (action, log_prob, value, new_hx)."""
        s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,D)
        with torch.no_grad():
            logits, values, new_hx = self(s, hx)
        dist   = Categorical(logits=logits[0, 0])
        action = dist.sample()
        return (
            int(action.item()),
            float(dist.log_prob(action).item()),
            float(values[0, 0].item()),
            new_hx,
        )


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Rollout Buffer (stores per-episode hidden states)
# ══════════════════════════════════════════════════════════════════════════════

class RecurrentRolloutBuffer:
    """Stores experience + LSTM hidden state snapshots for BPTT."""

    def __init__(self):
        self.clear()

    def clear(self):
        self.states:     List[np.ndarray]  = []
        self.actions:    List[int]         = []
        self.log_probs:  List[float]       = []
        self.rewards:    List[float]       = []
        self.values:     List[float]       = []
        self.dones:      List[bool]        = []
        # Store hidden state at episode start (for truncated BPTT)
        self.episode_starts: List[int]     = [0]

    def add(self, state, action, log_prob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        if done:
            self.episode_starts.append(len(self.states))

    def __len__(self): return len(self.rewards)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Recurrent PPO Agent
# ══════════════════════════════════════════════════════════════════════════════

class PPOAgent:
    """
    Recurrent PPO (R-PPO) with LSTM backbone and running observation normalization.

    Hyperparameters
    ---------------
    gamma         : 0.99     (discount factor)
    gae_lambda    : 0.95     (GAE smoothing)
    clip_epsilon  : 0.2      (PPO clip ratio)
    vf_clip       : 0.2      (value function clip — OpenAI trick)
    c1            : 0.5      (value loss weight)
    c2            : 0.01     (entropy bonus weight)
    lr_actor      : 3e-4     (actor Adam lr)
    lr_critic     : 1e-3     (critic Adam lr — typically higher)
    n_epochs      : 10       (PPO update epochs per rollout)
    rollout_steps : 512      (steps per update)
    mini_batch    : 64       (mini-batch size for SGD)
    max_kl        : 0.015    (KL early stopping threshold)
    lstm_hidden   : 256      (LSTM hidden units)
    """

    def __init__(
        self,
        state_dim:     int   = 6,
        n_actions:     int   = N_ACTIONS,
        gamma:         float = 0.99,
        gae_lambda:    float = 0.95,
        clip_epsilon:  float = 0.2,
        vf_clip:       float = 0.2,
        c1:            float = 0.5,
        c2:            float = 0.01,
        lr:            float = 3e-4,      # kept for compat
        lr_actor:      float = 3e-4,
        lr_critic:     float = 1e-3,
        n_epochs:      int   = 10,
        rollout_steps: int   = 512,
        mini_batch:    int   = 64,
        max_kl:        float = 0.015,
        lstm_hidden:   int   = 256,
        device:        str   = "cpu",
    ):
        self.gamma         = gamma
        self.gae_lambda    = gae_lambda
        self.clip_epsilon  = clip_epsilon
        self.vf_clip       = vf_clip
        self.c1            = c1
        self.c2            = c2
        self.n_epochs      = n_epochs
        self.rollout_steps = rollout_steps
        self.mini_batch    = mini_batch
        self.max_kl        = max_kl
        self.device        = torch.device(device)

        self.net = RecurrentActorCritic(state_dim, n_actions, lstm_hidden).to(self.device)

        # Separate actor / critic optimizers
        self.actor_optimizer  = optim.Adam(
            list(self.net.embed.parameters()) +
            list(self.net.lstm.parameters()) +
            list(self.net.actor.parameters()),
            lr=lr_actor, eps=1e-5,
        )
        self.critic_optimizer = optim.Adam(
            self.net.critic.parameters(), lr=lr_critic, eps=1e-5
        )

        # LR schedulers
        self.actor_sched  = optim.lr_scheduler.CosineAnnealingLR(
            self.actor_optimizer, T_max=2000, eta_min=lr_actor * 0.1
        )
        self.critic_sched = optim.lr_scheduler.CosineAnnealingLR(
            self.critic_optimizer, T_max=2000, eta_min=lr_critic * 0.1
        )

        # Running observation normalizer
        self.obs_norm = RunningNormalizer(shape=(state_dim,))

        # Buffer
        self.buffer = RecurrentRolloutBuffer()

        # Current LSTM hidden state (reset at episode start)
        self._hx = self.net.get_init_hidden(batch=1)

        self.loss_history: List[float] = []
        self.kl_history:   List[float] = []
        self.update_count: int         = 0

    def reset_hidden(self):
        """Call at the start of each new episode."""
        self._hx = self.net.get_init_hidden(batch=1)

    def normalize_obs(self, state: np.ndarray) -> np.ndarray:
        self.obs_norm.update(state)
        return self.obs_norm.normalize(state)

    # ── Single step: act ─────────────────────────────────────────────────────
    def collect_step(self, state: np.ndarray) -> Tuple[int, float, float]:
        norm_state = self.normalize_obs(state)
        action, log_prob, value, new_hx = self.net.act(norm_state, self._hx)
        self._hx = new_hx
        return action, log_prob, value

    def store(self, state, action, log_prob, reward, value, done):
        norm_state = self.obs_norm.normalize(state)
        self.buffer.add(norm_state, action, log_prob, reward, value, done)
        if done:
            self.reset_hidden()

    # ── GAE ──────────────────────────────────────────────────────────────────
    def _compute_gae(self, last_value: float) -> Tuple[np.ndarray, np.ndarray]:
        rewards = self.buffer.rewards
        values  = self.buffer.values
        dones   = self.buffer.dones
        T       = len(rewards)

        advantages = np.zeros(T, dtype=np.float32)
        gae  = 0.0
        next_v = last_value
        for t in reversed(range(T)):
            mask  = 1.0 - float(dones[t])
            delta = rewards[t] + self.gamma * next_v * mask - values[t]
            gae   = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[t] = gae
            next_v = values[t]

        returns = advantages + np.array(values, dtype=np.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    # ── PPO update with value clipping + KL early stopping ───────────────────
    def update(self, last_value: float = 0.0) -> float:
        advantages, returns = self._compute_gae(last_value)

        states   = torch.tensor(np.array(self.buffer.states),    dtype=torch.float32, device=self.device)
        actions  = torch.tensor(self.buffer.actions,             dtype=torch.long,    device=self.device)
        old_lp   = torch.tensor(self.buffer.log_probs,           dtype=torch.float32, device=self.device)
        old_vals = torch.tensor(self.buffer.values,              dtype=torch.float32, device=self.device)
        advs     = torch.tensor(advantages,                      dtype=torch.float32, device=self.device)
        rets     = torch.tensor(returns,                         dtype=torch.float32, device=self.device)

        T         = len(self.buffer)
        hx_zero   = self.net.get_init_hidden(batch=1)
        total_loss = 0.0
        n_updates  = 0

        for epoch in range(self.n_epochs):
            # Truncated BPTT: process mini-batches as sequences
            indices = np.random.permutation(T)
            for start in range(0, T, self.mini_batch):
                idx = indices[start: start + self.mini_batch]
                s_b = states[idx].unsqueeze(0)    # (1, B, D)
                a_b = actions[idx]
                olp_b = old_lp[idx]
                old_v_b = old_vals[idx]
                adv_b = advs[idx]
                ret_b = rets[idx]

                logits_b, values_b, _ = self.net(s_b, hx_zero)
                logits_b = logits_b.squeeze(0)
                values_b = values_b.squeeze(0)

                dist    = Categorical(logits=logits_b)
                new_lp  = dist.log_prob(a_b)
                entropy = dist.entropy()

                ratio  = torch.exp(new_lp - olp_b)
                surr1  = ratio * adv_b
                surr2  = torch.clamp(ratio, 1 - self.clip_epsilon,
                                     1 + self.clip_epsilon) * adv_b
                L_clip = -torch.min(surr1, surr2).mean()

                # Value function with clipping (PPO2 trick)
                v_clipped = old_v_b + torch.clamp(values_b - old_v_b,
                                                   -self.vf_clip, self.vf_clip)
                L_val = self.c1 * torch.max(
                    F.mse_loss(values_b, ret_b),
                    F.mse_loss(v_clipped, ret_b),
                ).mean()

                L_ent = -self.c2 * entropy.mean()
                loss  = L_clip + L_val + L_ent

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=0.5)
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                total_loss += loss.item()
                n_updates  += 1

            # KL early stopping
            with torch.no_grad():
                logits_all, _, _ = self.net(states.unsqueeze(0), hx_zero)
                new_lp_all = Categorical(logits=logits_all.squeeze(0)).log_prob(actions)
                approx_kl  = float(((old_lp - new_lp_all) ** 2).mean().item())
                self.kl_history.append(approx_kl)
                if approx_kl > self.max_kl:
                    break   # stop early if policy diverging

        self.actor_sched.step()
        self.critic_sched.step()
        self.buffer.clear()
        self.reset_hidden()
        self.update_count += 1

        mean_loss = total_loss / max(n_updates, 1)
        self.loss_history.append(mean_loss)
        return mean_loss

    # ── Save / load ───────────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        torch.save({
            "net":            self.net.state_dict(),
            "actor_opt":      self.actor_optimizer.state_dict(),
            "critic_opt":     self.critic_optimizer.state_dict(),
            "updates":        self.update_count,
            "obs_norm":       self.obs_norm.state_dict(),
        }, path)
        print(f"[R-PPO] Saved → {path}")

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.net.load_state_dict(ckpt["net"])
        self.actor_optimizer.load_state_dict(ckpt["actor_opt"])
        self.critic_optimizer.load_state_dict(ckpt["critic_opt"])
        self.update_count = ckpt["updates"]
        if "obs_norm" in ckpt:
            self.obs_norm.load_state_dict(ckpt["obs_norm"])
        self.reset_hidden()
        print(f"[R-PPO] Loaded ← {path}")
