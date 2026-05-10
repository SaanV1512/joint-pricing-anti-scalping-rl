

from __future__ import annotations

import math
import random
from collections import deque
from typing import Deque, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from environment import N_ACTIONS


# ══════════════════════════════════════════════════════════════════════════════
# 1.  NoisyLinear Layer
# ══════════════════════════════════════════════════════════════════════════════

class NoisyLinear(nn.Module):
    """
    Factorised Noisy Linear layer for parameter-space exploration.

    Replaces ε-greedy with learned, state-dependent exploration noise:
        y = (μ_w + σ_w ⊙ ε_w) x + (μ_b + σ_b ⊙ ε_b)

    Factorised noise uses p + q random numbers instead of p×q,
    making it practical for large layers.

    Reference: Fortunato et al., 2019 — "Noisy Networks for Exploration"
    """

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.sigma_init   = sigma_init

        # Learnable parameters
        self.weight_mu    = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu      = nn.Parameter(torch.empty(out_features))
        self.bias_sigma   = nn.Parameter(torch.empty(out_features))

        # Noise buffers (not optimized, re-sampled each forward pass)
        self.register_buffer("weight_eps", torch.empty(out_features, in_features))
        self.register_buffer("bias_eps",   torch.empty(out_features))

        self._reset_parameters()
        self.reset_noise()

    def _reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    @staticmethod
    def _scaled_noise(size: int) -> torch.Tensor:
        """Factorised noise: f(x) = sgn(x)·√|x|"""
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self):
        """Resample factorised noise. Call once per forward pass."""
        eps_p = self._scaled_noise(self.in_features)
        eps_q = self._scaled_noise(self.out_features)
        self.weight_eps.copy_(eps_q.outer(eps_p))
        self.bias_eps.copy_(eps_q)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_eps
            b = self.bias_mu   + self.bias_sigma   * self.bias_eps
        else:
            w = self.weight_mu
            b = self.bias_mu
        return F.linear(x, w, b)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Rainbow Network (Dueling + NoisyNet)
# ══════════════════════════════════════════════════════════════════════════════

class RainbowNetwork(nn.Module):
    """
    Dueling network with NoisyLinear heads.

    Backbone  : FC(256, LN, ReLU) × 2          [deterministic, shared]
    Value     : NoisyLinear(256→128→1)          [noisy]
    Advantage : NoisyLinear(256→128→n_actions)  [noisy]

    Q(s,a) = V(s) + A(s,a) − mean_a[A(s,a)]
    """

    def __init__(self, state_dim: int = 6, n_actions: int = N_ACTIONS,
                 hidden: int = 256, sigma_init: float = 0.5):
        super().__init__()
        self.n_actions = n_actions

        # Deterministic shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )

        # Noisy Value stream  V(s)
        self.value_hidden = NoisyLinear(hidden, 128, sigma_init)
        self.value_out    = NoisyLinear(128, 1,   sigma_init)

        # Noisy Advantage stream  A(s, a)
        self.adv_hidden   = NoisyLinear(hidden, 128, sigma_init)
        self.adv_out      = NoisyLinear(128, n_actions, sigma_init)

        self._init_backbone()

    def _init_backbone(self):
        for m in self.backbone.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        v = F.relu(self.value_hidden(h))
        v = self.value_out(v)                    # (B, 1)
        a = F.relu(self.adv_hidden(h))
        a = self.adv_out(a)                      # (B, A)
        return v + a - a.mean(dim=1, keepdim=True)

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()


# ══════════════════════════════════════════════════════════════════════════════
# 3.  N-step Transition Buffer
# ══════════════════════════════════════════════════════════════════════════════

class NStepBuffer:
    """
    Accumulates n consecutive transitions and computes the n-step return.

    G_t^n = r_t + γ·r_{t+1} + ... + γ^{n-1}·r_{t+n-1} + γ^n · V(s_{t+n})

    Flushed transitions are pushed to the PER replay buffer.
    """

    def __init__(self, n: int = 3, gamma: float = 0.99):
        self.n     = n
        self.gamma = gamma
        self.buf:  Deque = deque()

    def push(self, state, action, reward, next_state, done) -> Optional[Tuple]:
        """
        Push one transition.  Returns a completed n-step transition if ready,
        otherwise None.  Always flushes on episode end.
        """
        self.buf.append((state, action, reward, next_state, done))
        if done:
            # Flush all remaining transitions in buf
            results = []
            while self.buf:
                results.append(self._pop())
            return results   # list of completed transitions
        if len(self.buf) >= self.n:
            return [self._pop()]
        return None

    def _pop(self) -> Tuple:
        s0, a0, _, _, _ = self.buf[0]
        n_return = 0.0
        for i, (_, _, r, _, d) in enumerate(self.buf):
            n_return += (self.gamma ** i) * r
            if d:
                break
        _, _, _, sn, dn = self.buf[-1]
        self.buf.popleft()
        return (s0, a0, n_return, sn, dn)

    def clear(self):
        self.buf.clear()


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Prioritized Experience Replay — stable priority-array sampling
# ══════════════════════════════════════════════════════════════════════════════

class PrioritizedReplayBuffer:
    """
    Ring-buffer PER with numpy priority-proportional sampling.
    Uses np.random.choice(p=probs) instead of a SumTree —
    identical algorithmic guarantees, numerically stable.
    Reference: Schaul et al., 2015 — "Prioritized Experience Replay"
    """

    def __init__(self, capacity: int = 100_000, alpha: float = 0.6,
                 beta_start: float = 0.4, beta_end: float = 1.0,
                 beta_steps: int = 80_000, eps: float = 1e-6):
        self.capacity   = capacity
        self.alpha      = alpha
        self.beta_start = beta_start
        self.beta_end   = beta_end
        self.beta_steps = beta_steps
        self.eps        = eps
        # Ring-buffer storage
        self._s:   List = []
        self._a:   List = []
        self._r:   List = []
        self._ns:  List = []
        self._d:   List = []
        self._pri  = np.zeros(capacity, dtype=np.float32)
        self.ptr  = 0
        self.size = 0
        self.step = 0

    def push(self, *transition):
        s, a, r, ns, d = transition
        p = float(self._pri[:self.size].max()) if self.size else 1.0
        p = max(p, 1e-6)
        if self.size < self.capacity:
            self._s.append(s);  self._a.append(a);  self._r.append(r)
            self._ns.append(ns); self._d.append(d)
            self.size += 1
        else:
            self._s[self.ptr]  = s;  self._a[self.ptr]  = a
            self._r[self.ptr]  = r;  self._ns[self.ptr] = ns
            self._d[self.ptr]  = d
        self._pri[self.ptr] = p
        self.ptr = (self.ptr + 1) % self.capacity

    def sample(self, batch_size: int):
        self.step += 1
        beta = min(self.beta_end,
                   self.beta_start + (self.beta_end - self.beta_start)
                   * self.step / self.beta_steps)
        pris  = self._pri[:self.size] ** self.alpha
        total = pris.sum()
        probs = pris / total
        # clamp batch_size to avoid replace=False error if buffer is tiny
        bs    = min(batch_size, self.size)
        idx   = np.random.choice(self.size, size=bs, replace=False, p=probs)

        weights = (self.size * probs[idx]) ** (-beta)
        weights = (weights / weights.max()).astype(np.float32)

        s  = np.array([self._s[i]  for i in idx], dtype=np.float32)
        a  = np.array([self._a[i]  for i in idx], dtype=np.int64)
        r  = np.array([self._r[i]  for i in idx], dtype=np.float32)
        ns = np.array([self._ns[i] for i in idx], dtype=np.float32)
        d  = np.array([self._d[i]  for i in idx], dtype=np.float32)
        return s, a, r, ns, d, list(idx), weights

    def update_priorities(self, indices: List[int], td_errors: np.ndarray):
        for i, err in zip(indices, np.abs(td_errors)):
            self._pri[i] = float(err) + self.eps

    def __len__(self): return self.size


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Rainbow DQN Agent
# ══════════════════════════════════════════════════════════════════════════════

class DQNAgent:
    """
    Rainbow DQN = Double + Dueling + PER + N-step + NoisyNet.

    No ε-greedy — exploration driven entirely by NoisyNet weights.
    Noise is reset each time the policy network is queried.

    Hyperparameters (tuned for IPL environment)
    -------------------------------------------
    gamma              : 0.99
    n_step             : 3       (n-step return horizon)
    lr                 : 3e-4
    batch_size         : 256
    buffer_capacity    : 200 000
    target_update_freq : 150     (hard copy every N updates)
    per_alpha          : 0.6
    per_beta_start     : 0.4
    hidden             : 256
    sigma_init         : 0.5     (NoisyNet initial σ)
    """

    def __init__(
        self,
        state_dim:          int   = 6,
        n_actions:          int   = N_ACTIONS,
        gamma:              float = 0.99,
        n_step:             int   = 3,
        lr:                 float = 3e-4,
        batch_size:         int   = 256,
        buffer_capacity:    int   = 200_000,
        target_update_freq: int   = 150,
        per_alpha:          float = 0.6,
        per_beta_start:     float = 0.4,
        per_beta_steps:     int   = 80_000,
        hidden:             int   = 256,
        sigma_init:         float = 0.5,
        device:             str   = "cpu",
        # kept for API compatibility — not used (NoisyNet replaces ε-greedy)
        eps_start: float = 1.0,
        eps_end:   float = 0.0,
        eps_decay: int   = 1,
    ):
        self.n_actions          = n_actions
        self.gamma              = gamma
        self.n_step             = n_step
        self.batch_size         = batch_size
        self.target_update_freq = target_update_freq
        self.device             = torch.device(device)

        # Networks
        self.policy_net = RainbowNetwork(state_dim, n_actions, hidden, sigma_init).to(self.device)
        self.target_net = RainbowNetwork(state_dim, n_actions, hidden, sigma_init).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer + scheduler
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr, eps=1.5e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=500, T_mult=1, eta_min=lr * 0.05
        )

        # N-step buffer + PER replay
        self.n_buf  = NStepBuffer(n=n_step, gamma=gamma)
        self.memory = PrioritizedReplayBuffer(
            capacity   = buffer_capacity,
            alpha      = per_alpha,
            beta_start = per_beta_start,
            beta_steps = per_beta_steps,
        )

        self.steps_done = 0
        self.updates    = 0

        # Logging
        self.loss_history:     List[float] = []
        self.q_value_history:  List[float] = []
        self.epsilon_history:  List[float] = []   # kept for compat — always 0 (NoisyNet)

    # ── Action selection (NoisyNet-driven exploration) ────────────────────────
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        self.policy_net.train(training)
        if training:
            self.policy_net.reset_noise()
            self.steps_done += 1
            self.epsilon_history.append(0.0)  # NoisyNet ≡ ε=0

        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.policy_net(s)
            if training:
                self.q_value_history.append(float(q.max().item()))
            return int(q.argmax(dim=1).item())

    # ── Store transition (via N-step buffer) ─────────────────────────────────
    def store(self, state, action, reward, next_state, done) -> None:
        completed = self.n_buf.push(state, action, reward, next_state, done)
        if completed:
            for t in completed:
                self.memory.push(*t)

    # ── Gradient update ───────────────────────────────────────────────────────
    def update(self) -> Optional[float]:
        if len(self.memory) < self.batch_size:
            return None

        s, a, r, ns, d, indices, weights = self.memory.sample(self.batch_size)

        s  = torch.tensor(s,  dtype=torch.float32, device=self.device)
        a  = torch.tensor(a,  dtype=torch.long,    device=self.device)
        r  = torch.tensor(r,  dtype=torch.float32, device=self.device)
        ns = torch.tensor(ns, dtype=torch.float32, device=self.device)
        d  = torch.tensor(d,  dtype=torch.float32, device=self.device)
        w  = torch.tensor(weights, dtype=torch.float32, device=self.device)

        # Reset NoisyNet noise before policy forward
        self.policy_net.reset_noise()
        self.target_net.reset_noise()

        # Current Q values
        q_curr = self.policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # Double DQN target — n-step adjusted discount
        with torch.no_grad():
            best_a = self.policy_net(ns).argmax(dim=1, keepdim=True)
            q_next = self.target_net(ns).gather(1, best_a).squeeze(1)
            # n-step discount: γ^n instead of γ
            q_target = r + (self.gamma ** self.n_step) * q_next * (1.0 - d)

        td_errors = (q_curr - q_target).detach().cpu().numpy()

        # IS-weighted Huber loss
        loss_elem = F.smooth_l1_loss(q_curr, q_target, reduction="none")
        loss      = (w * loss_elem).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.scheduler.step()

        # Update PER priorities
        self.memory.update_priorities(indices, td_errors)

        self.updates += 1
        loss_val = float(loss.item())
        self.loss_history.append(loss_val)

        if self.updates % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss_val

    # ── Save / load ───────────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        torch.save({
            "policy_state": self.policy_net.state_dict(),
            "target_state": self.target_net.state_dict(),
            "optimizer":    self.optimizer.state_dict(),
            "steps_done":   self.steps_done,
            "updates":      self.updates,
        }, path)
        print(f"[Rainbow] Saved -> {path}")

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(ckpt["policy_state"])
        self.target_net.load_state_dict(ckpt["target_state"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.steps_done = ckpt["steps_done"]
        self.updates    = ckpt["updates"]
        print(f"[Rainbow] Loaded <- {path}")
