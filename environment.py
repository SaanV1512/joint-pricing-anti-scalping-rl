import gymnasium as gym
from gymnasium import spaces
import numpy as np

from suspicion_score import UserBehavior, EnsembleSuspicionSystem

PRICE_ACTIONS = ["increase", "decrease", "keep"]
LIMIT_ACTIONS = ["tighten", "relax", "maintain"]
N_ACTIONS = 9

def decode_action(action: int):
    return PRICE_ACTIONS[action // 3], LIMIT_ACTIONS[action % 3]


class AdaptiveScalperBot:
    """
    Adversarial bot that adapts its price tolerance based on
    observed market conditions. Simulates a scalper that
    dynamically adjusts its willingness-to-pay as it observes
    the RL agent raising prices.

    Attributes
    ----------
    base_tolerance   : float — initial price tolerance multiplier (e.g. 3.5×)
    adaptation_rate  : float — how fast the bot adapts to price changes
    observed_price   : float — EWMA of observed prices
    """

    def __init__(self, base_tolerance: float = 3.5, adaptation_rate: float = 0.05):
        self.base_tolerance    = base_tolerance
        self.adaptation_rate   = adaptation_rate
        self.observed_price    = None    # initialised on first step
        self.current_tolerance = base_tolerance

    def update(self, current_price: float, base_price: float):
        """Update observed price EWMA and adapt tolerance."""
        if self.observed_price is None:
            self.observed_price = current_price
        # Track the normalized price trend
        self.observed_price = (
            (1 - self.adaptation_rate) * self.observed_price +
            self.adaptation_rate * current_price
        )
        # If agent is consistently raising prices, bot lowers its tolerance
        # (bots exit the market when prices are too high, simulating realistic behavior)
        ratio = self.observed_price / base_price
        self.current_tolerance = max(1.5, self.base_tolerance - 0.3 * (ratio - 1.0))

    def will_buy(self, current_price: float, base_price: float) -> bool:
        return current_price <= base_price * self.current_tolerance


class IPLTicketingEnv(gym.Env):
    """
    Advanced IPL Ticketing Simulation Environment.

    Parameters
    ----------
    total_inventory : int   — starting ticket count
    match_duration  : int   — total time-steps per episode
    base_price      : float — starting price (₹)
    scalper_ratio   : float — fraction of arrivals that are bots (base)
    adaptive_bots   : bool  — if True, bots adapt to pricing policy
    seed            : int
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        total_inventory: int   = 2000,
        match_duration:  int   = 100,
        base_price:      float = 1000.0,
        scalper_ratio:   float = 0.30,
        adaptive_bots:   bool  = True,
        seed:            int   = 42,
    ):
        super().__init__()
        self.total_inventory = total_inventory
        self.match_duration  = match_duration
        self.base_price      = base_price
        self.scalper_ratio   = scalper_ratio
        self.adaptive_bots   = adaptive_bots
        self._seed           = seed

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32)
        self.action_space      = spaces.Discrete(N_ACTIONS)

        # ── Rihan's reward weights (α, β, γ, δ) ──────────────────────────────
        self.alpha = 1.0   # revenue
        self.beta  = 0.6   # fairness
        self.gamma = 0.8   # scalper penalty
        self.delta = 0.4   # dissatisfaction

        # ── Veda's Suspicion Score System (Hybrid ML) ─────────────────────────
        self.suspicion_system = EnsembleSuspicionSystem(ml_weight=0.7)
        # Generate synthetic warmup data to train the Isolation Forest
        np.random.seed(seed)
        warmup_data = []
        for _ in range(800):
            warmup_data.append(UserBehavior(np.random.randint(1, 3), np.random.uniform(0.1, 1.5), np.random.randint(2, 6), 0))
        for _ in range(200):
            warmup_data.append(UserBehavior(4, np.random.uniform(5.0, 15.0), 1, 1))
        self.suspicion_system.warmup(warmup_data)

        self.reset()

    # ── Demand model (Saanvi) ─────────────────────────────────────────────────
    def _arrival_rate(self, time_norm: float) -> float:
        """
        Three-phase demand model:
          1. Launch rush   (t < 15%) — massive spike, bots + genuine users flood in
          2. Mid-sale      (15–80%)  — exponential decay
          3. Closing rush  (t > 80%) — small bump as deadline approaches
        """
        if time_norm < 0.15:
            return 80.0 + 20.0 * (1 - time_norm / 0.15)  # peak ~100 at t=0
        elif time_norm < 0.80:
            return 20.0 * np.exp(-3.0 * (time_norm - 0.15))
        else:
            # Closing rush: last-minute genuine users
            return 5.0 + 10.0 * (time_norm - 0.80) / 0.20

    def _scalper_multiplier(self, time_norm: float) -> float:
        """Bots are most active during launch rush."""
        if time_norm < 0.15:
            return 1.5
        elif time_norm < 0.50:
            return 1.0
        else:
            return 0.5  # bots give up if prices are high mid-sale

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed or self._seed)
        self.rng             = np.random.default_rng(seed or self._seed)
        self.inventory       = self.total_inventory
        self.time_step       = 0
        self.current_price   = self.base_price
        self.purchase_limit  = 4
        self.total_revenue   = 0.0
        self.suspicion_score = 0.0

        if self.adaptive_bots:
            self.bot = AdaptiveScalperBot(base_tolerance=3.5, adaptation_rate=0.05)
        else:
            self.bot = None

        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        time_norm   = self.time_step / self.match_duration
        demand_raw  = self.rng.poisson(self._arrival_rate(time_norm))
        demand_norm = np.clip(demand_raw / 120.0, 0.0, 1.0)

        return np.array([
            self.inventory / self.total_inventory,           # s0: inventory
            time_norm,                                        # s1: time
            demand_norm,                                      # s2: demand
            np.clip(self.suspicion_score, 0.0, 1.0),         # s3: suspicion S'
            np.clip(self.purchase_limit / 4.0, 0.0, 1.0),   # s4: avg limit
            np.clip(self.current_price / (self.base_price * 4), 0.0, 1.0),  # s5: price
        ], dtype=np.float32)

    def step(self, action: int):
        price_act, limit_act = decode_action(int(action))
        time_norm = self.time_step / self.match_duration

        # ── 1. Price update ───────────────────────────────────────────────────
        step_pct = 0.10
        if price_act == "increase":
            self.current_price = min(self.current_price * (1 + step_pct),
                                     self.base_price * 4.0)
        elif price_act == "decrease":
            self.current_price = max(self.current_price * (1 - step_pct),
                                     self.base_price * 0.5)

        # ── 2. Limit update ───────────────────────────────────────────────────
        if limit_act == "tighten":
            self.purchase_limit = max(1, self.purchase_limit - 1)
        elif limit_act == "relax":
            self.purchase_limit = min(4, self.purchase_limit + 1)

        # ── 3. Update adaptive bot ────────────────────────────────────────────
        if self.bot is not None:
            self.bot.update(self.current_price, self.base_price)

        # ── 4. Simulate arrivals (Saanvi's demand model) ─────────────────────
        base_λ        = self._arrival_rate(time_norm)
        scalper_mult  = self._scalper_multiplier(time_norm)
        current_scalper_ratio = min(0.60, self.scalper_ratio * scalper_mult)

        n_arrivals = int(self.rng.poisson(base_λ))
        n_scalpers = int(n_arrivals * current_scalper_ratio)
        n_genuine  = n_arrivals - n_scalpers

        revenue = fair_tickets = scalper_tickets = 0.0
        step_behaviors = []

        # ── Genuine users — price-elastic with a smooth demand curve ──────────
        for _ in range(n_genuine):
            # Each user has their own willingness-to-pay (log-normal distributed)
            wtp = self.base_price * float(self.rng.lognormal(0.0, 0.25))
            if self.current_price <= wtp:
                attempts = int(self.rng.integers(1, 4))
                tickets  = min(int(self.rng.integers(1, 3)), self.purchase_limit,
                               self.inventory)
                if tickets > 0:
                    self.inventory  -= tickets
                    revenue         += tickets * self.current_price
                    fair_tickets    += tickets
                    speed = tickets / max(attempts, 1)
                    bought_max = 1 if tickets == self.purchase_limit else 0
                    step_behaviors.append(UserBehavior(tickets, speed, attempts, bought_max))

        # ── Scalper bots — adaptive, price-inelastic up to their tolerance ────
        for _ in range(n_scalpers):
            buys = (
                self.bot.will_buy(self.current_price, self.base_price)
                if self.bot else self.current_price <= self.base_price * 3.5
            )
            if buys:
                tickets = min(self.purchase_limit, self.inventory)
                if tickets > 0:
                    self.inventory  -= tickets
                    revenue         += tickets * self.current_price
                    scalper_tickets += tickets
                    step_behaviors.append(UserBehavior(tickets, tickets / 1.0, 1, 1))

        self.total_revenue += revenue
        fairness = fair_tickets / max(fair_tickets + scalper_tickets, 1)
        scalper_rate = scalper_tickets / max(fair_tickets + scalper_tickets, 1)

        # ── 5. Veda's Suspicion Score System (Hybrid ML) ──────────────────────
        S_prime = self.suspicion_system.aggregate_suspicion(step_behaviors)
        self.suspicion_score = 0.5 * self.suspicion_score + 0.5 * S_prime

        # ── 6. Rihan's Multi-Objective Reward (Equation 5) ───────────────────
        R_rev   = np.clip(revenue / (self.base_price * 50), 0, 1)
        R_fair  = np.clip(fair_tickets / max(n_genuine, 1), 0, 1)
        R_scalp = np.clip(scalper_tickets / max(n_scalpers * self.purchase_limit, 1), 0, 1)
        R_diss  = np.clip((self.current_price - self.base_price) / (self.base_price * 2), 0, 1)

        # Inventory urgency: penalise for wasting tickets at end of episode
        inv_frac = self.inventory / self.total_inventory
        R_waste  = 0.2 * inv_frac * (time_norm ** 2)   # grows towards end

        reward = (
            self.alpha * R_rev
            + self.beta  * R_fair
            - self.gamma * R_scalp
            - self.delta * R_diss
            - R_waste
        )

        self.time_step += 1
        terminated      = (self.inventory <= 0) or (self.time_step >= self.match_duration)

        info = {
            "revenue":         revenue,
            "fair_tickets":    fair_tickets,
            "scalper_tickets": scalper_tickets,
            "fairness":         fairness,
            "scalper_rate":     scalper_rate,
            "price":           self.current_price,
            "limit":           self.purchase_limit,
            "suspicion":       self.suspicion_score,
            "inventory":       self.inventory,
            "n_genuine":       n_genuine,
            "n_scalpers":      n_scalpers,
        }
        return self._get_obs(), float(reward), terminated, False, info
