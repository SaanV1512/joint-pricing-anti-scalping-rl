# IPL RL Ticketing System — Complete Paper Sections
## Reinforcement Learning for Joint Dynamic Pricing and Anti-Scalping Control in IPL Ticketing Systems

---

## II. SUSPICION SCORE FORMULATION *(Veda)*

### A. Feature Design

We define a behavioral feature vector **f** = (f₁, f₂, f₃, f₄) captured per user session during the ticket sale window:

| Feature | Symbol | Description |
|---------|--------|-------------|
| Tickets bought | f₁ | Total tickets purchased in one session |
| Purchase speed | f₂ | Tickets acquired per minute |
| Attempt count | f₃ | Number of purchase attempts (API calls) |
| Max-limit buyer | f₄ | Binary: 1 if user always buys the maximum allowed limit |

**Design rationale:**
- Genuine users typically buy 1–2 tickets slowly, with multiple browsing attempts before purchasing.
- Scalper bots are designed to buy the maximum limit in a single, instantaneous API call.
- f₄ is a particularly strong discriminator: genuine users rarely hit the exact maximum limit; bots always do by design.

### B. Score Calculation (Equation 2)

The raw suspicion score is computed as a weighted linear combination:

$$S = w_1 f_1 + w_2 f_2 + w_3 f_3 + w_4 f_4$$

We set **w₁ = 0.30, w₂ = 0.30, w₃ = 0.20, w₄ = 0.20** based on domain analysis. Purchase count and speed are weighted most heavily as they are the strongest empirical indicators of automated bot behavior.

### C. Normalization (Equation 3)

The raw score is normalized using an empirically determined maximum S_max = 10.0, producing a continuous suspicion indicator S' ∈ [0, 1]:

$$S' = \frac{S}{S_{max}}$$

S' is then incorporated as a direct feature in the RL agent's state vector, providing a smooth, differentiable signal representing the population-level threat at each time step.

**Classification threshold:** Users with S' ≥ 0.50 are flagged as "Suspected Scalpers." This threshold is tunable.

---

## III. PROBLEM FORMULATION *(Rihan)*

We model the IPL ticketing system as a Markov Decision Process **M = (S, A, T, R)**.

### A. State Space

The state vector **s ∈ ℝ⁶** (all features normalized to [0, 1]):

| Index | Feature | Normalization |
|-------|---------|---------------|
| s₀ | Remaining inventory | inventory / total_inventory |
| s₁ | Time elapsed | time_step / match_duration |
| s₂ | Current demand rate | arrivals / 100 |
| s₃ | Suspicion score S' | via Equation 3 |
| s₄ | Avg tickets per user | purchase_limit / 4 |
| s₅ | Current ticket price | price / (base_price × 4) |

### B. Action Space

The agent executes a **joint discrete action** from 9 combinations:

$$a = (a_{price},\ a_{limit}) \in \{0,1,2\}^2, \quad |A| = 9$$

| Action Index | Price Action | Limit Action |
|---|---|---|
| 0 | Increase (+10%) | Tighten (limit−1) |
| 1 | Increase (+10%) | Relax (limit+1) |
| 2 | Increase (+10%) | Maintain |
| 3 | Decrease (−10%) | Tighten |
| 4 | Decrease (−10%) | Relax |
| 5 | Decrease (−10%) | Maintain |
| 6 | Keep | Tighten |
| 7 | Keep | Relax |
| 8 | Keep | Maintain |

### C. Transition Dynamics

State transitions **T(s' | s, a)** depend on three stochastic factors:
1. **User arrival rates:** Modeled with a Poisson distribution λ(t), with a heavy spike (λ=80) during the first 15% of the sale window (launch rush), decaying exponentially thereafter.
2. **Demand elasticity:** Genuine users are price-sensitive with individual sensitivity thresholds drawn from U(0.7, 1.3) × base_price. Scalper bots are price-inelastic up to 3.5× base price.
3. **Bot behavior:** Scalper bots always attempt to buy the maximum purchase limit in a single, fast attempt.

### D. Multi-Objective Reward Function (Equation 5)

$$R = \alpha R_{rev} + \beta R_{fair} - \gamma R_{scalp} - \delta R_{diss}$$

| Component | Formula | Weight | Justification |
|-----------|---------|--------|---------------|
| R_rev | revenue / (base_price × 40) | α = 1.0 | Primary economic objective |
| R_fair | fair_tickets / n_genuine | β = 0.6 | Social equity enforcement |
| R_scalp | scalper_tickets / (n_bots × limit) | γ = 0.8 | Scalper suppression penalty |
| R_diss | (price − base_price) / (base_price × 2) | δ = 0.4 | User dissatisfaction penalty |

**Weight Justification:**
- α > β ensures the system remains commercially viable.
- γ > δ ensures scalper suppression is prioritized over dissatisfaction penalty.
- The β/γ balance (0.6 vs 0.8) deliberately leans toward scalper suppression over pure fairness, since fairness naturally improves as scalper rate decreases.

---

## IV. PROPOSED METHOD *(Shivansh)*

### A. Deep Q-Network (DQN) Architecture

The DQN learns an action-value function Q(s, a; θ) mapping state-action pairs to expected returns.

**Network architecture:**
```
Input:  s ∈ ℝ⁶
FC(128) → ReLU → FC(128) → ReLU → FC(64) → ReLU → FC(9)
Output: Q-values for each of the 9 joint actions
```

**Training procedure:**
- **Experience Replay:** Circular buffer (capacity = 50,000 transitions), mini-batch size = 64
- **Target Network:** Hard-updated every 500 gradient steps (prevents oscillation)
- **Exploration:** ε-greedy, ε: 1.0 → 0.05 decaying exponentially over 20,000 steps
- **Loss:** MSE loss, Adam optimizer (lr = 1×10⁻³), gradient clipping (max norm = 10)

### B. PPO Agent Architecture

PPO uses a shared Actor-Critic backbone to directly optimize the policy while constraining update size.

**Network architecture:**
```
Shared: FC(128) → ReLU → FC(128) → ReLU
Actor head:  FC(128 → 9) → Softmax   [policy π(a|s; θ)]
Critic head: FC(128 → 1)             [value V(s; φ)]
```

**Training procedure:**
- **Rollout collection:** 512 steps per update cycle
- **GAE:** λ = 0.95, γ = 0.99 for advantage estimation
- **Clipped objective:** clip(r_t, 1−ε, 1+ε) with ε = 0.2
- **Loss:** L = −L_clip + 0.5·L_val − 0.01·L_entropy
- **Optimizer:** Adam (lr = 3×10⁻⁴), K = 10 epochs per rollout, gradient clipping = 0.5

---

## V. SIMULATION ENVIRONMENT *(Saanvi)*

### A. User Types

**Genuine Users:**
- Arrival: Poisson(λ=80) during launch rush (t < 15%), exponentially declining after
- Price sensitivity: Individual threshold sampled from U(0.7, 1.3) × base_price
- Purchase quantity: 1–2 tickets per session
- Attempt behavior: 1–2 attempts before purchasing (browsing behavior)

**Scalper Bots:**
- Traffic share: 30% baseline; inflated to 45% during launch rush
- Price inelasticity: Will buy at any price up to 3.5× base price
- Purchase behavior: Always buys maximum allowed purchase limit
- Speed: Single-attempt, maximum-speed acquisition (f₂ is very high)

### B. Demand Spike Model

The arrival rate λ(t) models real IPL ticket sale dynamics:

$$\lambda(t) = \begin{cases} 80 & \text{if } t/T < 0.15 \text{ (launch rush)} \\ 20 \cdot e^{-3(t/T - 0.15)} & \text{otherwise} \end{cases}$$

This produces a realistic scenario where ~60% of all traffic arrives in the first 15% of the sale window—consistent with empirical IPL BookMyShow patterns.

### C. Inventory Dynamics

- Total inventory: 2,000 tickets (simulating a section/stand)
- Inventory is consumed atomically: tickets are only allocated if inventory ≥ requested quantity
- Episode terminates when inventory = 0 or time = T

---

## VI. EVALUATION METRICS *(Veda)*

We evaluate on three primary metrics averaged over 50 independent episodes:

| Metric | Formula | Ideal Direction |
|--------|---------|-----------------|
| Total Reward | Σ R_t per episode | Maximize |
| Total Revenue | Σ (tickets × price) | Maximize |
| Fairness Index | fair_tickets / total_sold | Maximize (→ 1.0) |
| Scalper Acquisition Rate | scalper_tickets / total_sold | Minimize (→ 0.0) |
| Average Price | Mean price over episode | Context-dependent |

---

## VII. BASELINE COMPARISON *(Veda)*

Three systems are compared:

| System | Description | Reward | Revenue | Fairness | ScalperRate |
|--------|-------------|--------|---------|----------|-------------|
| **Static** | Fixed ₹1,000 price, limit=4 | 8.06 | ₹20,00,000 | 0.196 | 0.804 |
| **Rule-Based** | Heuristic thresholds on suspicion/demand | -20.0 | ₹21,54,512 | 0.164 | 0.836 |
| **DQN (Ours)** | Joint RL control | **22.30** | ₹16,45,618 | **0.495** | **0.505** |
| **PPO (Ours)** | Joint RL control | 13.27 | ₹13,18,046 | 0.480 | 0.520 |

**Key finding:** The DQN agent achieves a **153% improvement in total reward** over the Static baseline and reduces the scalper acquisition rate from 80.4% to 50.5% — a **37% reduction in scalper success rate**. The rule-based agent, despite earning higher raw revenue, achieves the worst reward due to its price-gouging behavior (avg ₹1,487) heavily penalized by R_diss, and its high scalper rate.

---

## VIII. EXPECTED CONTRIBUTIONS

1. **Joint RL Control Framework:** First formulation of ticket price + purchase limit as a unified 9-action joint MDP.
2. **Behavioral Suspicion Score Integration:** Real-time S' from behavioral features (f₁–f₄) embedded directly into the RL state — enabling proactive, not reactive, bot suppression.
3. **Realistic IPL Simulation Environment:** Launch-rush demand spikes, price-elastic genuine users, and price-inelastic scalper bots provide an adversarially challenging training ground.
4. **3-Way Baseline Comparison:** Rigorous empirical validation against static and rule-based systems.

---

## IX. CONCLUSION

This work presents a Reinforcement Learning framework for joint optimization of dynamic ticket pricing and anti-scalping purchase limits in IPL ticketing systems. By integrating economic incentives (R_rev), social fairness (R_fair), bot suppression (R_scalp), and user dissatisfaction (R_diss) into a single multi-objective reward function, the DQN agent learns an adaptive policy that outperforms both static and rule-based approaches on all fairness and reward metrics.

The proposed system demonstrates that **proactive market regulation** via RL is significantly more effective than reactive fraud detection. The DQN agent reduces scalper acquisition rates by 37% while maintaining competitive revenue levels.

Future work includes: real-world deployment with live BookMyShow/BCCI data, multi-agent RL extensions where bots adapt to the pricing policy, and Transformer-based world models to better capture sequential demand patterns.
