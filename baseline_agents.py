

from __future__ import annotations
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 1: Static Pricing Agent
# ─────────────────────────────────────────────────────────────────────────────

class StaticAgent:
    """
    Baseline 1: Static pricing + fixed purchase limits.

    Never changes price or limit. Action = 8 (keep price, maintain limit).
    Represents the current real-world BCCI/BookMyShow approach.
    """
    def select_action(self, state: np.ndarray) -> int:
        return 8   # keep price (2) * 3 + maintain limit (2) = 8


# ─────────────────────────────────────────────────────────────────────────────
# Baseline 2: Rule-Based Dynamic Agent
# ─────────────────────────────────────────────────────────────────────────────

class RuleBasedAgent:
    """
    Baseline 2: Rule-based dynamic pricing and limit control.

    Uses simple threshold-based heuristics derived from domain knowledge.
    This is a deterministic non-learning agent that mirrors current
    "smart" ticket systems like TicketMaster dynamic pricing.

    State indices (from environment):
        0 : inventory_norm
        1 : time_norm
        2 : demand_norm
        3 : suspicion_norm   ← S'
        4 : avg_tickets_norm
        5 : price_norm

    Decision rules:
        If suspicion > 0.6 AND inventory < 50% → Increase price + Tighten limit
        If suspicion > 0.4                       → Increase price + Maintain limit
        If demand > 0.7 AND inventory < 30%      → Increase price + Tighten limit
        If demand < 0.2 AND inventory > 70%      → Decrease price + Relax limit
        Otherwise                                → Keep price + Maintain limit
    """

    def select_action(self, state: np.ndarray) -> int:
        inv_norm  = float(state[0])
        time_norm = float(state[1])
        dem_norm  = float(state[2])
        susp_norm = float(state[3])

        # Encode action: price_idx * 3 + limit_idx
        # price: 0=increase, 1=decrease, 2=keep
        # limit: 0=tighten,  1=relax,    2=maintain

        if susp_norm > 0.6 and inv_norm < 0.5:
            return 0 * 3 + 0   # increase price + tighten limit = 0

        if susp_norm > 0.4:
            return 0 * 3 + 2   # increase price + maintain limit = 2

        if dem_norm > 0.7 and inv_norm < 0.3:
            return 0 * 3 + 0   # increase price + tighten limit = 0

        if dem_norm < 0.2 and inv_norm > 0.7:
            return 1 * 3 + 1   # decrease price + relax limit = 4

        return 2 * 3 + 2       # keep price + maintain limit = 8


# ─────────────────────────────────────────────────────────────────────────────
# Convenience wrapper for evaluate.py
# ─────────────────────────────────────────────────────────────────────────────

def get_baseline_action(agent_type: str, state: np.ndarray) -> int:
    """Get action from a named baseline agent."""
    if agent_type == "static":
        return StaticAgent().select_action(state)
    elif agent_type == "rule_based":
        return RuleBasedAgent().select_action(state)
    else:
        raise ValueError(f"Unknown baseline: {agent_type}")
