from __future__ import annotations


def maintenance_cost(base_cost: float, health: float, gamma: float = 1.0) -> float:
    """
    Health in [0,1]. Lower health => larger workscope => higher shop cost.
    Linear model: C = base_cost * (1 + gamma*(1-health))
    """
    h = max(0.0, min(1.0, float(health)))
    return float(base_cost * (1.0 + gamma * (1.0 - h)))