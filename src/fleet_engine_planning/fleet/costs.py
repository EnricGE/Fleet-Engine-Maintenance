from __future__ import annotations


def maintenance_cost(base_cost: float, health: float, gamma_health_cost: float) -> float:
    """ Maintenance cost depends on health index"""

    return base_cost + (1-health) * base_cost