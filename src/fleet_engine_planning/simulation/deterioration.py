from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from fleet_engine_planning.fleet.engine import Fleet
from fleet_engine_planning.preprocessing.schema import DeteriorationParams


def _monthly_delta_health(
    rng: np.random.Generator,
    params: DeteriorationParams,
) -> float:
    """
    Sample the health deterioration for one month.

    mean = mu_base + mu_per_1000km * (km_per_month / 1000)
    delta ~ Normal(mean, sigma), truncated at 0 (no negative deterioration)
    """
    mean = params.mu_base + params.mu_per_1000km * (params.km_per_month / 1000.0)
    delta = rng.normal(loc=mean, scale=params.sigma)
    return float(max(0.0, delta))


def sample_deterioration_deltas(
    fleet: Fleet,
    horizon_months: int,
    n_scenarios: int,
    params: DeteriorationParams,
    seed: int = 42,
) -> Dict[Tuple[str, int, int], float]:
    """
    Returns dh[(engine_id, t, s)] for t=1..T and s=0..S-1
    where dh is the deterioration applied from start of month t to start of month t+1.
    """
    if horizon_months < 1:
        raise ValueError("horizon_months must be >= 1")
    if n_scenarios < 1:
        raise ValueError("n_scenarios must be >= 1")

    rng = np.random.default_rng(seed)
    T = int(horizon_months)
    S = int(n_scenarios)

    dh: Dict[Tuple[str, int, int], float] = {}
    for e in fleet.engines:
        for s in range(S):
            for t in range(1, T + 1):
                dh[(e.engine_id, t, s)] = _monthly_delta_health(rng, params)

    return dh


def health_at_start_of_month(
    h0: float,
    dh: Dict[Tuple[str, int, int], float],
    engine_id: str,
    month: int,
    scenario: int,
) -> float:
    """
    Health at the start of given month (1..T+1) with no maintenance.
    month=1 -> h0
    month=2 -> h0 - dh(month=1)
    etc.
    """
    h = float(h0)
    for t in range(1, month):  # apply months 1..(month-1)
        h -= dh[(engine_id, t, scenario)]
        if h <= 0.0:
            return 0.0
    return max(0.0, min(1.0, h))


def operable_under_single_shop(
    h0: float,
    dh: Dict[Tuple[str, int, int], float],
    engine_id: str,
    month: int,
    scenario: int,
    shop_month: int,
    h_min: float,
) -> int:
    """
    Returns 1 if engine is operable in 'month' (1..T) under scenario,
    given a single shop visit at 'shop_month' (0..T).

    Convention:
      - If shop_month = m >= 1: engine is IN SHOP during month m -> not operable.
      - Engine returns at start of month m+1 with health reset to 1.0.
      - Operability check is at start of month.

    If shop_month = 0: no shop.
    """
    if shop_month == 0:
        h = health_at_start_of_month(h0, dh, engine_id, month, scenario)
        return 1 if h >= h_min else 0

    if month == shop_month:
        return 0  # in shop

    if month < shop_month:
        h = health_at_start_of_month(h0, dh, engine_id, month, scenario)
        return 1 if h >= h_min else 0

    # month > shop_month: reset at start of month shop_month+1
    # health at start of month (shop_month+1) is 1.0
    h = 1.0
    # apply deterioration from months (shop_month+1) .. (month-1)
    for t in range(shop_month + 1, month):
        h -= dh[(engine_id, t, scenario)]
        if h <= 0.0:
            return 0
    return 1 if h >= h_min else 0