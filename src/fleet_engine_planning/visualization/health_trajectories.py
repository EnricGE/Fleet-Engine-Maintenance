from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt


def _health_path_under_single_shop(
    h0: float,
    dh: Dict[Tuple[str, int, int], float],
    engine_id: str,
    horizon_months: int,
    scenario: int,
    shop_start_month: int,        # 0..T
    shop_duration_months: int,    # e.g. 2
) -> np.ndarray:
    """
    Returns health at start of each month: months 1..T+1 (length T+1).
    Convention:
      - Shop occupies months [m, m+D-1] (engine unavailable)
      - Health resets to 1.0 at start of month (m+D)
      - Deterioration dh[(eid, t, s)] applies from start of month t -> start of month t+1
    """
    T = int(horizon_months)
    D = int(shop_duration_months)
    m = int(shop_start_month)

    h = np.zeros(T + 1, dtype=float)  # index 0 -> month 1, index T -> month T+1
    health = float(h0)

    # month 1
    h[0] = max(0.0, min(1.0, health))

    for t in range(1, T + 1):  # t is the month index for deterioration application (1..T)
        next_month = t + 1

        # If we are at the return month, reset health
        if m > 0 and next_month == m + D:
            health = 1.0

        # If we are not in shop during month t, apply deterioration
        in_shop = (m > 0) and (m <= t <= m + D - 1)
        if not in_shop:
            health -= float(dh[(engine_id, t, scenario)])

        health = max(0.0, min(1.0, health))
        h[next_month - 1] = health

    return h


def plot_engine_health_trajectories(
    engine_id: str,
    h0: float,
    dh: Dict[Tuple[str, int, int], float],
    horizon_months: int,
    n_scenarios: int,
    shop_start_month: int,          # from optimized schedule
    shop_duration_months: int,
    h_min: float,
    out_path: Path,
    n_paths: int = 30,              # 20–50 suggested
    seed: int = 0,
) -> None:
    """
    Slide-ready plot:
      - 20–50 Monte Carlo trajectories (thin grey)
      - Airworthiness threshold (red horizontal)
      - Optimized shop start (vertical)
      - Reset effect visible in trajectories
    """
    T = int(horizon_months)
    S = int(n_scenarios)
    n_paths = int(max(1, min(n_paths, S)))

    rng = np.random.default_rng(seed)
    scenarios = rng.choice(np.arange(S), size=n_paths, replace=False)

    x = np.arange(1, T + 2)  # months 1..T+1

    fig, ax = plt.subplots(figsize=(12, 6), dpi=220)

    # Monte Carlo paths
    for s in scenarios:
        path = _health_path_under_single_shop(
            h0=h0,
            dh=dh,
            engine_id=engine_id,
            horizon_months=T,
            scenario=int(s),
            shop_start_month=int(shop_start_month),
            shop_duration_months=int(shop_duration_months),
        )
        ax.plot(x, path, linewidth=1.0, alpha=0.35)  # default color, thin

    # Threshold
    ax.axhline(float(h_min), linewidth=2.0, color="red", label=f"Airworthiness threshold (h ≥ {h_min})")

    # Shop start marker
    if int(shop_start_month) > 0:
        ax.axvline(int(shop_start_month), linewidth=2.0, linestyle="--", label=f"Optimized shop start (month {shop_start_month})")

    # Cosmetics for a slide
    ax.set_title(f"Engine Health Trajectories (Monte Carlo) | {engine_id}")
    ax.set_xlabel("Month")
    ax.set_ylabel("Health index")
    ax.set_xlim(1, T + 1)
    ax.set_ylim(0.0, 1.02)
    ax.set_xticks(np.arange(1, T + 2))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)