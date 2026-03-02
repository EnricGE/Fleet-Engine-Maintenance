from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


# -----------------------------
# Styling / utilities
# -----------------------------

@dataclass(frozen=True)
class PlotStyle:
    dpi: int = 200
    figsize: Tuple[float, float] = (11, 5)
    grid: bool = True
    grid_alpha: float = 0.25
    title_size: int = 14
    label_size: int = 11
    tick_size: int = 10


def _apply_style(style: PlotStyle) -> None:
    plt.rcParams.update(
        {
            "figure.dpi": style.dpi,
            "axes.titlesize": style.title_size,
            "axes.labelsize": style.label_size,
            "xtick.labelsize": style.tick_size,
            "ytick.labelsize": style.tick_size,
            "legend.fontsize": style.tick_size,
        }
    )


def _save(fig: plt.Figure, out_path: Path, style: PlotStyle) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=style.dpi)
    plt.close(fig)


def _money_formatter(x, _pos) -> str:
    # 1_000 -> 1.0k, 1_000_000 -> 1.0M
    x = float(x)
    ax = abs(x)
    if ax >= 1_000_000:
        return f"{x/1_000_000:.1f}M"
    if ax >= 1_000:
        return f"{x/1_000:.1f}k"
    return f"{x:.0f}"


# -----------------------------
# Core computations
# -----------------------------

def _compute_shop_occupancy(
    schedule: Dict[str, int],
    horizon_months: int,
    shop_duration_months: int,
) -> np.ndarray:
    T = int(horizon_months)
    D = int(shop_duration_months)
    occ = np.zeros(T + 1, dtype=int)  # index 0 unused
    for m in schedule.values():
        m = int(m)
        if m <= 0:
            continue
        for t in range(m, min(T, m + D - 1) + 1):
            occ[t] += 1
    return occ


def _compute_shop_starts(schedule: Dict[str, int], horizon_months: int) -> np.ndarray:
    T = int(horizon_months)
    starts = np.zeros(T + 1, dtype=int)
    for m in schedule.values():
        m = int(m)
        if 0 <= m <= T:
            starts[m] += 1
    return starts


def _compute_rentals_downtime_stats(
    rentals: Dict[Tuple[int, int], int],
    downtime: Dict[Tuple[int, int], int],
    horizon_months: int,
    n_scenarios: int,
) -> Dict[str, np.ndarray]:
    T = int(horizon_months)
    S = int(n_scenarios)

    avg_r = np.zeros(T + 1, dtype=float)
    avg_d = np.zeros(T + 1, dtype=float)
    worst_d = np.zeros(T + 1, dtype=float)

    for t in range(1, T + 1):
        r_vals = np.array([rentals[(t, s)] for s in range(S)], dtype=float)
        d_vals = np.array([downtime[(t, s)] for s in range(S)], dtype=float)
        avg_r[t] = r_vals.mean()
        avg_d[t] = d_vals.mean()
        worst_d[t] = d_vals.max()

    return {"avg_r": avg_r, "avg_d": avg_d, "worst_d": worst_d}


# -----------------------------
# Plots
# -----------------------------

def plot_shop_occupancy_vs_capacity(
    schedule: Dict[str, int],
    shop_capacity: List[int],
    horizon_months: int,
    shop_duration_months: int,
    out_path: Path,
    style: PlotStyle = PlotStyle(),
) -> None:
    _apply_style(style)
    T = int(horizon_months)
    cap = np.array([0] + [int(x) for x in shop_capacity], dtype=int)
    occ = _compute_shop_occupancy(schedule, T, shop_duration_months)

    x = np.arange(1, T + 1)

    fig, ax = plt.subplots(figsize=style.figsize)
    # Capacity as a band (more readable than a line alone)
    ax.fill_between(x, 0, cap[1:], alpha=0.15, label="Capacity")
    ax.plot(x, cap[1:], marker="o", linewidth=1.5)

    # Occupancy line
    ax.plot(x, occ[1:], marker="o", linewidth=2.0, label="Shop occupancy")

    ax.set_xlabel("Month")
    ax.set_ylabel("Engines in shop")
    ax.set_title("Shop occupancy vs capacity")
    ax.set_xticks(x)

    if style.grid:
        ax.grid(True, alpha=style.grid_alpha)

    ax.legend(loc="upper left")
    _save(fig, out_path, style)


def plot_capacity_utilization_percent(
    schedule: Dict[str, int],
    shop_capacity: List[int],
    horizon_months: int,
    shop_duration_months: int,
    out_path: Path,
    style: PlotStyle = PlotStyle(),
) -> None:
    _apply_style(style)
    T = int(horizon_months)
    cap = np.array([0] + [int(x) for x in shop_capacity], dtype=float)
    occ = _compute_shop_occupancy(schedule, T, shop_duration_months).astype(float)

    util = np.zeros(T + 1, dtype=float)
    for t in range(1, T + 1):
        util[t] = 0.0 if cap[t] <= 0 else 100.0 * occ[t] / cap[t]

    x = np.arange(1, T + 1)

    fig, ax = plt.subplots(figsize=style.figsize)
    ax.plot(x, util[1:], marker="o", linewidth=2.0, label="Utilization")
    ax.axhline(100.0, linestyle="--", linewidth=1.5, label="100%")
    ax.set_xlabel("Month")
    ax.set_ylabel("Utilization (%)")
    ax.set_title("Shop capacity utilization (%)")
    ax.set_xticks(x)
    ax.set_ylim(0, max(110.0, float(np.max(util[1:]) + 10.0)))

    if style.grid:
        ax.grid(True, alpha=style.grid_alpha)

    ax.legend(loc="upper left")
    _save(fig, out_path, style)


def plot_shop_starts_histogram(
    schedule: Dict[str, int],
    horizon_months: int,
    out_path: Path,
    style: PlotStyle = PlotStyle(figsize=(11, 4)),
) -> None:
    _apply_style(style)
    T = int(horizon_months)
    starts = _compute_shop_starts(schedule, T)

    x = np.arange(0, T + 1)

    fig, ax = plt.subplots(figsize=style.figsize)
    ax.bar(x, starts)
    ax.set_xlabel("Shop start month (0 = none)")
    ax.set_ylabel("Number of engines")
    ax.set_title("Shop start month distribution")
    ax.set_xticks(x)

    if style.grid:
        ax.grid(True, axis="y", alpha=style.grid_alpha)

    _save(fig, out_path, style)


def plot_schedule_heatmap(
    schedule: Dict[str, int],
    horizon_months: int,
    shop_duration_months: int,
    out_path: Path,
    style: PlotStyle = PlotStyle(figsize=(12, 6)),
    max_engines_to_show: Optional[int] = 80,
) -> None:
    """
    Heatmap view: engines (rows) x months (cols).
    Cell = 1 if engine is in shop in that month, else 0.
    Very readable for duration>1.

    If fleet is huge, we cap number of engines shown to keep the plot legible.
    """
    _apply_style(style)
    T = int(horizon_months)
    D = int(shop_duration_months)

    engine_ids = sorted(schedule.keys())
    if max_engines_to_show is not None and len(engine_ids) > max_engines_to_show:
        engine_ids = engine_ids[:max_engines_to_show]

    n = len(engine_ids)
    mat = np.zeros((n, T), dtype=int)

    for r, eid in enumerate(engine_ids):
        m = int(schedule[eid])
        if m <= 0:
            continue
        for t in range(m, min(T, m + D - 1) + 1):
            mat[r, t - 1] = 1

    fig, ax = plt.subplots(figsize=style.figsize)
    im = ax.imshow(mat, aspect="auto", interpolation="nearest")
    ax.set_title("Shop schedule heatmap (in-shop blocks)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Engine (sorted subset)")
    ax.set_xticks(np.arange(T))
    ax.set_xticklabels([str(t) for t in range(1, T + 1)])

    if style.grid:
        ax.grid(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("In shop (1=yes, 0=no)")

    _save(fig, out_path, style)


def plot_rentals_downtime_profile(
    rentals: Dict[Tuple[int, int], int],
    downtime: Dict[Tuple[int, int], int],
    horizon_months: int,
    n_scenarios: int,
    out_path: Path,
    style: PlotStyle = PlotStyle(),
) -> None:
    _apply_style(style)
    T = int(horizon_months)
    stats = _compute_rentals_downtime_stats(rentals, downtime, T, n_scenarios)
    x = np.arange(1, T + 1)

    fig, ax = plt.subplots(figsize=style.figsize)
    ax.plot(x, stats["avg_r"][1:], marker="o", linewidth=2.0, label="Avg rentals")
    ax.plot(x, stats["avg_d"][1:], marker="o", linewidth=2.0, label="Avg downtime")
    ax.plot(x, stats["worst_d"][1:], linestyle="--", linewidth=2.0, label="Worst-case downtime")

    ax.set_xlabel("Month")
    ax.set_ylabel("Engines")
    ax.set_title("Rentals and downtime (avg and worst-case)")
    ax.set_xticks(x)

    if style.grid:
        ax.grid(True, alpha=style.grid_alpha)

    ax.legend(loc="upper left")
    _save(fig, out_path, style)


def plot_downtime_probability(
    downtime: Dict[Tuple[int, int], int],
    horizon_months: int,
    n_scenarios: int,
    out_path: Path,
    style: PlotStyle = PlotStyle(),
) -> None:
    _apply_style(style)
    T = int(horizon_months)
    S = int(n_scenarios)

    prob = np.zeros(T + 1, dtype=float)
    for t in range(1, T + 1):
        prob[t] = sum(1 for s in range(S) if int(downtime[(t, s)]) > 0) / S

    x = np.arange(1, T + 1)

    fig, ax = plt.subplots(figsize=style.figsize)
    ax.plot(x, prob[1:], marker="o", linewidth=2.0)
    ax.set_xlabel("Month")
    ax.set_ylabel("Probability")
    ax.set_title("Probability of downtime (P(downtime > 0))")
    ax.set_xticks(x)
    ax.set_ylim(-0.05, 1.05)

    if style.grid:
        ax.grid(True, alpha=style.grid_alpha)

    _save(fig, out_path, style)


def plot_rental_cap_hit_rate(
    rentals: Dict[Tuple[int, int], int],
    horizon_months: int,
    n_scenarios: int,
    max_rentals_per_month: int,
    out_path: Path,
    style: PlotStyle = PlotStyle(),
) -> None:
    _apply_style(style)
    T = int(horizon_months)
    S = int(n_scenarios)
    cap = int(max_rentals_per_month)

    prob = np.zeros(T + 1, dtype=float)
    for t in range(1, T + 1):
        prob[t] = sum(1 for s in range(S) if int(rentals[(t, s)]) == cap) / S

    x = np.arange(1, T + 1)

    fig, ax = plt.subplots(figsize=style.figsize)
    ax.plot(x, prob[1:], marker="o", linewidth=2.0)
    ax.set_xlabel("Month")
    ax.set_ylabel("Probability")
    ax.set_title("Rental cap hit rate (P(rentals == cap))")
    ax.set_xticks(x)
    ax.set_ylim(-0.05, 1.05)

    if style.grid:
        ax.grid(True, alpha=style.grid_alpha)

    _save(fig, out_path, style)


def plot_expected_cost_per_month(
    schedule: Dict[str, int],
    expected_shop_cost: Dict[Tuple[str, int], float],
    rentals: Dict[Tuple[int, int], int],
    downtime: Dict[Tuple[int, int], int],
    horizon_months: int,
    n_scenarios: int,
    rental_cost: float,
    downtime_cost: float,
    out_path: Path,
    style: PlotStyle = PlotStyle(figsize=(12, 5)),
) -> None:
    """
    Stacked monthly expected cost bars + cumulative expected cost line (2nd axis).
    Maintenance cost attributed to shop start month.
    """
    _apply_style(style)
    T = int(horizon_months)
    S = int(n_scenarios)

    maint = np.zeros(T + 1, dtype=float)
    rent = np.zeros(T + 1, dtype=float)
    down = np.zeros(T + 1, dtype=float)

    for engine_id, m in schedule.items():
        m = int(m)
        if m >= 1:
            maint[m] += float(expected_shop_cost[(engine_id, m)])

    for t in range(1, T + 1):
        avg_r = sum(rentals[(t, s)] for s in range(S)) / S
        avg_d = sum(downtime[(t, s)] for s in range(S)) / S
        rent[t] = avg_r * float(rental_cost)
        down[t] = avg_d * float(downtime_cost)

    total = maint + rent + down
    cum_total = np.cumsum(total)

    x = np.arange(1, T + 1)

    fig, ax1 = plt.subplots(figsize=style.figsize)
    ax1.bar(x, maint[1:], label="Maintenance")
    ax1.bar(x, rent[1:], bottom=maint[1:], label="Rentals")
    ax1.bar(x, down[1:], bottom=maint[1:] + rent[1:], label="Downtime")

    ax1.set_xlabel("Month")
    ax1.set_ylabel("Expected cost per month")
    ax1.set_title("Expected cost per month + cumulative expected cost")
    ax1.set_xticks(x)
    ax1.yaxis.set_major_formatter(FuncFormatter(_money_formatter))

    if style.grid:
        ax1.grid(True, axis="y", alpha=style.grid_alpha)

    ax2 = ax1.twinx()
    ax2.plot(x, cum_total[1:], marker="o", linewidth=2.0, label="Cumulative")
    ax2.set_ylabel("Cumulative expected cost")
    ax2.yaxis.set_major_formatter(FuncFormatter(_money_formatter))

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")

    _save(fig, out_path, style)


def plot_risk_pareto_summary(
    downtime: Dict[Tuple[int, int], int],
    horizon_months: int,
    n_scenarios: int,
    out_path: Path,
    style: PlotStyle = PlotStyle(figsize=(9, 6)),
) -> None:
    """
    Pareto-style monthly risk summary:
      x = P(downtime > 0)
      y = Avg downtime
      bubble size = Worst-case downtime
      labels = month index
    """
    _apply_style(style)
    T = int(horizon_months)
    S = int(n_scenarios)

    p = np.zeros(T + 1, dtype=float)
    avg = np.zeros(T + 1, dtype=float)
    worst = np.zeros(T + 1, dtype=float)

    for t in range(1, T + 1):
        vals = np.array([int(downtime[(t, s)]) for s in range(S)], dtype=float)
        p[t] = float(np.mean(vals > 0))
        avg[t] = float(np.mean(vals))
        worst[t] = float(np.max(vals))

    x = p[1:]
    y = avg[1:]
    w = worst[1:]

    # Bubble size scaling (avoid tiny/huge extremes)
    sizes = 80 + 220 * (w / max(1.0, float(np.max(w))))  # 80..300-ish

    fig, ax = plt.subplots(figsize=style.figsize)
    ax.scatter(x, y, s=sizes)

    for t in range(1, T + 1):
        ax.annotate(str(t), (p[t], avg[t]), textcoords="offset points", xytext=(6, 4))

    ax.set_xlabel("P(downtime > 0)")
    ax.set_ylabel("Avg downtime")
    ax.set_title("Risk summary by month (bubble = worst-case downtime)")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(bottom=0.0)

    if style.grid:
        ax.grid(True, alpha=style.grid_alpha)

    _save(fig, out_path, style)


# -----------------------------
# Public entrypoint
# -----------------------------

def save_standard_plots(
    schedule: Dict[str, int],
    rentals: Dict[Tuple[int, int], int],
    downtime: Dict[Tuple[int, int], int],
    horizon_months: int,
    n_scenarios: int,
    shop_capacity: List[int],
    shop_duration_months: int,
    max_rentals_per_month: int,
    expected_shop_cost: Dict[Tuple[str, int], float],
    rental_cost: float,
    downtime_cost: float,
    out_dir: Path,
    prefix: str = "cpsat",
) -> None:
    """
    Solver-agnostic. Works for CP-SAT now and GA later if it returns
    schedule/rentals/downtime dicts with the same keys.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_shop_occupancy_vs_capacity(
        schedule=schedule,
        shop_capacity=shop_capacity,
        horizon_months=horizon_months,
        shop_duration_months=shop_duration_months,
        out_path=out_dir / f"{prefix}_shop_occupancy_vs_capacity.png",
    )

    plot_capacity_utilization_percent(
        schedule=schedule,
        shop_capacity=shop_capacity,
        horizon_months=horizon_months,
        shop_duration_months=shop_duration_months,
        out_path=out_dir / f"{prefix}_capacity_utilization_pct.png",
    )

    plot_shop_starts_histogram(
        schedule=schedule,
        horizon_months=horizon_months,
        out_path=out_dir / f"{prefix}_shop_starts_hist.png",
    )

    plot_schedule_heatmap(
        schedule=schedule,
        horizon_months=horizon_months,
        shop_duration_months=shop_duration_months,
        out_path=out_dir / f"{prefix}_schedule_heatmap.png",
    )

    plot_rentals_downtime_profile(
        rentals=rentals,
        downtime=downtime,
        horizon_months=horizon_months,
        n_scenarios=n_scenarios,
        out_path=out_dir / f"{prefix}_rentals_downtime.png",
    )

    plot_downtime_probability(
        downtime=downtime,
        horizon_months=horizon_months,
        n_scenarios=n_scenarios,
        out_path=out_dir / f"{prefix}_downtime_probability.png",
    )

    plot_rental_cap_hit_rate(
        rentals=rentals,
        horizon_months=horizon_months,
        n_scenarios=n_scenarios,
        max_rentals_per_month=max_rentals_per_month,
        out_path=out_dir / f"{prefix}_rental_cap_hit_rate.png",
    )

    plot_expected_cost_per_month(
        schedule=schedule,
        expected_shop_cost=expected_shop_cost,
        rentals=rentals,
        downtime=downtime,
        horizon_months=horizon_months,
        n_scenarios=n_scenarios,
        rental_cost=rental_cost,
        downtime_cost=downtime_cost,
        out_path=out_dir / f"{prefix}_expected_cost_per_month.png",
    )

    plot_risk_pareto_summary(
        downtime=downtime,
        horizon_months=horizon_months,
        n_scenarios=n_scenarios,
        out_path=out_dir / f"{prefix}_risk_pareto_summary.png",
    )