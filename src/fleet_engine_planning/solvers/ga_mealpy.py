from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np

from fleet_engine_planning.fleet.engine import Fleet
from fleet_engine_planning.preprocessing.schema import CostParams

# MEALPY imports (v3.x)
try:
    from mealpy import GA, IntegerVar, Problem
    _MEALPY_V3 = True
except Exception:
    # Fallback for older mealpy versions (continuous vars only)
    from mealpy import GA, FloatVar, Problem  # type: ignore
    IntegerVar = None  # type: ignore
    _MEALPY_V3 = False


@dataclass(frozen=True)
class ScheduleResult:
    schedule: Dict[str, int]                # engine_id -> shop_month (0..T)
    objective: float
    rentals_avg: Dict[int, float]           # month -> avg rentals over scenarios
    downtime_avg: Dict[int, float]          # month -> avg downtime over scenarios


def _repair_capacity(months: np.ndarray, cap: List[int], T: int, rng: np.random.Generator) -> np.ndarray:
    """
    Repair a schedule vector months[i] in {0..T} so that
    for each month m=1..T: count(months==m) <= cap[m-1].

    Simple greedy repair:
      - For an overloaded month, move random engines to the nearest month with free cap,
        otherwise set to 0 (no shop).
    """
    months = months.astype(int).copy()

    for m in range(1, T + 1):
        limit = int(cap[m - 1])
        idx = np.where(months == m)[0]
        if len(idx) <= limit:
            continue

        excess = len(idx) - limit
        rng.shuffle(idx)
        move_idx = idx[:excess]

        for i in move_idx:
            placed = False

            # try nearest months first
            for delta in range(1, T + 1):
                for m2 in (m - delta, m + delta):
                    if 1 <= m2 <= T:
                        if int(np.sum(months == m2)) < int(cap[m2 - 1]):
                            months[i] = m2
                            placed = True
                            break
                if placed:
                    break

            if not placed:
                months[i] = 0  # give up: no shop in horizon

    return months


def _evaluate_schedule(
    engine_ids: List[str],
    months: np.ndarray,
    T: int,
    S: int,
    n_required: int,
    costs: CostParams,
    operable: Dict[Tuple[str, int, int, int], int],
    expected_shop_cost: Dict[Tuple[str, int], float],
) -> Tuple[float, Dict[int, float], Dict[int, float]]:
    """
    Compute objective + avg rentals/downtime per month.

    - Shop cost: sum expected_shop_cost[i,m] for m>=1
    - For each month t and scenario s:
        oper_count = sum_i operable[(i,t,s,m_i)]
        shortage = max(0, n_required - oper_count)
      cover shortage by cheaper slack:
        if rental_cost <= downtime_cost: rentals=shortage else downtime=shortage
    """
    months = months.astype(int)
    shop_cost = 0.0
    for i, m in zip(engine_ids, months):
        if m >= 1:
            shop_cost += float(expected_shop_cost[(i, int(m))])

    rentals_sum = {t: 0.0 for t in range(1, T + 1)}
    downtime_sum = {t: 0.0 for t in range(1, T + 1)}

    use_rentals = costs.rental_cost <= costs.downtime_cost

    for t in range(1, T + 1):
        for s in range(S):
            oper_count = 0
            for i, m in zip(engine_ids, months):
                oper_count += int(operable[(i, t, s, int(m))])

            shortage = max(0, int(n_required) - int(oper_count))
            if shortage > 0:
                if use_rentals:
                    rentals_sum[t] += shortage
                else:
                    downtime_sum[t] += shortage

    # expected (average over scenarios)
    rentals_avg = {t: rentals_sum[t] / S for t in range(1, T + 1)}
    downtime_avg = {t: downtime_sum[t] / S for t in range(1, T + 1)}

    # objective: expected costs (avg over scenarios for rentals/downtime)
    slack_cost = 0.0
    for t in range(1, T + 1):
        slack_cost += rentals_avg[t] * costs.rental_cost + downtime_avg[t] * costs.downtime_cost

    total = shop_cost + slack_cost
    return total, rentals_avg, downtime_avg


def solve_ga_mealpy(
    fleet: Fleet,
    horizon_months: int,
    shop_capacity: List[int],
    n_required: int,
    n_scenarios: int,
    costs: CostParams,
    operable: Dict[Tuple[str, int, int, int], int],
    expected_shop_cost: Dict[Tuple[str, int], float],
    seed: int = 42,
    epoch: int = 300,
    pop_size: int = 60,
    pc: float = 0.9,
    pm: float = 0.2,
) -> Optional[ScheduleResult]:
    """
    MEALPY GA solver for the same scheduling instance as CP-SAT.

    Genome:
      months[i] in {0..T} where:
        0 = no shop
        m>=1 = shop in month m

    Capacity enforced by repair.

    Returns ScheduleResult with objective and avg rentals/downtime per month.
    """
    T = int(horizon_months)
    S = int(n_scenarios)
    if len(shop_capacity) != T:
        raise ValueError("shop_capacity length must equal horizon_months")

    engine_ids = [e.engine_id for e in fleet.engines]
    n = len(engine_ids)
    rng = np.random.default_rng(seed)

    # --- Define MEALPY Problem ---
    if _MEALPY_V3:
        bounds = IntegerVar(lb=[0] * n, ub=[T] * n)

        class FleetScheduleProblem(Problem):
            def __init__(self):
                super().__init__(bounds=bounds, minmax="min")

            def generate_position(self):
                x = rng.integers(low=0, high=T + 1, size=n, dtype=int)
                x = _repair_capacity(x, shop_capacity, T, rng)
                return x

            def amend_position(self, solution):
                # MEALPY may pass numpy float; force int and repair
                x = np.rint(solution).astype(int)
                x = np.clip(x, 0, T)
                x = _repair_capacity(x, shop_capacity, T, rng)
                return x

            def obj_func(self, solution):
                x = self.amend_position(solution)
                total, _, _ = _evaluate_schedule(
                    engine_ids=engine_ids,
                    months=x,
                    T=T,
                    S=S,
                    n_required=n_required,
                    costs=costs,
                    operable=operable,
                    expected_shop_cost=expected_shop_cost,
                )
                return total

    else:
        # Older MEALPY fallback: FloatVar + round-to-int inside obj/repair
        bounds = FloatVar(lb=[0.0] * n, ub=[float(T)] * n)

        class FleetScheduleProblem(Problem):
            def __init__(self):
                super().__init__(bounds=bounds, minmax="min")

            def amend_position(self, solution):
                x = np.rint(solution).astype(int)
                x = np.clip(x, 0, T)
                x = _repair_capacity(x, shop_capacity, T, rng)
                return x

            def obj_func(self, solution):
                x = self.amend_position(solution)
                total, _, _ = _evaluate_schedule(
                    engine_ids=engine_ids,
                    months=x,
                    T=T,
                    S=S,
                    n_required=n_required,
                    costs=costs,
                    operable=operable,
                    expected_shop_cost=expected_shop_cost,
                )
                return total

    problem = FleetScheduleProblem()

    # --- Choose GA variant ---
    # MEALPY offers multiple GA variants; this is a solid default.
    model = GA.BaseGA(epoch=epoch, pop_size=pop_size, pc=pc, pm=pm)
    model.solve(problem)

    best = model.g_best.solution  # encoded solution
    best_int = problem.amend_position(best)

    obj, rentals_avg, downtime_avg = _evaluate_schedule(
        engine_ids=engine_ids,
        months=best_int,
        T=T,
        S=S,
        n_required=n_required,
        costs=costs,
        operable=operable,
        expected_shop_cost=expected_shop_cost,
    )

    schedule = {eid: int(m) for eid, m in zip(engine_ids, best_int)}
    return ScheduleResult(schedule=schedule, objective=float(obj), rentals_avg=rentals_avg, downtime_avg=downtime_avg)