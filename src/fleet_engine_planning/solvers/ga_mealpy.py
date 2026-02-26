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
    rentals: Dict[Tuple[int, int], int]
    downtime: Dict[Tuple[int, int], int]


def _capacity_usage(months: np.ndarray, T: int, D: int) -> np.ndarray:
    """
    usage[t] = number of engines occupying shop capacity in month t (1..T)
    where a start at month m occupies months m..m+D-1 (clipped to horizon).
    """
    usage = np.zeros(T + 1, dtype=int)  # index 0 unused
    for m in months:
        m = int(m)
        if m <= 0:
            continue
        for t in range(m, min(T, m + D - 1) + 1):
            usage[t] += 1
    return usage


def _repair_capacity_with_duration(
    months: np.ndarray,
    cap: List[int],
    T: int,
    D: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Repair a schedule vector months[i] in {0..T} so that for each month t:
      usage[t] <= cap[t-1]
    where usage accounts for shop duration D.

    Strategy:
      While any month overloaded:
        - pick an engine that contributes to an overloaded month
        - move its shop start to a feasible month (nearest with slack), otherwise set to 0
    """
    months = months.astype(int).copy()
    cap_arr = np.array([0] + [int(x) for x in cap], dtype=int)  # cap_arr[t] for t=1..T

    for _ in range(2000):  # safety to avoid infinite loops
        usage = _capacity_usage(months, T, D)
        overload_months = np.where(usage[1:] > cap_arr[1:])[0] + 1  # months 1..T

        if len(overload_months) == 0:
            return months

        t_over = int(rng.choice(overload_months))

        # Engines that occupy capacity in t_over
        idx = []
        for i, m in enumerate(months):
            m = int(m)
            if m > 0 and (m <= t_over <= min(T, m + D - 1)):
                idx.append(i)

        if not idx:
            # shouldn't happen, but break safely
            return months

        i = int(rng.choice(idx))
        m_old = int(months[i])

        # try to relocate start month
        placed = False
        candidates = list(range(1, T + 1))
        # sort by distance to current start
        candidates.sort(key=lambda mm: abs(mm - m_old))

        for m_new in candidates:
            if m_new == m_old:
                continue

            # test if moving i to m_new keeps capacity feasible
            test = months.copy()
            test[i] = m_new
            usage_test = _capacity_usage(test, T, D)
            if np.all(usage_test[1:] <= cap_arr[1:]):
                months[i] = m_new
                placed = True
                break

        if not placed:
            # fallback: remove the shop visit
            months[i] = 0

    # If repair didn't converge, return best effort
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
    max_rentals_per_month: int,
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

    rentals = {}
    downtime = {}

    max_r = int(max_rentals_per_month)

    for t in range(1, T + 1):
        for s in range(S):
            oper_count = 0
            for i, m in zip(engine_ids, months):
                oper_count += int(operable[(i, t, s, int(m))])

            shortage = max(0, int(n_required) - int(oper_count))

            rent = min(shortage, max_r)
            down = shortage - rent

            rentals[(t, s)] = rent
            downtime[(t, s)] = down

    # Compute expected slack cost
    slack_cost = 0.0
    for t in range(1, T + 1):
        avg_r = sum(rentals[(t, s)] for s in range(S)) / S
        avg_d = sum(downtime[(t, s)] for s in range(S)) / S
        slack_cost += avg_r * costs.rental_cost + avg_d * costs.downtime_cost

    total = shop_cost + slack_cost

    return total, rentals, downtime

def solve_ga_mealpy(
    fleet: Fleet,
    horizon_months: int,
    shop_capacity: List[int],
    shop_duration_months: int,
    max_rentals_per_month: int,
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

    D = int(shop_duration_months)

    # --- Define MEALPY Problem ---
    if _MEALPY_V3:
        bounds = IntegerVar(lb=[0] * n, ub=[T] * n)

        class FleetScheduleProblem(Problem):
            def __init__(self):
                super().__init__(bounds=bounds, minmax="min")

            def generate_position(self):
                x = rng.integers(low=0, high=T + 1, size=n, dtype=int)
                x = _repair_capacity_with_duration(x, shop_capacity, T, D, rng)
                return x

            def amend_position(self, solution):
                # MEALPY may pass numpy float; force int and repair
                x = np.rint(solution).astype(int)
                x = np.clip(x, 0, T)
                x = _repair_capacity_with_duration(x, shop_capacity, T, D, rng)
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
                    max_rentals_per_month=max_rentals_per_month,
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
                x = _repair_capacity_with_duration(x, shop_capacity, T, D, rng)
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
    model = GA.BaseGA(epoch=epoch, pop_size=pop_size, pc=pc, pm=pm)
    model.solve(problem)

    best = model.g_best.solution  # encoded solution
    best_int = problem.amend_position(best)

    obj, rentals, downtime = _evaluate_schedule(
        engine_ids=engine_ids,
        months=best_int,
        T=T,
        S=S,
        n_required=n_required,
        costs=costs,
        operable=operable,
        expected_shop_cost=expected_shop_cost,
        max_rentals_per_month=max_rentals_per_month,
    )

    schedule = {eid: int(m) for eid, m in zip(engine_ids, best_int)}
    return ScheduleResult(
        schedule=schedule,
        objective=float(obj),
        rentals=rentals,
        downtime=downtime,
)