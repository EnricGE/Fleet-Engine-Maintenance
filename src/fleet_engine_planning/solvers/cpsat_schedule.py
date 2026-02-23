from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

from ortools.sat.python import cp_model

from fleet_engine_planning.fleet.engine import Fleet
from fleet_engine_planning.preprocessing.schema import CostParams


@dataclass(frozen=True)
class ScheduleResult:
    schedule: Dict[str, int]  # engine_id -> shop_month (0..T)
    objective: float
    rentals: Dict[Tuple[int, int], int]   # (month, scenario) -> rentals
    downtime: Dict[Tuple[int, int], int]  # (month, scenario) -> downtime


def solve_cpsat_schedule_with_rentals(
    fleet: Fleet,
    horizon_months: int,
    shop_capacity: list[int],
    n_required: int,
    n_scenarios: int,
    costs: CostParams,
    operable: Dict[Tuple[str, int, int, int], int],
    expected_shop_cost: Dict[Tuple[str, int], float],
    time_limit_s: float = 10.0,
) -> Optional[ScheduleResult]:
    """
    Decision: choose one shop month m in {0..T} for each engine i.
      - m=0 means no shop in horizon
      - m>=1: engine in shop during month m (unavailable), returns at m+1 reset

    Constraints:
      - capacity per month
      - coverage per (month, scenario): operable engines + rentals + downtime >= n_required

    Objective:
      sum expected shop costs + expected rentals/downtime costs
    """
    T = int(horizon_months)
    S = int(n_scenarios)
    engine_ids = [e.engine_id for e in fleet.engines]

    if len(shop_capacity) != T:
        raise ValueError("shop_capacity length must equal horizon_months")

    model = cp_model.CpModel()

    # y[i,m] = 1 if engine i shops at month m (m=0..T)
    y: Dict[Tuple[str, int], cp_model.IntVar] = {}
    for i in engine_ids:
        for m in range(0, T + 1):
            y[(i, m)] = model.NewBoolVar(f"y_{i}_{m}")

    # each engine chooses exactly one option
    for i in engine_ids:
        model.Add(sum(y[(i, m)] for m in range(0, T + 1)) == 1)

    # capacity constraints (ignore m=0)
    for m in range(1, T + 1):
        model.Add(sum(y[(i, m)] for i in engine_ids) <= int(shop_capacity[m - 1]))

    # rentals and downtime per (t,s)
    r: Dict[Tuple[int, int], cp_model.IntVar] = {}
    d: Dict[Tuple[int, int], cp_model.IntVar] = {}
    for t in range(1, T + 1):
        for s in range(S):
            r[(t, s)] = model.NewIntVar(0, len(engine_ids), f"r_{t}_{s}")
            d[(t, s)] = model.NewIntVar(0, len(engine_ids), f"d_{t}_{s}")

            # operable engines count is a linear expression of y
            oper_expr = []
            for i in engine_ids:
                for m in range(0, T + 1):
                    a = int(operable[(i, t, s, m)])
                    if a == 1:
                        oper_expr.append(y[(i, m)])  # coefficient 1
                    # if a==0, skip

            model.Add(sum(oper_expr) + r[(t, s)] + d[(t, s)] >= int(n_required))

    # Objective: shop + expected rentals + expected downtime
    # Use integer scaling to keep CP-SAT happy with floats.
    SCALE = 100  # cents-ish; increase if needed
    obj_terms = []

    # shop costs
    for i in engine_ids:
        for m in range(1, T + 1):
            c = expected_shop_cost[(i, m)]
            obj_terms.append(int(round(c * SCALE)) * y[(i, m)])

    # rentals/downtime average over scenarios
    # expected cost = (1/S) * sum_{t,s} cost * var
    # implement as sum_{t,s} cost * var, then divide at the end for reporting
    for t in range(1, T + 1):
        for s in range(S):
            obj_terms.append(int(round(costs.rental_cost * SCALE)) * r[(t, s)])
            obj_terms.append(int(round(costs.downtime_cost * SCALE)) * d[(t, s)])

    model.Minimize(sum(obj_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None

    schedule = {}
    for i in engine_ids:
        chosen = 0
        for m in range(0, T + 1):
            if solver.Value(y[(i, m)]) == 1:
                chosen = m
                break
        schedule[i] = chosen

    rentals = {(t, s): int(solver.Value(r[(t, s)])) for t in range(1, T + 1) for s in range(S)}
    downtime = {(t, s): int(solver.Value(d[(t, s)])) for t in range(1, T + 1) for s in range(S)}

    objective = solver.ObjectiveValue() / SCALE
    return ScheduleResult(schedule=schedule, objective=objective, rentals=rentals, downtime=downtime)