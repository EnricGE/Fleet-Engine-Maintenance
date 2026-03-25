from __future__ import annotations

from typing import Dict, Tuple, Optional

from ortools.sat.python import cp_model

from fleet_engine_planning.domain.engine import Fleet
from fleet_engine_planning.preprocessing.schema import CostParams
from fleet_engine_planning.solvers import ScheduleResult


def solve_cpsat_schedule_with_rentals(
    fleet: Fleet,
    horizon_months: int,
    shop_capacity: list[int],
    shop_duration_months: int,
    max_rentals_per_month: int,
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
      - m>=1: engine in shop during month m (unavailable), returns at m+D reset

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
    D = int(shop_duration_months)

    for t in range(1, T + 1):
        starts_that_cover_t = []
        for m in range(1, T + 1):
            if m <= t <= m + D - 1:
                starts_that_cover_t.append(m)

        model.Add(
            sum(y[(i, m)] for i in engine_ids for m in starts_that_cover_t)
            <= int(shop_capacity[t - 1])
        )

    # rentals and downtime per (t,s)
    r: Dict[Tuple[int, int], cp_model.IntVar] = {}
    d: Dict[Tuple[int, int], cp_model.IntVar] = {}

    max_rentals = int(max_rentals_per_month)

    for t in range(1, T + 1):
        for s in range(S):
            r[(t, s)] = model.NewIntVar(0, max_rentals, f"r_{t}_{s}")
            d[(t, s)] = model.NewIntVar(0, len(engine_ids), f"d_{t}_{s}")

            # operable engines count is a linear expression of y
            oper_expr = []
            for i in engine_ids:
                for m in range(0, T + 1):
                    a = int(operable[(i, t, s, m)])
                    if a == 1:
                        oper_expr.append(y[(i, m)])  # coefficient 1
                    # if a==0, skip

            # Coverage constraint
            model.Add(sum(oper_expr) + r[(t, s)] + d[(t, s)] >= int(n_required))

    # Objective: shop + expected rentals + expected downtime
    # Use integer scaling to keep CP-SAT happy with floats.
    SCALE = 100 
    S_INV = max(1, S)
    obj_terms = []

    # shop costs
    for i in engine_ids:
        for m in range(1, T + 1):
            c = expected_shop_cost[(i, m)]
            obj_terms.append(int(round(c * SCALE)) * y[(i, m)])

    # rentals/downtime average over scenarios
    r_cost = max(1, int(round(costs.rental_cost * SCALE / S_INV)))
    d_cost = max(1, int(round(costs.downtime_cost * SCALE / S_INV)))

    for t in range(1, T + 1):
        for s in range(S):
            obj_terms.append(r_cost * r[(t, s)])
            obj_terms.append(d_cost * d[(t, s)])

    # Terminal penalty: penalize P(inoperable at start of month T+1)
    kappa = float(costs.terminal_inop_cost)
    if kappa > 0.0:
        t_term = T + 1  # now exists in operable tensor
        for i in engine_ids:
            for m in range(0, T + 1):
                p_inop = sum(1 - int(operable[(i, t_term, s, m)]) for s in range(S)) / S
                if p_inop > 0.0:
                    obj_terms.append(int(round(kappa * p_inop * SCALE)) * y[(i, m)])

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
    solver_status = "optimal" if status == cp_model.OPTIMAL else "feasible"

    return ScheduleResult(schedule=schedule, objective=objective, rentals=rentals, downtime=downtime, solver_status=solver_status)