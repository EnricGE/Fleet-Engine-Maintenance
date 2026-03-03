from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Callable

from fleet_engine_planning.fleet.engine import Engine, Fleet
from fleet_engine_planning.preprocessing.schema import Scenario
from fleet_engine_planning.simulation.deterioration import sample_deterioration_deltas
from fleet_engine_planning.optimizer.precompute import build_operability_tensor, build_expected_shop_costs


@dataclass
class RollingState:
    engines: Dict[str, Engine]
    shop_remaining: Dict[str, int]  # months remaining in shop at start of current month


@dataclass
class RollingPlan:
    # committed starts: (engine_id, abs_month) -> 0/1
    shop_starts: Dict[Tuple[str, int], int]
    # first start month view (keeps compatibility with current plots)
    first_start_month: Dict[str, int]
    rentals: Dict[Tuple[int, int], int]   # (abs_month, scenario)
    downtime: Dict[Tuple[int, int], int]  # (abs_month, scenario)


@dataclass(frozen=True)
class WindowSolveResult:
    schedule: Dict[str, int]                 # engine -> local start month
    rentals: Dict[Tuple[int, int], int]      # (local_month, s)
    downtime: Dict[Tuple[int, int], int]     # (local_month, s)

# Solver function signature (solver-agnostic)
# Returns: schedule mapping engine_id -> local shop start month in {0..W_eff}
SolveWindowFn = Callable[
    [Fleet, int, list[int], int, int, Scenario, dict, dict],
    WindowSolveResult,
]

def init_state_from_scenario(scenario: Scenario) -> RollingState:
    engines = {
        e.engine_id: Engine(
            engine_id=e.engine_id,
            age_months=float(e.age_months),
            distance_km=float(e.distance_km),
            health=float(e.health),
        )
        for e in scenario.fleet.engines
    }
    shop_remaining = {eid: 0 for eid in engines}
    return RollingState(engines=engines, shop_remaining=shop_remaining)


def _fleet_from_state(state: RollingState) -> Fleet:
    return Fleet(engines=[state.engines[eid] for eid in sorted(state.engines.keys())])


def _roll_forward_one_month(
    state: RollingState,
    deterioration_delta: Dict[str, float],
    km_per_month: float,
    shop_duration_months: int,
    shop_starts_this_month: Dict[str, int],
) -> None:
    D = int(shop_duration_months)

    for eid, eng in state.engines.items():
        # age always increases
        eng.age_months = float(eng.age_months + 1.0)

        # if already in shop
        if state.shop_remaining.get(eid, 0) > 0:
            state.shop_remaining[eid] -= 1
            if state.shop_remaining[eid] == 0:
                eng.health = 1.0  # reset after completing shop
            continue

        # start shop now?
        if shop_starts_this_month.get(eid, 0) == 1:
            state.shop_remaining[eid] = D
            # no distance / no deterioration during shop month by convention
            continue

        # operate
        dh = float(deterioration_delta.get(eid, 0.0))
        eng.health = float(max(0.0, min(1.0, eng.health - dh)))
        eng.distance_km = float(eng.distance_km + km_per_month)


def run_rolling_horizon(
    scenario: Scenario,
    solve_window: SolveWindowFn,
    H: int = 12,    # horizon_months
    W: int = 6,     # window_length
    K: int = 2,     # commit_length
    n_scenarios: int = 30,
    seed: int = 123,
    realized_scenario_index: int = 0,
) -> RollingPlan:
    """
    Rolling horizon controller.

    At each iteration t0:
      - build subproblem for W_eff months ahead
      - sample deterioration scenarios from current state
      - precompute operability and expected shop cost for that window
      - solve with the provided solve_window() callable
      - commit only the next K months
      - roll forward state using one realized scenario path (index = realized_scenario_index)

    Returns:
      - per-month shop start decisions across 1..H (binary)
      - first_start_month summary for compatibility with existing plots
    """
    if K > W:
        raise ValueError("K must be <= W")
    if realized_scenario_index < 0 or realized_scenario_index >= n_scenarios:
        raise ValueError("realized_scenario_index must be in [0, n_scenarios)")

    state = init_state_from_scenario(scenario)
    engine_ids = sorted(state.engines.keys())

    plan_starts: Dict[Tuple[str, int], int] = {(eid, t): 0 for eid in engine_ids for t in range(1, H + 1)}
    first_start_month: Dict[str, int] = {eid: 0 for eid in engine_ids}

    rentals_plan: Dict[Tuple[int, int], int] = {(t, s): 0 for t in range(1, H + 1) for s in range(n_scenarios)}
    downtime_plan: Dict[Tuple[int, int], int] = {(t, s): 0 for t in range(1, H + 1) for s in range(n_scenarios)}

    t0 = 0
    while t0 < H:
        W_eff = min(W, H - t0)
        cap_window = scenario.shop_capacity[t0 : t0 + W_eff].copy()

        # Reduce capacity in first months due to ongoing shop visits
        for eid, remaining in state.shop_remaining.items():
            if remaining > 0:
                for k in range(min(remaining, W_eff)):
                    cap_window[k] -= 1

        fleet_now = _fleet_from_state(state)
        n_required = max(0, len(fleet_now.engines) - scenario.spares)

        # sample scenarios from current state
        dh = sample_deterioration_deltas(
            fleet=fleet_now,
            horizon_months=W_eff,
            n_scenarios=n_scenarios,
            params=scenario.deterioration,
            seed=seed + t0,
        )

        oper = build_operability_tensor(
            fleet=fleet_now,
            dh=dh,
            horizon_months=W_eff,
            n_scenarios=n_scenarios,
            h_min=scenario.h_min,
            shop_duration_months=scenario.shop_duration_months,
        )

        c_shop = build_expected_shop_costs(
            fleet=fleet_now,
            dh=dh,
            horizon_months=W_eff,
            n_scenarios=n_scenarios,
            costs=scenario.costs,
        )

        # solve the window (returns engine -> local start month 0..W_eff)
        window_res = solve_window(
            fleet_now,
            W_eff,
            cap_window,
            n_required,
            n_scenarios,
            scenario,
            oper,
            c_shop,
        )
        schedule_window = window_res.schedule

        # commit month-by-month for K months
        for k_step in range(K):
            abs_month = t0 + 1 + k_step
            if abs_month > H:
                break

            local_month = 1 + k_step

            shop_starts_now = {eid: 1 if int(schedule_window[eid]) == local_month else 0 for eid in engine_ids}

            for eid, start in shop_starts_now.items():
                plan_starts[(eid, abs_month)] = int(start)
                if start == 1 and first_start_month[eid] == 0:
                    first_start_month[eid] = abs_month

            # record slack for the committed month from the window solution
            for s in range(n_scenarios):
                for s in range(n_scenarios):
                    rentals_plan[(abs_month, s)] = int(window_res.rentals[(local_month, s)])
                    downtime_plan[(abs_month, s)] = int(window_res.downtime[(local_month, s)])

            # realized deterioration path for this committed month
            delta_realized = {eid: dh[(eid, local_month, realized_scenario_index)] for eid in engine_ids}

            _roll_forward_one_month(
                state=state,
                deterioration_delta=delta_realized,
                km_per_month=float(scenario.deterioration.km_per_month),
                shop_duration_months=int(scenario.shop_duration_months),
                shop_starts_this_month=shop_starts_now,
            )

        t0 += K

    return RollingPlan(
        shop_starts=plan_starts,
        first_start_month=first_start_month,
        rentals=rentals_plan,
        downtime=downtime_plan,
    )