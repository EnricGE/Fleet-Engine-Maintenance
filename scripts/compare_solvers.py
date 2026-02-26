from __future__ import annotations

from collections import Counter

from fleet_engine_planning.preprocessing.loaders import load_scenario
from fleet_engine_planning.simulation.deterioration import sample_deterioration_deltas
from fleet_engine_planning.optimizer.precompute import build_operability_tensor, build_expected_shop_costs
from fleet_engine_planning.solvers.cpsat_schedule import solve_cpsat_schedule_with_rentals
from fleet_engine_planning.solvers.ga_mealpy import solve_ga_mealpy


def _month_hist(schedule: dict[str, int], horizon: int) -> dict[int, int]:
    c = Counter(schedule.values())
    return {m: c.get(m, 0) for m in range(0, horizon + 1)}


def main() -> None:
    scenario = load_scenario("data/v1/scenarios/baseline.json")

    T = scenario.horizon_months
    S = 30 
    seed = 123

    n_required = max(0, len(scenario.fleet.engines) - scenario.spares)
    print("Engines:", len(scenario.fleet.engines), ", Spares:", scenario.spares, ", Required:", n_required)
    print("Horizon:", T, "Scenarios:", S)
    print()

    # Shared stochastic scenarios (same dh -> fair comparison)
    dh = sample_deterioration_deltas(
        fleet=scenario.fleet,
        horizon_months=T,
        n_scenarios=S,
        params=scenario.deterioration,
        seed=seed,
    )

    oper = build_operability_tensor(
        fleet=scenario.fleet,
        dh=dh,
        horizon_months=T,
        n_scenarios=S,
        h_min=scenario.h_min,
        shop_duration_months=scenario.shop_duration_months,
    )

    c_shop = build_expected_shop_costs(
        fleet=scenario.fleet,
        dh=dh,
        horizon_months=T,
        n_scenarios=S,
        costs=scenario.costs,
    )

    # --- CP-SAT ---
    cpsat = solve_cpsat_schedule_with_rentals(
        fleet=scenario.fleet,
        horizon_months=T,
        shop_capacity=scenario.shop_capacity,
        shop_duration_months=scenario.shop_duration_months,
        max_rentals_per_month=scenario.max_rentals_per_month,
        n_required=n_required,
        n_scenarios=S,
        costs=scenario.costs,
        operable=oper,
        expected_shop_cost=c_shop,
        time_limit_s=10.0,
    )

    # --- GA (MEALPY) ---
    ga = solve_ga_mealpy(
        fleet=scenario.fleet,
        horizon_months=T,
        shop_capacity=scenario.shop_capacity,
        n_required=n_required,
        n_scenarios=S,
        costs=scenario.costs,
        operable=oper,
        expected_shop_cost=c_shop,
        shop_duration_months=scenario.shop_duration_months,
        max_rentals_per_month=scenario.max_rentals_per_month,
        seed=seed,
        epoch=200,
        pop_size=80,
        pc=0.9,
        pm=0.2,
    )

    # --- Print summary ---
    print("=== RESULTS ===")
    if cpsat is None:
        print("CP-SAT: No feasible solution")
    else:
        print(f"CP-SAT objective: {cpsat.objective:.2f}")

    if ga is None:
        print("GA: No solution returned")
    else:
        print(f"GA objective:     {ga.objective:.2f}")

    print()

    # Capacity sanity check
    cap = {m + 1: scenario.shop_capacity[m] for m in range(T)}

    if cpsat is not None:
        hist = _month_hist(cpsat.schedule, T)
        print("CP-SAT shop-month histogram (0 means none):")
        for m in range(0, T + 1):
            if m == 0:
                print(f"  m=0: {hist[m]}")
            else:
                ok = "OK" if hist[m] <= cap[m] else "VIOLATION"
                print(f"  m={m:02d}: {hist[m]} / cap={cap[m]}  {ok}")
        print()

    if ga is not None:
        hist = _month_hist(ga.schedule, T)
        print("GA shop-month histogram (0 means none):")
        for m in range(0, T + 1):
            if m == 0:
                print(f"  m=0: {hist[m]}")
            else:
                ok = "OK" if hist[m] <= cap[m] else "VIOLATION"
                print(f"  m={m:02d}: {hist[m]} / cap={cap[m]}  {ok}")
        print()

    # Rentals/Downtime profile
    print("Expected rentals/downtime per month (avg over scenarios):")
    for t in range(1, T + 1):
        if cpsat is not None:
            avg_r_c = sum(cpsat.rentals[(t, s)] for s in range(S)) / S
            avg_d_c = sum(cpsat.downtime[(t, s)] for s in range(S)) / S
            max_d_c = max(cpsat.downtime[(t, s)] for s in range(S))
        else:
            avg_r_c = avg_d_c = max_d_c = float("nan")

        if ga is not None:
            avg_r_g = sum(ga.rentals[(t, s)] for s in range(S)) / S
            avg_d_g = sum(ga.downtime[(t, s)] for s in range(S)) / S
            max_d_g = max(ga.downtime[(t, s)] for s in range(S))
        else:
            avg_r_g = avg_d_g = max_d_g = float("nan")

        print(
            f"month {t:02d} | "
            f"CP-SAT rentals={avg_r_c:6.2f} downtime={avg_d_c:6.2f} worst={max_d_c:2.0f} "
            f"|| GA rentals={avg_r_g:6.2f} downtime={avg_d_g:6.2f} worst={max_d_g:2.0f}"
        )

    if cpsat is not None:
        print("CP-SAT schedule (engine -> month):")
        for k in sorted(cpsat.schedule.keys()):
            print(" ", k, "->", cpsat.schedule[k])
        print()

    if ga is not None:
        print("GA schedule (engine -> month):")
        for k in sorted(ga.schedule.keys()):
            print(" ", k, "->", ga.schedule[k])


if __name__ == "__main__":
    main()