# scripts/run_ga_mealpy_schedule.py

from __future__ import annotations

from fleet_engine_planning.preprocessing.loaders import load_scenario
from fleet_engine_planning.simulation.deterioration import sample_deterioration_deltas
from fleet_engine_planning.optimizer.precompute import build_operability_tensor, build_expected_shop_costs
from fleet_engine_planning.solvers.ga_mealpy import solve_ga_mealpy


def main() -> None:
    scenario = load_scenario("data/v1/scenarios/baseline.json")

    T = scenario.horizon_months
    S = 30

    n_required = max(0, len(scenario.fleet.engines) - scenario.spares)
    print("Engines:", len(scenario.fleet.engines), "Spares:", scenario.spares, "Required:", n_required)

    # Sample stochastic deterioration scenarios (reproducible)
    dh = sample_deterioration_deltas(
        fleet=scenario.fleet,
        horizon_months=T,
        n_scenarios=S,
        params=scenario.deterioration,
        seed=123,
    )

    # Precompute operability tensor a[(engine_id, month, scenario, shop_month)]
    oper = build_operability_tensor(
        fleet=scenario.fleet,
        dh=dh,
        horizon_months=T,
        n_scenarios=S,
        h_min=scenario.h_min,
    )

    # Precompute expected shop costs E[C_shop | shop at month m]
    c_shop = build_expected_shop_costs(
        fleet=scenario.fleet,
        dh=dh,
        horizon_months=T,
        n_scenarios=S,
        costs=scenario.costs,
    )

    # Solve with GA (MEALPY)
    res = solve_ga_mealpy(
        fleet=scenario.fleet,
        horizon_months=T,
        shop_capacity=scenario.shop_capacity,
        n_required=n_required,
        n_scenarios=S,
        costs=scenario.costs,
        operable=oper,
        expected_shop_cost=c_shop,
        seed=123,
        epoch=400,
        pop_size=80,
        pc=0.9,
        pm=0.2,
    )

    if res is None:
        print("No solution returned by GA.")
        return

    print("\n=== GA Schedule (engine -> shop_month, 0 means none) ===")
    for k in sorted(res.schedule.keys()):
        print(k, "->", res.schedule[k])

    print("\n=== Expected rentals/downtime per month (avg over scenarios) ===")
    for t in range(1, T + 1):
        print(f"month {t:02d}: rentals={res.rentals_avg[t]:.2f}, downtime={res.downtime_avg[t]:.2f}")

    print("\nGA Objective (expected cost):", round(res.objective, 2))


if __name__ == "__main__":
    main()