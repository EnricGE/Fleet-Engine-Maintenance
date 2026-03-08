from __future__ import annotations

from pathlib import Path

from fleet_engine_planning.preprocessing.loaders import load_scenario
from fleet_engine_planning.simulation.deterioration import sample_deterioration_deltas
from fleet_engine_planning.optimization.precompute import build_operability_tensor, build_expected_shop_costs
from fleet_engine_planning.solvers.cpsat_schedule import solve_cpsat_schedule_with_rentals
from fleet_engine_planning.visualization.schedule_plots import save_standard_plots
from fleet_engine_planning.visualization.health_trajectories import plot_engine_health_trajectories


def main() -> None:
    scenario = load_scenario("data/v1/scenarios/baseline.json")

    T = scenario.horizon_months
    S = 30

    n_required = max(0, len(scenario.fleet.engines) - scenario.spares)
    print("Engines:", len(scenario.fleet.engines), ", Spares:", scenario.spares, ", Required:", n_required)

    dh = sample_deterioration_deltas(
        fleet=scenario.fleet,
        horizon_months=T+1,
        n_scenarios=S,
        params=scenario.deterioration,
        seed=123,
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

    res = solve_cpsat_schedule_with_rentals(
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

    if res is None:
        print("No feasible solution found.")
        return

    print("\n=== Schedule (engine -> shop_month, 0 means none) ===")
    for k in sorted(res.schedule.keys()):
        print(k, "->", res.schedule[k])

    # summarize expected rentals/downtime per month
    print("\n=== Expected rentals/downtime per month (avg over scenarios) ===")
    for t in range(1, T + 1):
        avg_r = sum(res.rentals[(t, s)] for s in range(S)) / S
        avg_d = sum(res.downtime[(t, s)] for s in range(S)) / S
        max_d = max(res.downtime[(t,s)] for s in range(S))
        print(f"month {t:02d}: rentals={avg_r:.2f}, downtime={avg_d:.2f}, worst-case={max_d}")

    # objective reporting: note rentals/downtime were summed across scenarios in objective
    # (so objective scale is consistent, but not divided by S). We'll keep it as-is for now.
    print("\nObjective (scaled sum):", round(res.objective, 2))

    save_standard_plots(
        schedule=res.schedule,
        rentals=res.rentals,
        downtime=res.downtime,
        horizon_months=T,
        n_scenarios=S,
        shop_capacity=scenario.shop_capacity,
        shop_duration_months=scenario.shop_duration_months,
        max_rentals_per_month=scenario.max_rentals_per_month,
        expected_shop_cost=c_shop,
        rental_cost=scenario.costs.rental_cost,
        downtime_cost=scenario.costs.downtime_cost,
        out_dir=Path("outputs"),
        prefix="cpsat",
    )
    print("\nSaved plots to outputs/")

    engine = scenario.fleet.engines[0]
    eid = engine.engine_id

    plot_engine_health_trajectories(
        engine_id=eid,
        h0=engine.health,
        dh=dh,
        horizon_months=T,
        n_scenarios=S,
        shop_start_month=res.schedule[eid],
        shop_duration_months=scenario.shop_duration_months,
        h_min=scenario.h_min,
        out_path=Path("outputs") / f"cpsat_health_trajectories_{eid}.png",
        n_paths=35,   # 20–50
        seed=1,
    )

    print(f"Saved health trajectory plot for {eid} to outputs/")


if __name__ == "__main__":
    main()