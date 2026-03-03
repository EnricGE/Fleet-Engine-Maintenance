from __future__ import annotations

from pathlib import Path

from fleet_engine_planning.preprocessing.loaders import load_scenario
from fleet_engine_planning.controllers.rolling_horizon import run_rolling_horizon
from fleet_engine_planning.solvers.cpsat_schedule import solve_cpsat_schedule_with_rentals
from fleet_engine_planning.visualization.schedule_plots import save_standard_plots
from fleet_engine_planning.controllers.rolling_horizon import WindowSolveResult


def solve_window_cpsat(
    fleet,
    W_eff: int,
    cap_window: list[int],
    n_required: int,
    n_scenarios: int,
    scenario,
    oper,
    c_shop,
) -> dict[str, int]:
    res = solve_cpsat_schedule_with_rentals(
        fleet=fleet,
        horizon_months=W_eff,
        shop_capacity=cap_window,
        n_required=n_required,
        n_scenarios=n_scenarios,
        costs=scenario.costs,
        operable=oper,
        expected_shop_cost=c_shop,
        shop_duration_months=scenario.shop_duration_months,
        max_rentals_per_month=scenario.max_rentals_per_month,
        time_limit_s=10.0,
    )
    if res is None:
        raise RuntimeError("CP-SAT returned no solution for rolling window")
    
    return WindowSolveResult(
        schedule=res.schedule,
        rentals=res.rentals,
        downtime=res.downtime,
    )


def main() -> None:
    scenario = load_scenario("data/v1/scenarios/baseline.json")

    H = int(scenario.horizon_months)
    W = int(scenario.window_length)
    K = int(scenario.commit_length)
    S = 30
    seed = 123

    plan = run_rolling_horizon(
        scenario=scenario,
        solve_window=solve_window_cpsat,
        H=H,
        W=W,
        K=K,
        n_scenarios=S,
        seed=seed,
        realized_scenario_index=0,
    )

    print("\n=== Rolling horizon (first shop start month) ===")
    for eid in sorted(plan.first_start_month.keys()):
        print(eid, "->", plan.first_start_month[eid])

    rentals_full = plan.rentals
    downtime_full = plan.downtime
    expected_shop_cost_full = {(eid, t): 0.0 for eid in plan.first_start_month.keys() for t in range(1, H + 1)}

    save_standard_plots(
        schedule=plan.first_start_month,
        rentals=rentals_full,
        downtime=downtime_full,
        horizon_months=H,
        n_scenarios=S,
        shop_capacity=scenario.shop_capacity[:H],
        shop_duration_months=scenario.shop_duration_months,
        max_rentals_per_month=scenario.max_rentals_per_month,
        expected_shop_cost=expected_shop_cost_full,
        rental_cost=scenario.costs.rental_cost,
        downtime_cost=scenario.costs.downtime_cost,
        out_dir=Path("outputs"),
        prefix="rolling_cpsat",
    )

    # Save plan starts to file
    out_plan = Path("outputs") / "rolling_plan_shop_starts.txt"
    out_plan.parent.mkdir(parents=True, exist_ok=True)
    with open(out_plan, "w") as f:
        f.write("engine_id,month,shop_start\n")
        for eid in sorted(plan.first_start_month.keys()):
            for t in range(1, H + 1):
                f.write(f"{eid},{t},{plan.shop_starts[(eid, t)]}\n")

    print(f"\nSaved rolling plan to {out_plan}")
    print("Saved rolling plots to outputs/")


if __name__ == "__main__":
    main()