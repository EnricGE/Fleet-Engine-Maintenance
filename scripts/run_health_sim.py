from fleet_engine_planning.preprocessing.loaders import load_scenario
from fleet_engine_planning.simulation.deterioration import simulate_health_paths, operable_flags


def main() -> None:
    scenario = load_scenario("data/v1/scenarios/baseline.json")

    T = scenario.horizon_months
    S = 30 
    paths = simulate_health_paths(
        fleet=scenario.fleet,
        horizon_months=T,
        n_scenarios=S,
        params=scenario.deterioration,
        seed=123,
    )
    op = operable_flags(paths, scenario.h_min)

    # Print a quick summary for first engine
    e0 = scenario.fleet.engines[3].engine_id
    print("Engine:", e0)
    for t in [0, 1, 3, 6, T]:
        hs = [paths[(e0, t, s)] for s in range(S)]
        ops = sum(op[(e0, t, s)] for s in range(S))
        print(f"t={t:02d}  health avg={sum(hs)/S:.3f}  operable={ops}/{S}")

    # Check required engines feasibility at t=0 across scenarios (should be ok)
    n_required = max(0, len(scenario.fleet.engines) - scenario.spares)
    print("n_required:", n_required)

    for t in [0, 3, 6, T]:
        operable_counts = []
        for s in range(S):
            count = sum(op[(e.engine_id, t, s)] for e in scenario.fleet.engines)
            operable_counts.append(count)
        print(f"t={t:02d}  operable engines avg={sum(operable_counts)/S:.2f}  min={min(operable_counts)}")

if __name__ == "__main__":
    main()