from fleet_engine_planning.fleet.engine import Engine, Fleet
from fleet_engine_planning.fleet.failure import WeibullFailureModel
from fleet_engine_planning.fleet.evaluate import PolicyParams, expected_cost_fleet, greedy_policy_by_threshold


def main() -> None:
    fleet = Fleet(
        engines=[
            Engine("E01", age=18, distance=250_000, health=0.80),
            Engine("E02", age=36, distance=520_000, health=0.55),
            Engine("E03", age=10, distance=120_000, health=0.95),
            Engine("E04", age=44, distance=700_000, health=0.35),
        ]
    )

    model = WeibullFailureModel(
        shape_k=2.2,
        scale_lambda_months=60.0,
        beta_age=0.6,
        beta_distance=0.4,
        beta_bad_health=1.2,
    )
    params = PolicyParams(base_maint_cost=12000.0, failure_penalty=250000.0)

    horizon = 12.0  # months
    decisions = greedy_policy_by_threshold(fleet, health_threshold=0.6)
    total_cost = expected_cost_fleet(fleet, model, horizon, decisions, params)

    for e in fleet.engines:
        print(e.engine_id, model.prob_fail_in_horizon(e, 12.0))

    print("Decisions:", decisions)
    print("Expected fleet cost:", round(total_cost, 2))


if __name__ == "__main__":
    main()