"""
Unit tests for the GA solver (ga_mealpy.py).

Tests cover the three internal helpers directly — _capacity_usage,
_repair_capacity_with_duration, _evaluate_schedule — and the public
solve_ga_mealpy entry point. No mocking: all tests exercise real logic
with minimal hand-crafted fixtures (3 engines, 4-month horizon).
"""
from __future__ import annotations

import numpy as np
import pytest

from fleet_engine_planning.domain.engine import Engine, Fleet
from fleet_engine_planning.preprocessing.schema import CostParams
from fleet_engine_planning.solvers.ga_mealpy import (
    _capacity_usage,
    _evaluate_schedule,
    _repair_capacity_with_duration,
    solve_ga_mealpy,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

T = 4   # horizon months
S = 2   # scenarios
D = 1   # shop duration months

ENGINE_IDS = ["E01", "E02", "E03"]
CAP = [1, 1, 1, 1]  # 1 engine per month

FLEET = Fleet(engines=[
    Engine(engine_id="E01", age_months=10, distance_km=100_000, health=0.90),
    Engine(engine_id="E02", age_months=30, distance_km=400_000, health=0.55),
    Engine(engine_id="E03", age_months=40, distance_km=600_000, health=0.30),
])

COSTS = CostParams(
    base_maint_cost=100.0,
    rental_cost=500.0,
    downtime_cost=5000.0,
)

# All engines operable regardless of schedule / scenario
OPERABLE_ALL: dict = {
    (eid, t, s, m): 1
    for eid in ENGINE_IDS
    for t in range(1, T + 2)
    for s in range(S)
    for m in range(0, T + 1)
}

# Flat shop cost: 1000.0 per engine per month
C_SHOP: dict = {(eid, m): 1000.0 for eid in ENGINE_IDS for m in range(1, T + 1)}


# ---------------------------------------------------------------------------
# TestCapacityUsage
# ---------------------------------------------------------------------------

class TestCapacityUsage:

    def test_no_shop_yields_zero(self):
        months = np.array([0, 0, 0])
        usage = _capacity_usage(months, T, D)
        assert all(usage[t] == 0 for t in range(1, T + 1))

    def test_single_engine_duration_one(self):
        months = np.array([2])
        usage = _capacity_usage(months, T, D=1)
        assert usage[2] == 1
        assert usage[1] == 0
        assert usage[3] == 0

    def test_duration_two_spans_two_months(self):
        months = np.array([2])
        usage = _capacity_usage(months, T, D=2)
        assert usage[2] == 1
        assert usage[3] == 1
        assert usage[1] == 0
        assert usage[4] == 0

    def test_two_engines_same_month_cumulative(self):
        months = np.array([1, 1])
        usage = _capacity_usage(months, T, D=1)
        assert usage[1] == 2


# ---------------------------------------------------------------------------
# TestRepairCapacity
# ---------------------------------------------------------------------------

class TestRepairCapacity:

    def test_feasible_schedule_unchanged(self):
        # One engine per month — exactly at capacity
        months = np.array([1, 2, 3])
        rng = np.random.default_rng(0)
        result = _repair_capacity_with_duration(months, CAP, T, D, rng)
        assert list(result) == [1, 2, 3]

    def test_overcrowded_month_gets_repaired(self):
        # Two engines in month 1 but cap=1 — one must move or be set to 0
        months = np.array([1, 1, 3])
        rng = np.random.default_rng(0)
        result = _repair_capacity_with_duration(months, CAP, T, D, rng)
        usage = _capacity_usage(result, T, D)
        cap_arr = np.array([0] + CAP)
        assert np.all(usage[1:] <= cap_arr[1:])

    def test_result_always_feasible(self):
        # All engines crammed into month 1 — repair must fix all violations
        months = np.array([1, 1, 1])
        rng = np.random.default_rng(42)
        result = _repair_capacity_with_duration(months, CAP, T, D, rng)
        usage = _capacity_usage(result, T, D)
        cap_arr = np.array([0] + CAP)
        assert np.all(usage[1:] <= cap_arr[1:])

    def test_health_aware_evicts_healthiest(self):
        # E01 (health=0.90) and E02 (health=0.55) both in month 1, cap=1.
        # Only 1 slot → the healthier engine (E01, index 0) should be evicted to 0.
        months = np.array([1, 1])
        health = np.array([0.90, 0.55])
        cap = [1, 1, 1, 1]
        rng = np.random.default_rng(0)
        result = _repair_capacity_with_duration(months, cap, T, D, rng, engine_health=health)
        # E01 (index 0) should be moved; E02 (index 1) should stay at month 1
        assert result[0] != 1 or result[1] != 1  # at least one moved
        # The one that stayed in month 1 should be the sicker engine (E02)
        if result[0] == 1:
            assert health[0] <= health[1]
        else:
            assert result[1] == 1 and health[1] <= health[0]


# ---------------------------------------------------------------------------
# TestEvaluateSchedule
# ---------------------------------------------------------------------------

class TestEvaluateSchedule:

    def test_all_operable_zero_slack_cost(self):
        months = np.array([0, 0, 0])
        total, rentals, downtime = _evaluate_schedule(
            engine_ids=ENGINE_IDS, months=months,
            T=T, S=S, n_required=2,
            costs=COSTS, operable=OPERABLE_ALL,
            expected_shop_cost=C_SHOP,
            max_rentals_per_month=4,
        )
        assert all(rentals[(t, s)] == 0 for t in range(1, T + 1) for s in range(S))
        assert all(downtime[(t, s)] == 0 for t in range(1, T + 1) for s in range(S))

    def test_shortage_fills_rentals_first(self):
        # E01 inoperable, n_required=3, max_r=2 → shortage=1, rent=1, down=0
        operable = dict(OPERABLE_ALL)
        for t in range(1, T + 2):
            for s in range(S):
                for m in range(0, T + 1):
                    operable[("E01", t, s, m)] = 0

        months = np.array([0, 0, 0])
        _, rentals, downtime = _evaluate_schedule(
            engine_ids=ENGINE_IDS, months=months,
            T=T, S=S, n_required=3,
            costs=COSTS, operable=operable,
            expected_shop_cost=C_SHOP,
            max_rentals_per_month=2,
        )
        for t in range(1, T + 1):
            for s in range(S):
                assert rentals[(t, s)] == 1
                assert downtime[(t, s)] == 0

    def test_shortage_spills_to_downtime(self):
        # All engines inoperable, n_required=3, max_r=1 → shortage=3, rent=1, down=2
        operable_none = {k: 0 for k in OPERABLE_ALL}
        months = np.array([0, 0, 0])
        _, rentals, downtime = _evaluate_schedule(
            engine_ids=ENGINE_IDS, months=months,
            T=T, S=S, n_required=3,
            costs=COSTS, operable=operable_none,
            expected_shop_cost=C_SHOP,
            max_rentals_per_month=1,
        )
        for t in range(1, T + 1):
            for s in range(S):
                assert rentals[(t, s)] == 1
                assert downtime[(t, s)] == 2

    def test_shop_cost_added_for_scheduled_engine(self):
        # E01 scheduled at m=2 → shop cost of 1000.0 included
        months = np.array([2, 0, 0])
        total, _, _ = _evaluate_schedule(
            engine_ids=ENGINE_IDS, months=months,
            T=T, S=S, n_required=0,
            costs=COSTS, operable=OPERABLE_ALL,
            expected_shop_cost=C_SHOP,
            max_rentals_per_month=4,
        )
        assert total == pytest.approx(1000.0)

    def test_no_shop_engine_excluded_from_shop_cost(self):
        # All at m=0 → no shop cost, total=0
        months = np.array([0, 0, 0])
        total, _, _ = _evaluate_schedule(
            engine_ids=ENGINE_IDS, months=months,
            T=T, S=S, n_required=0,
            costs=COSTS, operable=OPERABLE_ALL,
            expected_shop_cost=C_SHOP,
            max_rentals_per_month=4,
        )
        assert total == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TestSolveGaMealpy
# ---------------------------------------------------------------------------

class TestSolveGaMealpy:
    """Fast integration tests — small epoch/pop_size to keep runtime low."""

    GA_KWARGS = dict(epoch=10, pop_size=10)

    def test_returns_schedule_result(self):
        result = solve_ga_mealpy(
            fleet=FLEET, horizon_months=T, shop_capacity=CAP,
            shop_duration_months=D, max_rentals_per_month=2,
            n_required=2, n_scenarios=S, costs=COSTS,
            operable=OPERABLE_ALL, expected_shop_cost=C_SHOP,
            seed=42, **self.GA_KWARGS,
        )
        assert result is not None

    def test_schedule_keys_match_engine_ids(self):
        result = solve_ga_mealpy(
            fleet=FLEET, horizon_months=T, shop_capacity=CAP,
            shop_duration_months=D, max_rentals_per_month=2,
            n_required=2, n_scenarios=S, costs=COSTS,
            operable=OPERABLE_ALL, expected_shop_cost=C_SHOP,
            seed=42, **self.GA_KWARGS,
        )
        assert set(result.schedule.keys()) == set(ENGINE_IDS)

    def test_schedule_values_in_valid_range(self):
        result = solve_ga_mealpy(
            fleet=FLEET, horizon_months=T, shop_capacity=CAP,
            shop_duration_months=D, max_rentals_per_month=2,
            n_required=2, n_scenarios=S, costs=COSTS,
            operable=OPERABLE_ALL, expected_shop_cost=C_SHOP,
            seed=42, **self.GA_KWARGS,
        )
        for month in result.schedule.values():
            assert 0 <= month <= T

    def test_capacity_respected(self):
        result = solve_ga_mealpy(
            fleet=FLEET, horizon_months=T, shop_capacity=CAP,
            shop_duration_months=D, max_rentals_per_month=2,
            n_required=2, n_scenarios=S, costs=COSTS,
            operable=OPERABLE_ALL, expected_shop_cost=C_SHOP,
            seed=42, **self.GA_KWARGS,
        )
        months = np.array([result.schedule[eid] for eid in ENGINE_IDS])
        usage = _capacity_usage(months, T, D)
        cap_arr = np.array([0] + CAP)
        assert np.all(usage[1:] <= cap_arr[1:])

    def test_converges_to_optimal_on_trivial_problem(self):
        # All engines always operable, n_required=0 → optimal cost is 0.0
        # (no shop visits needed). Both runs should find it reliably.
        kwargs = dict(
            fleet=FLEET, horizon_months=T, shop_capacity=CAP,
            shop_duration_months=D, max_rentals_per_month=2,
            n_required=0, n_scenarios=S, costs=COSTS,
            operable=OPERABLE_ALL, expected_shop_cost=C_SHOP,
            seed=42, **self.GA_KWARGS,
        )
        result_a = solve_ga_mealpy(**kwargs)
        result_b = solve_ga_mealpy(**kwargs)
        assert result_a.objective == pytest.approx(0.0)
        assert result_b.objective == pytest.approx(0.0)
