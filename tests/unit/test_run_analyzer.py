"""
Unit tests for RunAnalyzer — pure logic, no DB required.
"""
from __future__ import annotations

import pytest

from app.analyzers.run_analyzer import RunAnalyzer
from app.schemas.optimization import MonthlyKPI, OptimizationResult


def _make_result(
    schedule: dict[str, int],
    monthly_kpis: list[MonthlyKPI],
) -> OptimizationResult:
    return OptimizationResult(
        run_id="test-run-id",
        solver="cpsat",
        objective=1234.56,
        schedule=schedule,
        monthly_kpis=monthly_kpis,
        status="success",
    )


def _kpi(month: int, rentals: float, downtime: float, worst: float) -> MonthlyKPI:
    return MonthlyKPI(
        month=month,
        expected_rentals=rentals,
        expected_downtime=downtime,
        worst_case_downtime=worst,
    )


@pytest.fixture
def analyzer():
    return RunAnalyzer()


class TestAnalyzeResultEngineCounts:

    def test_maintained_engines_identified(self, analyzer):
        result = _make_result(
            schedule={"E01": 3, "E02": 0, "E03": 7, "E04": 0},
            monthly_kpis=[],
        )
        analysis = analyzer.analyze_result(result, max_rentals_per_month=4)
        assert set(analysis.schedule_summary.maintained_engines) == {"E01", "E03"}

    def test_unmaintained_engines_identified(self, analyzer):
        result = _make_result(
            schedule={"E01": 3, "E02": 0, "E03": 7, "E04": 0},
            monthly_kpis=[],
        )
        analysis = analyzer.analyze_result(result, max_rentals_per_month=4)
        assert set(analysis.schedule_summary.unmaintained_engines) == {"E02", "E04"}

    def test_n_engines_total(self, analyzer):
        result = _make_result(
            schedule={"E01": 1, "E02": 2, "E03": 0},
            monthly_kpis=[],
        )
        analysis = analyzer.analyze_result(result, max_rentals_per_month=4)
        assert analysis.summary.n_engines == 3
        assert analysis.summary.n_maintained_engines == 2
        assert analysis.summary.n_unmaintained_engines == 1

    def test_all_maintained(self, analyzer):
        result = _make_result(
            schedule={"E01": 1, "E02": 2},
            monthly_kpis=[],
        )
        analysis = analyzer.analyze_result(result, max_rentals_per_month=4)
        assert analysis.summary.n_unmaintained_engines == 0

    def test_none_maintained(self, analyzer):
        result = _make_result(
            schedule={"E01": 0, "E02": 0},
            monthly_kpis=[],
        )
        analysis = analyzer.analyze_result(result, max_rentals_per_month=4)
        assert analysis.summary.n_maintained_engines == 0


class TestAnalyzeResultShopDistribution:

    def test_shop_month_distribution(self, analyzer):
        result = _make_result(
            schedule={"E01": 3, "E02": 3, "E03": 7, "E04": 0},
            monthly_kpis=[],
        )
        analysis = analyzer.analyze_result(result, max_rentals_per_month=4)
        dist = analysis.schedule_summary.shop_month_distribution
        assert dist[3] == 2
        assert dist[7] == 1
        assert dist[0] == 1


class TestAnalyzeResultKPIAggregation:

    def test_avg_expected_rentals(self, analyzer):
        kpis = [_kpi(1, 2.0, 0.0, 0.0), _kpi(2, 4.0, 0.0, 0.0)]
        result = _make_result(schedule={"E01": 1}, monthly_kpis=kpis)
        analysis = analyzer.analyze_result(result, max_rentals_per_month=4)
        assert analysis.summary.avg_expected_rentals == pytest.approx(3.0)

    def test_avg_expected_downtime(self, analyzer):
        kpis = [_kpi(1, 0.0, 1.0, 1.0), _kpi(2, 0.0, 3.0, 3.0)]
        result = _make_result(schedule={"E01": 1}, monthly_kpis=kpis)
        analysis = analyzer.analyze_result(result, max_rentals_per_month=4)
        assert analysis.summary.avg_expected_downtime == pytest.approx(2.0)

    def test_worst_case_downtime(self, analyzer):
        kpis = [_kpi(1, 0.0, 1.0, 2.0), _kpi(2, 0.0, 0.5, 5.0), _kpi(3, 0.0, 0.0, 0.0)]
        result = _make_result(schedule={"E01": 1}, monthly_kpis=kpis)
        analysis = analyzer.analyze_result(result, max_rentals_per_month=4)
        assert analysis.summary.worst_case_downtime == pytest.approx(5.0)

    def test_n_months_with_downtime_risk(self, analyzer):
        kpis = [_kpi(1, 0.0, 1.0, 1.0), _kpi(2, 0.0, 0.0, 0.0), _kpi(3, 0.0, 0.5, 1.0)]
        result = _make_result(schedule={"E01": 1}, monthly_kpis=kpis)
        analysis = analyzer.analyze_result(result, max_rentals_per_month=4)
        assert analysis.summary.n_months_with_downtime_risk == 2

    def test_n_months_hitting_rental_cap(self, analyzer):
        # cap is 4
        kpis = [_kpi(1, 4.0, 0.0, 0.0), _kpi(2, 3.9, 0.0, 0.0), _kpi(3, 5.0, 0.0, 0.0)]
        result = _make_result(schedule={"E01": 1}, monthly_kpis=kpis)
        analysis = analyzer.analyze_result(result, max_rentals_per_month=4)
        assert analysis.summary.n_months_hitting_rental_cap == 2

    def test_empty_kpis_return_zeros(self, analyzer):
        result = _make_result(schedule={"E01": 1}, monthly_kpis=[])
        analysis = analyzer.analyze_result(result, max_rentals_per_month=4)
        assert analysis.summary.avg_expected_rentals == 0.0
        assert analysis.summary.avg_expected_downtime == 0.0
        assert analysis.summary.worst_case_downtime == 0.0
        assert analysis.summary.n_months_with_downtime_risk == 0
        assert analysis.summary.n_months_hitting_rental_cap == 0
