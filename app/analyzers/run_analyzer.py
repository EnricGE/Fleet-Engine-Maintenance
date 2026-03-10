from __future__ import annotations

from collections import Counter

from app.schemas.optimization import OptimizationResult
from app.schemas.run_analysis import RunAnalysisResult, RunSummary, ScheduleSummary


class RunAnalyzer:
    """
    Interprets optimization results and converts them into
    analysis-ready business / operational KPIs.
    """

    def analyze_result(
        self,
        result: OptimizationResult,
        max_rentals_per_month: int,
    ) -> RunAnalysisResult:
        schedule = result.schedule
        monthly_kpis = result.monthly_kpis

        n_engines = len(schedule)
        maintained_engines = [engine_id for engine_id, month in schedule.items() if month > 0]
        unmaintained_engines = [engine_id for engine_id, month in schedule.items() if month == 0]

        month_counter = Counter(schedule.values())
        shop_month_distribution = {int(month): int(count) for month, count in sorted(month_counter.items())}

        if monthly_kpis:
            avg_expected_rentals = sum(k.expected_rentals for k in monthly_kpis) / len(monthly_kpis)
            avg_expected_downtime = sum(k.expected_downtime for k in monthly_kpis) / len(monthly_kpis)
            worst_case_downtime = max(k.worst_case_downtime for k in monthly_kpis)

            n_months_with_downtime_risk = sum(1 for k in monthly_kpis if k.expected_downtime > 0)
            n_months_hitting_rental_cap = sum(
                1 for k in monthly_kpis if k.expected_rentals >= max_rentals_per_month
            )
        else:
            avg_expected_rentals = 0.0
            avg_expected_downtime = 0.0
            worst_case_downtime = 0.0
            n_months_with_downtime_risk = 0
            n_months_hitting_rental_cap = 0

        summary = RunSummary(
            run_id=result.run_id,
            solver=result.solver,
            objective=result.objective,
            n_engines=n_engines,
            n_maintained_engines=len(maintained_engines),
            n_unmaintained_engines=len(unmaintained_engines),
            avg_expected_rentals=avg_expected_rentals,
            avg_expected_downtime=avg_expected_downtime,
            worst_case_downtime=worst_case_downtime,
            n_months_with_downtime_risk=n_months_with_downtime_risk,
            n_months_hitting_rental_cap=n_months_hitting_rental_cap,
        )

        schedule_summary = ScheduleSummary(
            maintained_engines=maintained_engines,
            unmaintained_engines=unmaintained_engines,
            shop_month_distribution=shop_month_distribution,
        )

        return RunAnalysisResult(
            summary=summary,
            schedule_summary=schedule_summary,
        )