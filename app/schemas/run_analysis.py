from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class RunSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    solver: str
    objective: float

    n_engines: int
    n_maintained_engines: int
    n_unmaintained_engines: int

    avg_expected_rentals: float
    avg_expected_downtime: float
    worst_case_downtime: float

    n_months_with_downtime_risk: int
    n_months_hitting_rental_cap: int


class ScheduleSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    maintained_engines: list[str]
    unmaintained_engines: list[str]
    shop_month_distribution: dict[int, int]


class RunAnalysisResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: RunSummary
    schedule_summary: ScheduleSummary
