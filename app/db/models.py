from typing import Optional
from datetime import datetime, UTC

from sqlmodel import SQLModel, Field


class OptimizationRun(SQLModel, table=True):
    run_id: str = Field(primary_key=True)

    solver: str
    solver_status: str
    objective: float
    status: str

    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    horizon_months: int
    n_engines: int
    max_rentals_per_month: int
    shop_duration_months: int


class ScheduleEntry(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)

    run_id: str = Field(index=True, foreign_key="optimizationrun.run_id")
    engine_id: str
    shop_month: int


class MonthlyKPIRecord(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)

    run_id: str = Field(index=True, foreign_key="optimizationrun.run_id")
    month: int

    expected_rentals: float
    expected_downtime: float
    worst_case_downtime: float