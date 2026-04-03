from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class StoredRunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    run_id: str
    solver: str
    solver_status: str
    objective: float
    status: str
    created_at: datetime
    horizon_months: int
    n_engines: int
    max_rentals_per_month: int
    shop_duration_months: int


class ScheduleEntryOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    run_id: str
    engine_id: str
    shop_month: int


class MonthlyKPIOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    month: int
    expected_rentals: float
    expected_downtime: float
    worst_case_downtime: float