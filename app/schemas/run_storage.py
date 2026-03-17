from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class StoredRunOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    run_id: str
    solver: str
    objective: float
    status: str
    created_at: datetime
    horizon_months: int
    n_engines: int


class ScheduleEntryOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    run_id: str
    engine_id: str
    shop_month: int