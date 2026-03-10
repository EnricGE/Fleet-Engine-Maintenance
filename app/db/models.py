from typing import Optional
from datetime import datetime

from sqlmodel import SQLModel, Field


class OptimizationRun(SQLModel, table=True):
    run_id: str = Field(primary_key=True)

    solver: str
    objective: float
    status: str

    created_at: datetime = Field(default_factory=datetime.utcnow)

    horizon_months: int
    n_engines: int


class ScheduleEntry(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)

    run_id: str = Field(index=True)
    engine_id: str
    shop_month: int