from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class ScheduleResult:
    schedule: Dict[str, int]              # engine_id -> shop_month (0..T)
    objective: float
    rentals: Dict[Tuple[int, int], int]   # (month, scenario) -> rentals
    downtime: Dict[Tuple[int, int], int]  # (month, scenario) -> downtime
    solver_status: str = "unknown"        # "optimal", "feasible", or "unknown"
