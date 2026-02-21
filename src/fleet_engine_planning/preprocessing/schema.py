from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict

from fleet_engine_planning.fleet.engine import Fleet


@dataclass(frozen=True)
class Scenario:
    horizon_months: int
    shop_capacity: List[int]
    costs: Dict[str, float]
    failure_model: Dict[str, float]
    fleet: Fleet
    