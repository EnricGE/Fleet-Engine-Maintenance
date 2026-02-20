from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Engine:
    """State of one engine at decision time."""
    engine_id: str
    age_months: float          # month
    distance_km: float     # km
    health: float # [0..1]


@dataclass
class Fleet:
    engines: List[Engine]