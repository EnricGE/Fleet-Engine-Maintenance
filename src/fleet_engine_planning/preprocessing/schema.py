from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict

from fleet_engine_planning.fleet.engine import Fleet


@dataclass(frozen=True)
class Scenario:
    horizon_months: int
    shop_capacity: list[int]
    shop_duration_months: int
    spares: int
    max_rentals_per_month: int
    h_min: float
    costs: CostParams
    deterioration: DeteriorationParams
    fleet: Fleet

    @property
    def n_required(self) -> int:
        return max(0, len(self.fleet.engines) - self.spares)
    

@dataclass(frozen=True)
class CostParams:
    base_maint_cost: float
    rental_cost: float
    downtime_cost: float
    gamma_health_cost: float = 1.0


@dataclass(frozen=True)
class DeteriorationParams:
    km_per_month: float
    mu_base: float
    mu_per_1000km: float
    sigma: float