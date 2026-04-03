from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, ConfigDict


class EngineStateIn(BaseModel):
    """
    External input representation of one engine. 
    What API client sends.
    """
    engine_id: str = Field(..., description="Unique engine identifier")
    age_months: float = Field(..., ge=0, description="Engine age in months")
    distance_km: float = Field(..., ge=0, description="Engine accumulated distance in km")
    health: float = Field(..., ge=0.0, le=1.0, description="Health index in [0,1]")


class OptimizationSettings(BaseModel):
    """
    Run-time options for the optimization service.
    """
    solver: Literal["cpsat", "ga", "rolling_cpsat"] = Field(
        default="cpsat",
        description="Optimization method to use",
    )
    n_scenarios: int = Field(default=30, ge=1, description="Number of deterioration scenarios")
    random_seed: int = Field(default=123, description="Random seed for reproducibility")
    time_limit_s: float = Field(default=10.0, gt=0, description="Solver time limit in seconds")


class OptimizationRequest(BaseModel):
    """
    Full request payload for schedule optimization.
    """
    model_config = ConfigDict(extra="forbid")

    engines: list[EngineStateIn] = Field(..., description="Fleet engine states")

    horizon_months: int = Field(..., ge=1)
    shop_capacity: list[int] = Field(..., description="Monthly shop capacity")
    shop_duration_months: int = Field(..., ge=1)
    spares: int = Field(..., ge=0)
    h_min: float = Field(..., ge=0.0, le=1.0)
    max_rentals_per_month: int = Field(..., ge=0)

    base_maint_cost: float = Field(..., ge=0)
    rental_cost: float = Field(..., ge=0)
    downtime_cost: float = Field(..., ge=0)
    gamma_health_cost: float = Field(default=1.0, ge=0)

    terminal_inop_cost: float = Field(default=0.0, ge=0)
    terminal_shortfall_cost: float = Field(default=0.0, ge=0)

    km_per_month: float = Field(..., ge=0)
    mu_base: float = Field(..., ge=0)
    mu_per_1000km: float = Field(..., ge=0)
    sigma: float = Field(..., ge=0)

    window_length: int = Field(default=6, ge=1)
    commit_length: int = Field(default=1, ge=1)

    settings: OptimizationSettings = Field(default_factory=OptimizationSettings)


class MonthlyKPI(BaseModel):
    """
    One month of KPI outputs.
    """
    month: int
    expected_rentals: float
    expected_downtime: float
    worst_case_downtime: float


class OptimizationResult(BaseModel):
    """
    Standard service response returned by the optimization layer.
    """
    model_config = ConfigDict(extra="forbid")

    run_id: str
    solver: str
    objective: float
    schedule: dict[str, int]
    monthly_kpis: list[MonthlyKPI]
    status: str = "success"
    solver_status: str = "unknown"


