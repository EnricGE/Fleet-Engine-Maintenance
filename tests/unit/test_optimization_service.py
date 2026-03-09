import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from app.schemas.optimization import OptimizationRequest
from app.services.optimization_service import OptimizationService


payload = {
    "engines": [
        {
            "engine_id": "E01",
            "age_months": 24,
            "distance_km": 420000,
            "health": 0.72,
        },
        {
            "engine_id": "E02",
            "age_months": 30,
            "distance_km": 510000,
            "health": 0.55,
        },
        {
            "engine_id": "E03",
            "age_months": 18,
            "distance_km": 300000,
            "health": 0.66,
        },
        {
            "engine_id": "E04",
            "age_months": 40,
            "distance_km": 700000,
            "health": 0.30,
        },
    ],
    "horizon_months": 12,
    "shop_capacity": [2] * 12,
    "shop_duration_months": 2,
    "spares": 1,
    "h_min": 0.2,
    "max_rentals_per_month": 4,
    "base_maint_cost": 300000,
    "rental_cost": 50000,
    "downtime_cost": 2000000,
    "gamma_health_cost": 1.5,
    "terminal_inop_cost": 100000,
    "terminal_shortfall_cost": 100000,
    "km_per_month": 25000,
    "mu_base": 0.01,
    "mu_per_1000km": 0.00025,
    "sigma": 0.004,
    "window_length": 6,
    "commit_length": 2,
    "settings": {
        "solver": "cpsat",
        "n_scenarios": 10,
        "random_seed": 123,
        "time_limit_s": 5.0,
    },
}

request = OptimizationRequest(**payload)
service = OptimizationService()
result = service.optimize_schedule(request)

print(result.model_dump())