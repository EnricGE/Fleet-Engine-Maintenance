"""
Shared pytest fixtures for the Fleet Engine Planning test suite.
"""
from __future__ import annotations

import pytest
from sqlmodel import SQLModel, create_engine, Session
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient

# Import models so SQLModel.metadata is populated before any in_memory_engine
# fixture calls create_all.  Must happen at module level, before the fixtures run.
import app.db.models  # noqa: F401


# ---------------------------------------------------------------------------
# Reusable request payload (small fleet, fast solver settings)
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_payload() -> dict:
    return {
        "engines": [
            {"engine_id": "E01", "age_months": 24, "distance_km": 420000, "health": 0.72},
            {"engine_id": "E02", "age_months": 30, "distance_km": 510000, "health": 0.55},
            {"engine_id": "E03", "age_months": 18, "distance_km": 300000, "health": 0.66},
            {"engine_id": "E04", "age_months": 40, "distance_km": 700000, "health": 0.30},
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
            "n_scenarios": 5,
            "random_seed": 42,
            "time_limit_s": 10.0,
        },
    }


# ---------------------------------------------------------------------------
# In-memory SQLite engine (function-scoped → fresh DB per test)
# ---------------------------------------------------------------------------

@pytest.fixture
def in_memory_engine():
    # StaticPool reuses a single connection so tables created by create_all
    # are visible to all subsequent Session() calls on this engine.
    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(eng)
    yield eng
    SQLModel.metadata.drop_all(eng)


# ---------------------------------------------------------------------------
# FastAPI TestClient wired to the in-memory DB
# ---------------------------------------------------------------------------

@pytest.fixture
def test_client(in_memory_engine, monkeypatch):
    """
    Patches the module-level engine references so that all DB operations
    inside the API and the service use the in-memory SQLite instance.
    """
    def mock_get_session():
        return Session(in_memory_engine)

    # All routes and OptimizationService use get_session() — one patch covers both
    monkeypatch.setattr("app.main.get_session", mock_get_session)
    monkeypatch.setattr("app.services.optimization_service.get_session", mock_get_session)

    # Prevent lifespan from touching the real file-based DB
    monkeypatch.setattr("app.db.database.create_db_and_tables", lambda: None)

    from app.main import app
    with TestClient(app, raise_server_exceptions=True) as client:
        yield client
