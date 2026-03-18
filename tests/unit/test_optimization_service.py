"""
Unit tests for OptimizationService.

The CP-SAT solver is mocked so tests run fast and focus on the service's
own responsibilities: run ID consistency, status handling, and DB writes.
"""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import pytest
from sqlmodel import Session, select

from app.schemas.optimization import OptimizationRequest
from app.services.optimization_service import OptimizationService
from app.db.models import OptimizationRun
from fleet_engine_planning.solvers.cpsat_schedule import ScheduleResult


# ---------------------------------------------------------------------------
# Minimal fake ScheduleResult returned by the mocked solver
# ---------------------------------------------------------------------------

def _fake_schedule_result(n_scenarios: int = 5) -> ScheduleResult:
    schedule = {"E01": 3, "E02": 6, "E03": 0, "E04": 1}
    rentals = {(m, s): 0 for m in range(1, 13) for s in range(n_scenarios)}
    downtime = {(m, s): 0 for m in range(1, 13) for s in range(n_scenarios)}
    return ScheduleResult(
        schedule=schedule,
        objective=500000.0,
        rentals=rentals,
        downtime=downtime,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def service(in_memory_engine, monkeypatch):
    """OptimizationService wired to the in-memory DB."""
    def mock_get_session():
        return Session(in_memory_engine)

    monkeypatch.setattr("app.services.optimization_service.get_session", mock_get_session)
    return OptimizationService()


@pytest.fixture
def request_obj(sample_payload):
    return OptimizationRequest(**sample_payload)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRunIdConsistency:

    def test_returned_run_id_is_saved_in_db(self, service, request_obj, in_memory_engine):
        """The run_id in the returned result must match the one stored in the DB."""
        with patch(
            "app.services.optimization_service.solve_cpsat_schedule_with_rentals",
            return_value=_fake_schedule_result(n_scenarios=5),
        ):
            result = service.optimize_schedule(request_obj)

        with Session(in_memory_engine) as session:
            run = session.get(OptimizationRun, result.run_id)

        assert run is not None, "run_id returned to the caller was not found in the DB"
        assert run.run_id == result.run_id

    def test_run_id_is_a_string(self, service, request_obj):
        with patch(
            "app.services.optimization_service.solve_cpsat_schedule_with_rentals",
            return_value=_fake_schedule_result(),
        ):
            result = service.optimize_schedule(request_obj)

        assert isinstance(result.run_id, str)
        assert len(result.run_id) > 0


class TestSuccessResult:

    def test_status_is_success(self, service, request_obj):
        with patch(
            "app.services.optimization_service.solve_cpsat_schedule_with_rentals",
            return_value=_fake_schedule_result(),
        ):
            result = service.optimize_schedule(request_obj)

        assert result.status == "success"

    def test_schedule_is_populated(self, service, request_obj):
        with patch(
            "app.services.optimization_service.solve_cpsat_schedule_with_rentals",
            return_value=_fake_schedule_result(),
        ):
            result = service.optimize_schedule(request_obj)

        assert isinstance(result.schedule, dict)
        assert len(result.schedule) > 0

    def test_monthly_kpis_length_matches_horizon(self, service, request_obj):
        with patch(
            "app.services.optimization_service.solve_cpsat_schedule_with_rentals",
            return_value=_fake_schedule_result(),
        ):
            result = service.optimize_schedule(request_obj)

        assert len(result.monthly_kpis) == request_obj.horizon_months

    def test_objective_is_finite(self, service, request_obj):
        with patch(
            "app.services.optimization_service.solve_cpsat_schedule_with_rentals",
            return_value=_fake_schedule_result(),
        ):
            result = service.optimize_schedule(request_obj)

        import math
        assert math.isfinite(result.objective)


class TestNoSolutionResult:

    def test_status_is_no_solution_when_solver_returns_none(self, service, request_obj):
        with patch(
            "app.services.optimization_service.solve_cpsat_schedule_with_rentals",
            return_value=None,
        ):
            result = service.optimize_schedule(request_obj)

        assert result.status == "no_solution"

    def test_no_solution_schedule_is_empty(self, service, request_obj):
        with patch(
            "app.services.optimization_service.solve_cpsat_schedule_with_rentals",
            return_value=None,
        ):
            result = service.optimize_schedule(request_obj)

        assert result.schedule == {}

    def test_no_solution_does_not_write_to_db(self, service, request_obj, in_memory_engine):
        with patch(
            "app.services.optimization_service.solve_cpsat_schedule_with_rentals",
            return_value=None,
        ):
            result = service.optimize_schedule(request_obj)

        with Session(in_memory_engine) as session:
            run = session.get(OptimizationRun, result.run_id)

        assert run is None, "A no_solution run should not be persisted"


class TestUnsupportedSolver:

    def test_unsupported_solver_raises_not_implemented(self, service, sample_payload):
        sample_payload["settings"]["solver"] = "ga"
        request = OptimizationRequest(**sample_payload)
        with pytest.raises(NotImplementedError):
            service.optimize_schedule(request)
