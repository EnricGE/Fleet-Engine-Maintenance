"""
Integration tests for the Fleet Engine Planning API.

Uses the real CP-SAT solver with a small fleet and few scenarios
so tests are fast but exercise the full stack end-to-end.
"""
from __future__ import annotations

import pytest


class TestHealthEndpoint:

    def test_health_returns_200(self, test_client):
        response = test_client.get("/health")
        assert response.status_code == 200

    def test_health_returns_ok(self, test_client):
        assert test_client.get("/health").json() == {"status": "ok"}


class TestOptimizeScheduleEndpoint:

    def test_optimize_returns_200(self, test_client, sample_payload):
        response = test_client.post("/optimize_schedule", json=sample_payload)
        assert response.status_code == 200

    def test_optimize_returns_run_id(self, test_client, sample_payload):
        data = test_client.post("/optimize_schedule", json=sample_payload).json()
        assert "run_id" in data
        assert isinstance(data["run_id"], str)
        assert len(data["run_id"]) > 0

    def test_optimize_returns_success_status(self, test_client, sample_payload):
        data = test_client.post("/optimize_schedule", json=sample_payload).json()
        assert data["status"] == "success"

    def test_optimize_returns_solver_status(self, test_client, sample_payload):
        data = test_client.post("/optimize_schedule", json=sample_payload).json()
        assert data["solver_status"] in {"optimal", "feasible", "unknown"}

    def test_optimize_returns_schedule(self, test_client, sample_payload):
        data = test_client.post("/optimize_schedule", json=sample_payload).json()
        assert isinstance(data["schedule"], dict)
        assert len(data["schedule"]) == len(sample_payload["engines"])

    def test_optimize_returns_monthly_kpis(self, test_client, sample_payload):
        data = test_client.post("/optimize_schedule", json=sample_payload).json()
        kpis = data["monthly_kpis"]
        assert len(kpis) == sample_payload["horizon_months"]
        assert all("month" in k for k in kpis)

    def test_unsupported_solver_returns_400(self, test_client, sample_payload):
        sample_payload["settings"]["solver"] = "ga"
        response = test_client.post("/optimize_schedule", json=sample_payload)
        assert response.status_code == 400

    def test_invalid_health_returns_422(self, test_client, sample_payload):
        sample_payload["engines"][0]["health"] = 2.0
        response = test_client.post("/optimize_schedule", json=sample_payload)
        assert response.status_code == 422

    def test_extra_field_returns_422(self, test_client, sample_payload):
        sample_payload["not_a_real_field"] = "oops"
        response = test_client.post("/optimize_schedule", json=sample_payload)
        assert response.status_code == 422


class TestRunRetrieval:

    @pytest.fixture
    def run_id(self, test_client, sample_payload):
        data = test_client.post("/optimize_schedule", json=sample_payload).json()
        return data["run_id"]

    def test_returned_run_id_is_retrievable(self, test_client, run_id):
        response = test_client.get(f"/runs/{run_id}")
        assert response.status_code == 200

    def test_retrieved_run_has_correct_id(self, test_client, run_id):
        data = test_client.get(f"/runs/{run_id}").json()
        assert data["run_id"] == run_id

    def test_unknown_run_returns_404(self, test_client):
        response = test_client.get("/runs/does-not-exist")
        assert response.status_code == 404

    def test_list_runs_includes_new_run(self, test_client, run_id):
        runs = test_client.get("/runs").json()
        run_ids = [r["run_id"] for r in runs]
        assert run_id in run_ids


class TestScheduleRetrieval:

    @pytest.fixture
    def run_id(self, test_client, sample_payload):
        data = test_client.post("/optimize_schedule", json=sample_payload).json()
        return data["run_id"]

    def test_schedule_returns_200(self, test_client, run_id):
        assert test_client.get(f"/runs/{run_id}/schedule").status_code == 200

    def test_schedule_has_one_entry_per_engine(self, test_client, run_id, sample_payload):
        entries = test_client.get(f"/runs/{run_id}/schedule").json()
        assert len(entries) == len(sample_payload["engines"])

    def test_schedule_entries_have_correct_run_id(self, test_client, run_id):
        entries = test_client.get(f"/runs/{run_id}/schedule").json()
        assert all(e["run_id"] == run_id for e in entries)

    def test_schedule_unknown_run_returns_404(self, test_client):
        assert test_client.get("/runs/no-such-run/schedule").status_code == 404


class TestSummaryRetrieval:

    @pytest.fixture
    def run_id(self, test_client, sample_payload):
        data = test_client.post("/optimize_schedule", json=sample_payload).json()
        return data["run_id"]

    def test_summary_returns_200(self, test_client, run_id):
        assert test_client.get(f"/runs/{run_id}/summary").status_code == 200

    def test_summary_has_run_id(self, test_client, run_id):
        data = test_client.get(f"/runs/{run_id}/summary").json()
        assert data["summary"]["run_id"] == run_id

    def test_summary_engine_counts_correct(self, test_client, run_id, sample_payload):
        data = test_client.get(f"/runs/{run_id}/summary").json()
        s = data["summary"]
        n = len(sample_payload["engines"])
        assert s["n_engines"] == n
        assert s["n_maintained_engines"] + s["n_unmaintained_engines"] == n

    def test_summary_unknown_run_returns_404(self, test_client):
        assert test_client.get("/runs/no-such-run/summary").status_code == 404
