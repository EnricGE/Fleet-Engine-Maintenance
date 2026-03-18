"""
Unit tests for the Pydantic request/response schemas.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.optimization import OptimizationRequest, OptimizationSettings


class TestOptimizationRequestValidation:

    def test_valid_payload_parses(self, sample_payload):
        req = OptimizationRequest(**sample_payload)
        assert len(req.engines) == 4
        assert req.horizon_months == 12
        assert req.settings.solver == "cpsat"

    def test_default_settings_applied_when_omitted(self, sample_payload):
        del sample_payload["settings"]
        req = OptimizationRequest(**sample_payload)
        assert req.settings.solver == "cpsat"
        assert req.settings.n_scenarios == 30
        assert req.settings.random_seed == 123
        assert req.settings.time_limit_s == 10.0

    def test_shop_capacity_preserved(self, sample_payload):
        req = OptimizationRequest(**sample_payload)
        assert req.shop_capacity == [2] * 12

    def test_health_above_one_rejected(self, sample_payload):
        sample_payload["engines"][0]["health"] = 1.5
        with pytest.raises(ValidationError):
            OptimizationRequest(**sample_payload)

    def test_health_below_zero_rejected(self, sample_payload):
        sample_payload["engines"][0]["health"] = -0.1
        with pytest.raises(ValidationError):
            OptimizationRequest(**sample_payload)

    def test_negative_age_rejected(self, sample_payload):
        sample_payload["engines"][0]["age_months"] = -1
        with pytest.raises(ValidationError):
            OptimizationRequest(**sample_payload)

    def test_negative_distance_rejected(self, sample_payload):
        sample_payload["engines"][0]["distance_km"] = -100
        with pytest.raises(ValidationError):
            OptimizationRequest(**sample_payload)

    def test_extra_fields_rejected(self, sample_payload):
        sample_payload["unexpected_field"] = "should_fail"
        with pytest.raises(ValidationError):
            OptimizationRequest(**sample_payload)

    def test_missing_required_horizon_rejected(self, sample_payload):
        del sample_payload["horizon_months"]
        with pytest.raises(ValidationError):
            OptimizationRequest(**sample_payload)

    def test_missing_engines_rejected(self, sample_payload):
        del sample_payload["engines"]
        with pytest.raises(ValidationError):
            OptimizationRequest(**sample_payload)

    def test_horizon_months_zero_rejected(self, sample_payload):
        sample_payload["horizon_months"] = 0
        with pytest.raises(ValidationError):
            OptimizationRequest(**sample_payload)

    def test_spares_zero_accepted(self, sample_payload):
        sample_payload["spares"] = 0
        req = OptimizationRequest(**sample_payload)
        assert req.spares == 0

    def test_unsupported_solver_rejected(self, sample_payload):
        sample_payload["settings"]["solver"] = "magic_solver"
        with pytest.raises(ValidationError):
            OptimizationRequest(**sample_payload)
