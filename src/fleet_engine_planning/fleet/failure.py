from __future__ import annotations

import math
from dataclasses import dataclass

from .engine import Engine


@dataclass(frozen=True)
class WeibullFailureModel:
    """
    Weibull baseline + proportional hazards depending on engine covariates.

    Survival:
        S(t|x) = exp( - (t/lambda)^k * exp(beta·x) )
    """

    shape_k: float
    scale_lambda_months: float

    # Feature scaling
    age_ref_months: float = 60.0          # ~5 years
    distance_ref_km: float = 600_000.0    # typical high mileage

     # Risk coefficients
    beta_age: float = 0.0
    beta_distance: float = 0.0
    beta_bad_health: float = 0.0          # applied to (1 - health)

    def _linpred(self, e: Engine) -> float:
        age_n = max(0.0, e.age) / self.age_ref_months
        dist_n = max(0.0, e.distance) / self.distance_ref_km
        bad_health = 1.0 - max(0.0, min(1.0, e.health))
        return self.beta_age * age_n + self.beta_distance * dist_n + self.beta_bad_health * bad_health

    def cdf(self, e: Engine, t_months: float) -> float:
        """P(failure by time t_months) for engine e."""
        if t_months <= 0:
            return 0.0

        k = self.shape_k
        lam = self.scale_lambda_months
        lp = self._linpred(e)

        # cumulative hazard H(t|x) = (t/lam)^k * exp(lp)
        H = ((t_months / lam) ** k) * math.exp(lp)
        return 1.0 - math.exp(-H)

    def prob_fail_in_horizon(self, e: Engine, horizon_months: float) -> float:
        """P(failure within [0, horizon_months])."""
        return self.cdf(e, horizon_months)


