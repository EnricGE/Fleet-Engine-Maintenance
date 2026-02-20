from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List

from .engine import Engine, Fleet
from .costs import maintenance_cost
from .failure import WeibullFailureModel


@dataclass(frozen=True)
class PolicyParams:
    base_maint_cost: float
    failure_penalty: float
    gamma_health_cost: float = 1.0


def expected_cost_one_engine(
    engine: Engine,
    model: WeibullFailureModel,
    horizon_months: float,
    do_preventive: bool,
    params: PolicyParams,
) -> float:
    """
    Simple one-step decision:
    - If preventive: pay maintenance now, assume no failure cost in horizon
    - If corrective (CBM-style wait): expected failure penalty within horizon
    """
    if do_preventive:
        return maintenance_cost(params.base_maint_cost, engine.health)

    p_fail = model.prob_fail_in_horizon(engine, horizon_months)
    return p_fail * params.failure_penalty


def expected_cost_fleet(
    fleet: Fleet,
    model: WeibullFailureModel,
    horizon_months: float,
    preventive_decisions: Dict[str, bool],
    params: PolicyParams,
) -> float:
    """
    preventive_decisions maps engine_id -> True/False
    """
    total = 0.0
    for e in fleet.engines:
        do_prev = preventive_decisions.get(e.engine_id, False)
        total += expected_cost_one_engine(e, model, horizon_months, do_prev, params)
    return total


def greedy_policy_by_threshold(
    fleet: Fleet,
    health_threshold: float,
) -> Dict[str, bool]:
    """
    Simple starter policy:
    - preventive if health < threshold
    """
    thr = max(0.0, min(1.0, health_threshold))
    return {e.engine_id: (e.health < thr) for e in fleet.engines}