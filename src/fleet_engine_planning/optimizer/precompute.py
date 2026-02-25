from __future__ import annotations

from typing import Dict, Tuple

from fleet_engine_planning.fleet.costs import maintenance_cost
from fleet_engine_planning.fleet.engine import Fleet
from fleet_engine_planning.preprocessing.schema import CostParams
from fleet_engine_planning.simulation.deterioration import (
    health_at_start_of_month,
    operable_under_single_shop,
)


def build_operability_tensor(
    fleet: Fleet,
    dh: Dict[Tuple[str, int, int], float],
    horizon_months: int,
    n_scenarios: int,
    h_min: float,
    shop_duration_months: int,
) -> Dict[Tuple[str, int, int, int], int]:
    """
    Returns a[(engine_id, month, scenario, shop_month)] in {0,1}
      month = 1..T
      scenario = 0..S-1
      shop_month = 0..T
    """
    T = int(horizon_months)
    S = int(n_scenarios)
    a: Dict[Tuple[str, int, int, int], int] = {}

    for e in fleet.engines:
        for s in range(S):
            for m in range(0, T + 1):
                for t in range(1, T + 1):
                    a[(e.engine_id, t, s, m)] = operable_under_single_shop(
                        h0=e.health,
                        dh=dh,
                        engine_id=e.engine_id,
                        month=t,
                        scenario=s,
                        shop_month=m,
                        h_min=h_min,
                        shop_duration=shop_duration_months,
                    )
    return a


def build_expected_shop_costs(
    fleet: Fleet,
    dh: Dict[Tuple[str, int, int], float],
    horizon_months: int,
    n_scenarios: int,
    costs: CostParams,
) -> Dict[Tuple[str, int], float]:
    """
    Expected cost of scheduling a shop visit for engine i in month m (1..T).
    Uses expected health at start of month m to compute maintenance cost.
    """
    T = int(horizon_months)
    S = int(n_scenarios)

    c_shop: Dict[Tuple[str, int], float] = {}

    for e in fleet.engines:
        for m in range(1, T + 1):
            total = 0.0
            for s in range(S):
                h_pre = health_at_start_of_month(e.health, dh, e.engine_id, m, s)
                total += maintenance_cost(costs.base_maint_cost, h_pre, costs.gamma_health_cost)
            c_shop[(e.engine_id, m)] = total / S

    return c_shop