from __future__ import annotations

import json
from pathlib import Path
from typing import List

from fleet_engine_planning.fleet.engine import Engine, Fleet
from fleet_engine_planning.preprocessing.schema import Scenario


def load_json(path: str | Path) -> dict:
    path = Path(path)
    with open(path, "r") as f:
        return json.load(f)


def load_fleet_from_json(path: str | Path) -> Fleet:
    data = load_json(path)

    engines_data = data.get("engines", [])
    engines: List[Engine] = []

    for e in engines_data:
        engine = Engine(
            engine_id=e["engine_id"],
            age_months=float(e["age_months"]),
            distance_km=float(e["distance_km"]),
            health=float(e["health"]),
        )

        # Basic validation
        if not (0.0 <= engine.health <= 1.0):
            raise ValueError(f"Health must be between 0 and 1 for {engine.engine_id}")

        engines.append(engine)

    return Fleet(engines=engines)


def load_scenario(path: str | Path) -> Scenario:
    """
    Load scheduling scenario.
    Scenario JSON must contain:
        - fleet_file
        - horizon_months
        - shop_capacity
        - costs
        - failure_model
    """

    path = Path(path)
    data = load_json(path)

    # Resolve fleet file relative to scenario file
    fleet_path = Path(data["fleet_file"])
    if not fleet_path.is_absolute():
        fleet_path = path.parent / fleet_path

    fleet = load_fleet_from_json(fleet_path)

    horizon = int(data["horizon_months"])
    capacity = [int(x) for x in data["shop_capacity"]]

    if len(capacity) != horizon:
        raise ValueError("shop_capacity length must equal horizon_months")

    return Scenario(
        horizon_months=horizon,
        shop_capacity=capacity,
        costs=data["costs"],
        failure_model=data["failure_model"],
        fleet=fleet,
    )