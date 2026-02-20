from __future__ import annotations

import json
from pathlib import Path
from typing import List

from fleet_engine_planning.fleet.engine import Engine, Fleet


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