from __future__ import annotations

import httpx


class FleetAPIClient:
    """Thin synchronous HTTP wrapper around the Fleet Engine Planning API."""

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        self._base = base_url.rstrip("/")

    def _get(self, path: str) -> dict | list:
        r = httpx.get(f"{self._base}{path}", timeout=60)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, payload: dict) -> dict:
        r = httpx.post(f"{self._base}{path}", json=payload, timeout=120)
        r.raise_for_status()
        return r.json()

    def health(self) -> bool:
        try:
            r = httpx.get(f"{self._base}/health", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def optimize(self, payload: dict) -> dict:
        return self._post("/optimize_schedule", payload)

    def list_runs(self) -> list[dict]:
        return self._get("/runs")  # type: ignore[return-value]

    def get_run(self, run_id: str) -> dict:
        return self._get(f"/runs/{run_id}")  # type: ignore[return-value]

    def get_schedule(self, run_id: str) -> list[dict]:
        return self._get(f"/runs/{run_id}/schedule")  # type: ignore[return-value]

    def get_summary(self, run_id: str) -> dict:
        return self._get(f"/runs/{run_id}/summary")  # type: ignore[return-value]
