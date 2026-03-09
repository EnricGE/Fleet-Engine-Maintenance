from __future__ import annotations

from fastapi import FastAPI, HTTPException

from app.schemas.optimization import OptimizationRequest, OptimizationResult
from app.services.optimization_service import OptimizationService

app = FastAPI(
    title="Fleet Engine Planning API",
    description="Optimization service for fleet-level engine maintenance scheduling",
    version="0.1.0",
)

service = OptimizationService()

@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}

@app.post("/optimize_schedule", response_model=OptimizationResult)
def optimize_schedule(request: OptimizationRequest) -> OptimizationResult:
    try:
        return service.optimize_schedule(request)
    except NotImplementedError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Optimization failed: {exc}") from exc
