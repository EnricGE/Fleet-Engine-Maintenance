from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from sqlmodel import Session

from app.db.database import engine
from app.db.models import OptimizationRun

from app.schemas.optimization import OptimizationRequest, OptimizationResult
from app.services.optimization_service import OptimizationService
from app.repositories.run_repository import RunRepository
from app.schemas.run_storage import StoredRunOut, ScheduleEntryOut
from app.db.database import create_db_and_tables


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    create_db_and_tables()
    yield
    # Shutdown (nothing yet)


app = FastAPI(
    title="Fleet Engine Planning API",
    description="Optimization service for fleet-level engine maintenance scheduling",
    version="0.1.0",
    lifespan=lifespan,
)

service = OptimizationService()
repo = RunRepository()

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


@app.get("/runs", response_model=list[StoredRunOut])
def list_runs():
    with Session(engine) as session:
        return repo.list_runs(session)
    

@app.get("/runs/{run_id}")
def get_run(run_id: str):

    with Session(engine) as session:
        run = session.get(OptimizationRun, run_id)

        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")

        return run


@app.get("/runs/{run_id}/schedule", response_model=list[ScheduleEntryOut])
def get_run_schedule(run_id: str):
    with Session(engine) as session:
        run = repo.get_run(session, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")

        return repo.get_schedule(session, run_id)
    