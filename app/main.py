from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.db.models import OptimizationRun

from app.schemas.optimization import OptimizationRequest, OptimizationResult
from app.schemas.run_analysis import RunAnalysisResult
from app.schemas.run_storage import StoredRunOut, ScheduleEntryOut, MonthlyKPIOut

from app.services.optimization_service import OptimizationService
from app.analyzers.run_analyzer import RunAnalyzer
from app.repositories.run_repository import RunRepository

from app.db.database import create_db_and_tables, get_session

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    create_db_and_tables()
    logger.info("Database ready")
    yield
    # Shutdown (nothing yet)


app = FastAPI(
    title="Fleet Engine Planning API",
    description="Optimization service for fleet-level engine maintenance scheduling",
    version="0.1.0",
    lifespan=lifespan,
)

repo = RunRepository()
service = OptimizationService(repo=repo)
analyzer = RunAnalyzer()

@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/optimize_schedule", response_model=OptimizationResult)
def optimize_schedule(request: OptimizationRequest) -> OptimizationResult:
    logger.info(
        "Optimization request received — engines=%d horizon=%d solver=%s scenarios=%d",
        len(request.engines),
        request.horizon_months,
        request.settings.solver,
        request.settings.n_scenarios,
    )
    t0 = time.perf_counter()
    try:
        result = service.optimize_schedule(request)
        elapsed = time.perf_counter() - t0
        logger.info(
            "Optimization complete — run_id=%s status=%s objective=%.2f elapsed=%.2fs",
            result.run_id,
            result.status,
            result.objective,
            elapsed,
        )
        return result
    except NotImplementedError as exc:
        logger.warning("Unsupported solver requested: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        logger.exception("Optimization failed after %.2fs: %s", elapsed, exc)
        raise HTTPException(status_code=500, detail="Optimization failed") from exc


@app.get("/runs", response_model=list[StoredRunOut])
def list_runs():
    with get_session() as session:
        return repo.list_runs(session)


@app.get("/runs/{run_id}")
def get_run(run_id: str):
    with get_session() as session:
        run = session.get(OptimizationRun, run_id)

        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")

        return run


@app.get("/runs/{run_id}/schedule", response_model=list[ScheduleEntryOut])
def get_run_schedule(run_id: str):
    with get_session() as session:
        run = repo.get_run(session, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")

        return repo.get_schedule(session, run_id)


@app.get("/runs/{run_id}/kpis", response_model=list[MonthlyKPIOut])
def get_run_kpis(run_id: str):
    with get_session() as session:
        run = repo.get_run(session, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return repo.get_monthly_kpis(session, run_id)


@app.get("/runs/{run_id}/summary", response_model=RunAnalysisResult)
def get_run_summary(run_id: str):
    with get_session() as session:

        result = repo.get_run_full(session, run_id)

        if result is None:
            raise HTTPException(status_code=404, detail="Run not found")

        run, schedule_rows, monthly_kpi_rows = result

        analysis = analyzer.analyze_stored_run(
            run,
            schedule_rows,
            monthly_kpi_rows,
            max_rentals_per_month=run.max_rentals_per_month,
        )

        return analysis
