from __future__ import annotations

from uuid import uuid4

from app.schemas.optimization import (
    EngineStateIn,
    MonthlyKPI,
    OptimizationRequest,
    OptimizationResult,
)

from fleet_engine_planning.domain.engine import Engine, Fleet
from fleet_engine_planning.preprocessing.schema import (
    Scenario,
    CostParams,
    DeteriorationParams,
)
from fleet_engine_planning.simulation.deterioration import sample_deterioration_deltas
from fleet_engine_planning.optimization.precompute import (
    build_operability_tensor,
    build_expected_shop_costs,
)
from fleet_engine_planning.solvers.cpsat_schedule import solve_cpsat_schedule_with_rentals

from app.db.database import get_session
from app.db.models import OptimizationRun
from app.repositories.run_repository import RunRepository


class OptimizationService:
    """
    Application service that bridges external request schemas
    and the internal optimization core.
    """

    def __init__(self):
        self.repo = RunRepository()

    def optimize_schedule(self, request: OptimizationRequest) -> OptimizationResult:
        if request.settings.solver != "cpsat":
            raise NotImplementedError(
                f"Solver '{request.settings.solver}' is not implemented yet in OptimizationService."
            )

        scenario = self._build_scenario_from_request(request)

        T = scenario.horizon_months
        S = request.settings.n_scenarios
        seed = request.settings.random_seed

        # Build uncertainty scenarios
        dh = sample_deterioration_deltas(
            fleet=scenario.fleet,
            horizon_months=T + 1,  # +1 for terminal operability
            n_scenarios=S,
            params=scenario.deterioration,
            seed=seed,
        )

        # Precompute operability tensor and expected shop costs
        oper = build_operability_tensor(
            fleet=scenario.fleet,
            dh=dh,
            horizon_months=T,
            n_scenarios=S,
            h_min=scenario.h_min,
            shop_duration_months=scenario.shop_duration_months,
        )

        c_shop = build_expected_shop_costs(
            fleet=scenario.fleet,
            dh=dh,
            horizon_months=T,
            n_scenarios=S,
            costs=scenario.costs,
        )

        n_required = max(0, len(scenario.fleet.engines) - scenario.spares)

        result = solve_cpsat_schedule_with_rentals(
            fleet=scenario.fleet,
            horizon_months=T,
            shop_capacity=scenario.shop_capacity,
            shop_duration_months=scenario.shop_duration_months,
            max_rentals_per_month=scenario.max_rentals_per_month,
            n_required=n_required,
            n_scenarios=S,
            costs=scenario.costs,
            operable=oper,
            expected_shop_cost=c_shop,
            time_limit_s=request.settings.time_limit_s,
        )

        run_id = str(uuid4())

        if result is None:
            return OptimizationResult(
                run_id=run_id,
                solver=request.settings.solver,
                objective=float("nan"),
                schedule={},
                monthly_kpis=[],
                status="no_solution",
            )

        monthly_kpis = self._build_monthly_kpis(
            rentals=result.rentals,
            downtime=result.downtime,
            horizon_months=T,
            n_scenarios=S,
        )

        with get_session() as session:

            run = OptimizationRun(
                run_id=run_id,
                solver=request.settings.solver,
                objective=result.objective,
                status="success",
                horizon_months=request.horizon_months,
                n_engines=len(request.engines),
            )

            self.repo.save_run(session, run)
            self.repo.save_schedule(session, run_id, result.schedule)

            session.commit()

        return OptimizationResult(
            run_id=str(uuid4()),
            solver=request.settings.solver,
            objective=result.objective,
            schedule=result.schedule,
            monthly_kpis=monthly_kpis,
            status="success",
        )

    def _build_scenario_from_request(self, request: OptimizationRequest) -> Scenario:
        fleet = Fleet(
            engines=[self._build_engine(engine_in) for engine_in in request.engines]
        )

        costs = CostParams(
            base_maint_cost=request.base_maint_cost,
            rental_cost=request.rental_cost,
            downtime_cost=request.downtime_cost,
            gamma_health_cost=request.gamma_health_cost,
            terminal_shortfall_cost=request.terminal_shortfall_cost,
            terminal_inop_cost=request.terminal_inop_cost,
        )

        deterioration = DeteriorationParams(
            km_per_month=request.km_per_month,
            mu_base=request.mu_base,
            mu_per_1000km=request.mu_per_1000km,
            sigma=request.sigma,
        )

        return Scenario(
            horizon_months=request.horizon_months,
            shop_capacity=request.shop_capacity,
            shop_duration_months=request.shop_duration_months,
            spares=request.spares,
            h_min=request.h_min,
            max_rentals_per_month=request.max_rentals_per_month,
            costs=costs,
            deterioration=deterioration,
            fleet=fleet,
            window_length=request.window_length,
            commit_length=request.commit_length,
        )

    @staticmethod
    def _build_engine(engine_in: EngineStateIn) -> Engine:
        return Engine(
            engine_id=engine_in.engine_id,
            age_months=engine_in.age_months,
            distance_km=engine_in.distance_km,
            health=engine_in.health,
        )

    @staticmethod
    def _build_monthly_kpis(
        rentals: dict[tuple[int, int], int],
        downtime: dict[tuple[int, int], int],
        horizon_months: int,
        n_scenarios: int,
    ) -> list[MonthlyKPI]:
        kpis: list[MonthlyKPI] = []

        for month in range(1, horizon_months + 1):
            avg_r = sum(rentals[(month, s)] for s in range(n_scenarios)) / n_scenarios
            avg_d = sum(downtime[(month, s)] for s in range(n_scenarios)) / n_scenarios
            worst_d = max(downtime[(month, s)] for s in range(n_scenarios))

            kpis.append(
                MonthlyKPI(
                    month=month,
                    expected_rentals=avg_r,
                    expected_downtime=avg_d,
                    worst_case_downtime=worst_d,
                )
            )

        return kpis