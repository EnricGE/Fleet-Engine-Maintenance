from sqlmodel import Session, select

from app.db.models import OptimizationRun, ScheduleEntry, MonthlyKPIRecord
from app.schemas.optimization import MonthlyKPI


class RunRepository:

    def save_run(self, session: Session, run: OptimizationRun):
        session.add(run)

    def save_schedule(self, session: Session, run_id: str, schedule: dict[str, int]):
        for engine_id, month in schedule.items():
            entry = ScheduleEntry(
                run_id=run_id,
                engine_id=engine_id,
                shop_month=month,
            )
            session.add(entry)

    def save_monthly_kpis(self, session: Session, run_id: str, monthly_kpis: list[MonthlyKPI]):
        for kpi in monthly_kpis:
            row = MonthlyKPIRecord(
                run_id=run_id,
                month=kpi.month,
                expected_rentals=kpi.expected_rentals,
                expected_downtime=kpi.expected_downtime,
                worst_case_downtime=kpi.worst_case_downtime,
            )
            session.add(row)

    def get_run(self, session: Session, run_id: str):
        return session.get(OptimizationRun, run_id)

    def get_schedule(self, session: Session, run_id: str):
        statement = select(ScheduleEntry).where(ScheduleEntry.run_id == run_id)
        return session.exec(statement).all()
    
    def get_monthly_kpis(self, session: Session, run_id: str):
        statement = (
            select(MonthlyKPIRecord)
            .where(MonthlyKPIRecord.run_id == run_id)
            .order_by(MonthlyKPIRecord.month)
        )
        return session.exec(statement).all()
    
    def list_runs(self, session: Session):
        statement = select(OptimizationRun).order_by(OptimizationRun.created_at.desc())
        return session.exec(statement).all()
    
    def get_run_full(self, session: Session, run_id: str):
        run = session.get(OptimizationRun, run_id)

        if run is None:
            return None

        schedule = self.get_schedule(session, run_id)
        monthly_kpis = self.get_monthly_kpis(session, run_id)

        return run, schedule, monthly_kpis