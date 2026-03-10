from sqlmodel import Session

from app.db.models import OptimizationRun, ScheduleEntry


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

    def get_run(self, session: Session, run_id: str):
        return session.get(OptimizationRun, run_id)

    def get_schedule(self, session: Session, run_id: str):
        return session.exec(
            ScheduleEntry.select().where(ScheduleEntry.run_id == run_id)
        ).all()