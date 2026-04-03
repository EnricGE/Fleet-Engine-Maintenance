import os

from sqlalchemy import event
from sqlalchemy.engine import Engine as SAEngine
from sqlmodel import SQLModel, create_engine, Session

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///fleet_engine_planning.db")

engine = create_engine(DATABASE_URL, echo=False)


@event.listens_for(SAEngine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)


def get_session():
    return Session(engine)


