import psycopg
from psycopg.types.json import Json

from income.config import settings

DDL = """
CREATE TABLE IF NOT EXISTS predictions (
    request_id UUID PRIMARY KEY,
    ts TIMESTAMPTZ NOT NULL DEFAULT now(),
    model_version TEXT NOT NULL,
    features JSONB NOT NULL,
    score DOUBLE PRECISION NOT NULL,
    latency_ms REAL,
    income_more_50k BOOLEAN NOT NULL,
    status_code INTEGER NOT NULL
)
"""


def init() -> None:
    if not settings.database_url:
        return
    with psycopg.connect(settings.database_url) as conn:
        conn.execute(DDL)


def save_prediction(request_id: str, features: dict, score: float, income_more_50k: bool, model_version: str,
                    latency_ms: float, status_code: int) -> None:
    if not settings.database_url:
        return
    with psycopg.connect(settings.database_url) as conn:
        conn.execute(
            "INSERT INTO predictions (request_id, model_version, features, score, latency_ms, income_more_50k, status_code)"
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (request_id, model_version, Json(features), score, latency_ms, income_more_50k, status_code),
        )
