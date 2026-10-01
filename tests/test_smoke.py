import os
from uuid import UUID, uuid4

import pandas as pd
import psycopg
import pytest
from psycopg.rows import dict_row


def test_predict_smoke(client, good_row):
    r = client.post("/v1/predict", json=good_row)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["score"] <= 1.0
    assert isinstance(body["income_more_50k"], bool)
    assert body["latency_ms"] >= 0
    assert body["model_version"]


def test_predict_handles_missing_occupation(client, good_row):
    r = client.post("/v1/predict", json={**good_row, "occupation": None})
    assert r.status_code == 200


def test_predict_identical(client, good_row):
    s1 = client.post("/v1/predict", json=good_row).json()["score"]
    s2 = client.post("/v1/predict", json=good_row).json()["score"]
    assert abs(s1 - s2) < 1e-12


def test_predict_matches_pipeline(client, good_row):
    response = client.post("/v1/predict", json=good_row)
    assert response.status_code == 200

    state = client.app.state
    row = pd.DataFrame([good_row]).reindex(columns=state.meta["features"])
    expected = state.pipeline.predict_proba(row)[0, 1]

    assert response.json()["score"] == pytest.approx(float(expected), abs=1e-12)

DATABASE_URL = os.getenv("DATABASE_URL")
@pytest.mark.skipif(not DATABASE_URL, reason="нужен Postgres: задайте DATABASE_URL")
def test_validation_error_is_saved(client, good_row):
    payload = {
        **good_row,
        "age": -10,
        "workclass": f"test-{uuid4()}",
    }

    with psycopg.connect(DATABASE_URL, row_factory=dict_row) as conn:
        try:
            response = client.post("/v1/predict", json=payload)

            assert response.status_code == 422

            with psycopg.connect(DATABASE_URL) as conn:
                records = conn.execute(
                    "SELECT model_version, score, features->>'education', status_code"
                    "FROM predictions WHERE request_id = %s",
                    (response["request_id"],),
                ).fetchone()

            assert len(records) == 1, (
                f"Ожидалась одна запись ошибки, найдено: {len(records)}"
            )

            record = records[0]

            assert record["status_code"] == 422
            assert record["features"] == payload
            assert record["score"] is None
            assert record["income_more_50k"] is None
            assert record["model_version"] == client.app.state.version
            assert record["latency_ms"] >= 0
            UUID(str(record["request_id"]))

        finally:
            conn.execute(
                "DELETE FROM predictions WHERE request_id = %s",
                (response["request_id"],),
            )
            conn.commit()
