import pandas as pd
import pytest


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
