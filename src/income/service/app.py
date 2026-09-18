import time
import uuid
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import joblib

from income import db
from income.config import settings


class Features(BaseModel):
    model_config = {"extra": "forbid"}

    age: int = Field(ge=17, le=90)
    workclass: str | None = None
    education: str
    education_num: int
    marital_status: str
    occupation: str | None = None
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str | None = None


class Prediction(BaseModel):
    score: float
    income_more_50k: bool
    model_version: str
    request_id: str
    latency_ms: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    bundle = joblib.load(settings.model_path)
    app.state.pipeline = bundle["pipeline"]
    app.state.meta = bundle["metadata"]
    app.state.version = bundle["metadata"]["model_version"]

    db.init()
    yield
    app.state.pipeline = None


app = FastAPI(title="Income Service", version="1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_version": getattr(app.state, "version", "unknown")}


@app.get("/ready")
def ready():
    if getattr(app.state, "pipeline", None) is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return {"status": "ready"}


@app.post("/v1/predict")
def predict(x: Features, bg: BackgroundTasks):
    t0 = time.perf_counter()
    request_id = str(uuid.uuid4())
    payload = x.model_dump()
    frame = pd.DataFrame([payload]).reindex(columns=app.state.meta["features"])

    score = float(app.state.pipeline.predict_proba(frame)[0, 1])

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    status_code = 200
    income_more_50k = score >= app.state.meta["threshold_lr"]

    bg.add_task(db.save_prediction, request_id, payload, score, income_more_50k, app.state.version, latency_ms,
                status_code)

    return Prediction(score=score,
                      income_more_50k=income_more_50k,
                      model_version=app.state.version,
                      request_id=request_id,
                      latency_ms=latency_ms)


@app.post("/v1/predict/batch")
def predict_batch(rows: list[Features], bg: BackgroundTasks):
    t0 = time.perf_counter()
    request_id = str(uuid.uuid4())
    payloads = [x.model_dump() for x in rows]
    frame = pd.DataFrame(payloads).reindex(columns=app.state.meta["features"])

    scores = app.state.pipeline.predict_proba(frame)[:, 1]

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    status_code = 200
    income_more_50k = scores >= app.state.meta["threshold_lr"]

    bg.add_task(db.save_prediction, request_id, payloads[0], scores[0], income_more_50k[0], app.state.version,
                latency_ms,
                status_code)

    return Prediction(score=scores[0],
                      income_more_50k=income_more_50k[0],
                      model_version=app.state.version,
                      request_id=request_id,
                      latency_ms=latency_ms)
