import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import status
import joblib
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from income import db
from income.config import settings

logger = logging.getLogger(__name__)


class Features(BaseModel):
    model_config = {"extra": "forbid"}

    age: int = Field(ge=17, le=90)
    workclass: str | None = None
    education: str
    education_num: int = Field(ge=0)
    marital_status: str
    occupation: str | None = None
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(ge=0)
    capital_loss: int = Field(ge=0)
    hours_per_week: int = Field(ge=0)
    native_country: str | None = None



class FeatureRows(BaseModel):
    rows: Annotated[list[Features], Field(min_length=1, max_length=1000)]


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


@app.middleware("http")
async def save_prediction_errors(request: Request, call_next):
    paths = {"/v1/predict", "/v1/predict/batch"}
    if request.method != "POST" or request.url.path not in paths:
        return await call_next(request)

    t0 = time.perf_counter()
    request_id = str(uuid.uuid4())

    # Validation errors occur before the endpoint receives Features.
    try:
        payload = await request.json()
    except ValueError:
        payload = {
            "raw_body": (await request.body()).decode("utf-8", errors="replace")
        }
    
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("Prediction failed: %s", request_id)
        response = JSONResponse(
            status_code=500,
            content={"detail": "Internal Server Error"},
        )
    # Successful predictions are already saved by the endpoint.
    if response.status_code >= 400:
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        try:
            await run_in_threadpool(
                db.save_prediction,
                request_id=request_id,
                features=payload,
                score=None,
                income_more_50k=None,
                model_version=getattr(app.state, "version", "unknown"),
                latency_ms=latency_ms,
                status_code=response.status_code,
            )
        except Exception:
            logger.exception("Failed to save prediction error: %s", request_id)

    return response


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
    """
        Single prediction
    """
    t0 = time.perf_counter()
    request_id = str(uuid.uuid4())
    payload = x.model_dump()
    frame = pd.DataFrame([payload]).reindex(columns=app.state.meta["features"])

    score = float(app.state.pipeline.predict_proba(frame)[0, 1])

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    income_more_50k = score >= app.state.meta["threshold"]

    bg.add_task(db.save_prediction, request_id, payload, score, income_more_50k, app.state.version, latency_ms,
                status.HTTP_200_OK)

    return Prediction(score=score,
                      income_more_50k=income_more_50k,
                      model_version=app.state.version,
                      request_id=request_id,
                      latency_ms=latency_ms)


@app.post("/v1/predict/batch")
def predict_batch(rows: FeatureRows, bg: BackgroundTasks):
    """
        Batch prediction
    """
    t0 = time.perf_counter()
    payloads = [x.model_dump() for x in rows.rows]
    frame = pd.DataFrame(payloads).reindex(columns=app.state.meta["features"])

    scores = app.state.pipeline.predict_proba(frame)[:, 1]

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    results = []
    for payload, raw_score in zip(payloads, scores):
        request_id = str(uuid.uuid4())
        score = float(raw_score)
        income_more_50k = bool(score >= app.state.meta["threshold"])

        bg.add_task(
            db.save_prediction,
            request_id,
            payload,
            score,
            income_more_50k,
            app.state.version,
            latency_ms,
            status.HTTP_200_OK,
        )

        results.append(
            Prediction(
                score=score,
                income_more_50k=income_more_50k,
                model_version=app.state.version,
                request_id=request_id,
                latency_ms=latency_ms,
            )
        )

    return results
