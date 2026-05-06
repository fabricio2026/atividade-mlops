from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app import model
from app.config import API_VERSION, MODEL_ALGORITHM
from app.schemas import CustomerFeatures, HealthResponse, InfoResponse, PredictionResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    model.load_model()
    yield


app = FastAPI(title="Bank Marketing Subscription Predictor", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_loaded=model.is_loaded(),
        version=API_VERSION,
    )


@app.get("/info", response_model=InfoResponse)
def info():
    if not model.is_loaded():
        raise HTTPException(status_code=503, detail="Model not available")
    meta = model.get_metadata()
    return InfoResponse(
        algorithm=MODEL_ALGORITHM,
        features=meta.get("features", []),
        metrics=meta.get("metrics", {}),
        version=API_VERSION,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(data: CustomerFeatures):
    if not model.is_loaded():
        raise HTTPException(status_code=503, detail="Model not available")
    prediction, probability = model.predict(data.model_dump())
    return PredictionResponse(
        prediction=prediction,
        probability=probability,
        label="yes" if prediction == 1 else "no",
    )
