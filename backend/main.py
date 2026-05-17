"""
╔══════════════════════════════════════════════════════════════════╗
║          🌧️  RainCast AI — FastAPI Backend                       ║
║          Ensemble Model: Logistic Regression + XGBoost + SVC    ║
╚══════════════════════════════════════════════════════════════════╝
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import numpy as np
import pickle
import time
import os
from datetime import datetime
from typing import Optional, List
import logging

# ─── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("raincast")

# ─── App Init ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="🌧️ RainCast AI API",
    description=(
        "Production-ready Rainfall Prediction API powered by an ensemble of "
        "Logistic Regression, XGBoost, and SVC models. "
        "Predicts whether it will rain based on meteorological parameters."
    ),
    version="1.0.0",
    contact={"name": "RainCast AI Team"},
    license_info={"name": "MIT"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Model Loading ─────────────────────────────────────────────────────────────
MODEL_PATH  = os.getenv("MODEL_PATH",  "model.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "scaler.pkl")

def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        models = pickle.load(f)   # list: [LogReg, XGB, SVC]
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    return models, scaler

try:
    MODELS, SCALER = load_artifacts()
    MODEL_NAMES = ["Logistic Regression", "XGBoost", "SVC"]
    logger.info("✅ Models & scaler loaded successfully.")
except Exception as e:
    logger.error(f"❌ Failed to load model artifacts: {e}")
    MODELS, SCALER = None, None

# ─── Feature order (must match training) ──────────────────────────────────────
FEATURE_ORDER = [
    "pressure", "temperature", "dewpoint",
    "humidity", "cloud", "sunshine",
    "winddirection", "windspeed",
]

# ─── Schemas ───────────────────────────────────────────────────────────────────
class WeatherInput(BaseModel):
    pressure:     float = Field(..., ge=900,  le=1100, description="Atmospheric pressure (hPa)")
    temperature:  float = Field(..., ge=-20,  le=60,   description="Mean temperature (°C)")
    dewpoint:     float = Field(..., ge=-30,  le=40,   description="Dew point temperature (°C)")
    humidity:     float = Field(..., ge=0,    le=100,  description="Relative humidity (%)")
    cloud:        float = Field(..., ge=0,    le=8,    description="Cloud cover (oktas 0–8)")
    sunshine:     float = Field(..., ge=0,    le=24,   description="Sunshine hours per day")
    winddirection:float = Field(..., ge=0,    le=360,  description="Wind direction (degrees)")
    windspeed:    float = Field(..., ge=0,    le=200,  description="Wind speed (km/h)")

    class Config:
        schema_extra = {
            "example": {
                "pressure": 1015.0,
                "temperature": 22.5,
                "dewpoint": 14.0,
                "humidity": 75,
                "cloud": 5,
                "sunshine": 3.2,
                "winddirection": 180,
                "windspeed": 25.0,
            }
        }


class BatchWeatherInput(BaseModel):
    records: List[WeatherInput] = Field(..., max_items=100)


class ModelPrediction(BaseModel):
    model_name:  str
    prediction:  str
    confidence:  float


class PredictionResponse(BaseModel):
    request_id:         str
    timestamp:          str
    final_prediction:   str
    rain_probability:   float
    no_rain_probability:float
    confidence_level:   str
    individual_models:  List[ModelPrediction]
    input_features:     dict
    inference_time_ms:  float


class HealthResponse(BaseModel):
    status:      str
    models_loaded: bool
    model_names: List[str]
    timestamp:   str
    version:     str


class ModelInfoResponse(BaseModel):
    ensemble_size:   int
    models:          List[str]
    feature_names:   List[str]
    scaler_type:     str
    prediction_classes: List[str]

# ─── Helpers ───────────────────────────────────────────────────────────────────
def make_features(data: WeatherInput) -> np.ndarray:
    return np.array([[
        data.pressure,
        data.temperature,
        data.dewpoint,
        data.humidity,
        data.cloud,
        data.sunshine,
        data.winddirection,
        data.windspeed,
    ]])


def ensemble_predict(X_scaled: np.ndarray):
    """Soft-voting ensemble across all 3 models."""
    proba_list = []
    individual = []

    for name, model in zip(MODEL_NAMES, MODELS):
        p = model.predict_proba(X_scaled)[0]   # [P(no_rain), P(rain)]
        proba_list.append(p)
        pred_label = "Rain" if np.argmax(p) == 1 else "No Rain"
        individual.append(ModelPrediction(
            model_name=name,
            prediction=pred_label,
            confidence=round(float(max(p)) * 100, 2),
        ))

    avg_proba = np.mean(proba_list, axis=0)
    rain_prob   = float(avg_proba[1])
    no_rain_prob = float(avg_proba[0])
    final_pred  = "Rain" if rain_prob >= 0.5 else "No Rain"

    if rain_prob >= 0.80 or no_rain_prob >= 0.80:
        conf_level = "High"
    elif rain_prob >= 0.65 or no_rain_prob >= 0.65:
        conf_level = "Moderate"
    else:
        conf_level = "Low"

    return final_pred, rain_prob, no_rain_prob, conf_level, individual

# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "🌧️ Welcome to RainCast AI API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    return HealthResponse(
        status="healthy" if MODELS else "degraded",
        models_loaded=MODELS is not None,
        model_names=MODEL_NAMES if MODELS else [],
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0",
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info():
    if not MODELS:
        raise HTTPException(503, "Models not loaded")
    return ModelInfoResponse(
        ensemble_size=len(MODELS),
        models=MODEL_NAMES,
        feature_names=FEATURE_ORDER,
        scaler_type=type(SCALER).__name__,
        prediction_classes=["No Rain", "Rain"],
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(data: WeatherInput):
    if not MODELS:
        raise HTTPException(503, "Models not loaded. Place model.pkl & scaler.pkl next to main.py.")

    t0 = time.perf_counter()
    X = make_features(data)
    X_scaled = SCALER.transform(X)
    final_pred, rain_prob, no_rain_prob, conf_level, individual = ensemble_predict(X_scaled)
    elapsed = (time.perf_counter() - t0) * 1000

    return PredictionResponse(
        request_id=f"req_{int(time.time()*1000)}",
        timestamp=datetime.utcnow().isoformat(),
        final_prediction=final_pred,
        rain_probability=round(rain_prob * 100, 2),
        no_rain_probability=round(no_rain_prob * 100, 2),
        confidence_level=conf_level,
        individual_models=individual,
        input_features=data.dict(),
        inference_time_ms=round(elapsed, 3),
    )


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(batch: BatchWeatherInput):
    if not MODELS:
        raise HTTPException(503, "Models not loaded.")

    results = []
    for i, record in enumerate(batch.records):
        X = make_features(record)
        X_scaled = SCALER.transform(X)
        final_pred, rain_prob, no_rain_prob, conf_level, individual = ensemble_predict(X_scaled)
        results.append({
            "index": i,
            "final_prediction": final_pred,
            "rain_probability": round(rain_prob * 100, 2),
            "no_rain_probability": round(no_rain_prob * 100, 2),
            "confidence_level": conf_level,
        })

    return {"total": len(results), "predictions": results}


@app.get("/features", tags=["Model"])
async def get_features():
    """Return feature definitions for UI rendering."""
    return {
        "features": [
            {"name": "pressure",     "label": "Atmospheric Pressure", "unit": "hPa",    "min": 900,  "max": 1100, "step": 0.1},
            {"name": "temperature",  "label": "Temperature",          "unit": "°C",     "min": -20,  "max": 60,   "step": 0.1},
            {"name": "dewpoint",     "label": "Dew Point",            "unit": "°C",     "min": -30,  "max": 40,   "step": 0.1},
            {"name": "humidity",     "label": "Humidity",             "unit": "%",      "min": 0,    "max": 100,  "step": 1},
            {"name": "cloud",        "label": "Cloud Cover",          "unit": "oktas",  "min": 0,    "max": 8,    "step": 1},
            {"name": "sunshine",     "label": "Sunshine Hours",       "unit": "hrs",    "min": 0,    "max": 24,   "step": 0.1},
            {"name": "winddirection","label": "Wind Direction",       "unit": "°",      "min": 0,    "max": 360,  "step": 1},
            {"name": "windspeed",    "label": "Wind Speed",           "unit": "km/h",   "min": 0,    "max": 200,  "step": 0.1},
        ]
    }


# ─── Exception Handler ────────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(status_code=500, content={"detail": str(exc)})
