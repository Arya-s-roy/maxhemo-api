"""FastAPI service hosting the TCS34725 hemoglobin estimation model.

Deployed on Render (see render.yaml). Exposes:
  GET  /health     - liveness / artifact check
  GET  /model-info - trained model metadata (features, importances, scaler stats)
  POST /predict    - estimate total hemoglobin (g/dL) from raw TCS34725 readings
"""

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).parent
FEATURES = [
    "Norm_Red",
    "Norm_Green",
    "Norm_Blue",
    "BMI",
    "Subject_Age",
    "Subject_Sex",
]

# Loose plausibility bounds - sensors are noisy, so only reject clearly broken input.
MAX_CHANNEL_RATIO = 1.05  # an RGB channel should never meaningfully exceed the clear channel
HB_PLAUSIBLE_RANGE = (4.0, 20.0)  # g/dL

model = joblib.load(BASE_DIR / "tcs34725_hb_model.pkl")
scaler = joblib.load(BASE_DIR / "tcs34725_scaler.pkl")

# Training distribution of the optical ratios; predictions outside it are extrapolations.
_training = pd.read_csv(BASE_DIR / "tcs34725_hb_data.csv")
_pad = 0.01
TRAINING_ENVELOPE = {
    col: (float(_training[col].min() - _pad), float(_training[col].max() + _pad))
    for col in ("Norm_Red", "Norm_Green", "Norm_Blue")
}

app = FastAPI(
    title="TCS34725 Hemoglobin Estimation API",
    description=(
        "Estimates total hemoglobin (g/dL) from TCS34725 color-sensor readings "
        "and patient biometrics. Send one or more raw readings; they are "
        "averaged before prediction."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Schemas -----------------------------------------------------------------


class Reading(BaseModel):
    r: float = Field(..., ge=0, description="Raw red channel count")
    g: float = Field(..., ge=0, description="Raw green channel count")
    b: float = Field(..., ge=0, description="Raw blue channel count")
    c: float = Field(..., gt=0, description="Raw clear channel count (must be > 0)")


class Patient(BaseModel):
    bmi: float = Field(..., gt=0, le=100)
    age: int = Field(..., ge=1, le=120)
    sex: int = Field(..., ge=0, le=1, description="1 = male, 0 = female")


class PredictRequest(BaseModel):
    readings: list[Reading] = Field(
        ...,
        min_length=1,
        max_length=256,
        description="One or more raw TCS34725 samples (r, g, b, c); averaged server-side",
    )
    patient: Patient


class PredictResponse(BaseModel):
    predicted_hb_g_dl: float
    readings_used: int
    mean_norm: dict[str, float]
    anemia_classification: str
    warning: str | None = None


# --- Helpers -----------------------------------------------------------------


def classify_anemia(hb: float, sex: int) -> str:
    """WHO-style thresholds: normal >= 13 (men) / >= 12 (women)."""
    normal_cutoff = 13.0 if sex == 1 else 12.0
    if hb >= normal_cutoff:
        return "normal"
    if hb >= 11.0:
        return "mild_anemia"
    if hb >= 8.0:
        return "moderate_anemia"
    return "severe_anemia"


# --- Endpoints ---------------------------------------------------------------


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "scaler_features": list(getattr(scaler, "feature_names_in_", [])),
    }


@app.get("/model-info")
def model_info():
    importances = getattr(model, "feature_importances_", None)
    # The XGBRegressor was trained on the scaler's numpy output, so it carries no
    # feature names of its own; the scaler does.
    names = list(getattr(scaler, "feature_names_in_", FEATURES))
    return {
        "model_type": type(model).__name__,
        "features": names,
        "feature_importances": {
            n: float(v) for n, v in zip(names, importances)
        }
        if importances is not None
        else None,
        "scaler_mean": [float(v) for v in scaler.mean_],
        "scaler_scale": [float(v) for v in scaler.scale_],
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # Per-reading normalized ratios (r/c, g/c, b/c), same normalization as training.
    rows = []
    for i, rd in enumerate(req.readings):
        row = {
            "Norm_Red": rd.r / rd.c,
            "Norm_Green": rd.g / rd.c,
            "Norm_Blue": rd.b / rd.c,
        }
        if any(v > MAX_CHANNEL_RATIO for v in row.values()):
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Reading {i}: channel/clear ratio above {MAX_CHANNEL_RATIO} "
                    f"(got r/c={row['Norm_Red']:.3f}, g/c={row['Norm_Green']:.3f}, "
                    f"b/c={row['Norm_Blue']:.3f}). Check sensor placement and gain."
                ),
            )
        rows.append(row)

    # Average the optical ratios across readings to damp sensor noise.
    mean_norm = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}

    features = pd.DataFrame(
        [
            {
                **mean_norm,
                "BMI": req.patient.bmi,
                "Subject_Age": req.patient.age,
                "Subject_Sex": req.patient.sex,
            }
        ]
    )[FEATURES]

    features_scaled = scaler.transform(features)
    hb = float(model.predict(features_scaled)[0])

    warnings = []
    outside = [
        k for k, (lo, hi) in TRAINING_ENVELOPE.items() if not (lo <= mean_norm[k] <= hi)
    ]
    if outside:
        warnings.append(
            f"Mean ratios for {outside} fall outside the training distribution; "
            "prediction is an extrapolation and unreliable."
        )
    if not (HB_PLAUSIBLE_RANGE[0] <= hb <= HB_PLAUSIBLE_RANGE[1]):
        warnings.append(
            f"Predicted Hb {hb:.2f} g/dL is outside the plausible range "
            f"{HB_PLAUSIBLE_RANGE}; treat this reading as invalid."
        )

    return PredictResponse(
        predicted_hb_g_dl=round(hb, 2),
        readings_used=len(rows),
        mean_norm={k: round(v, 4) for k, v in mean_norm.items()},
        anemia_classification=classify_anemia(hb, req.patient.sex),
        warning="; ".join(warnings) or None,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
