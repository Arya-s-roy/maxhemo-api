from hb_features import HbFeatureEngineer
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from sklearn.base import BaseEstimator, TransformerMixin


# ============================================================
# CUSTOM FEATURE ENGINEERING
# ============================================================

class HbFeatureEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Gender encoding
        X["Gender"] = X["Gender"].map({
            "Male": 1,
            "Female": 0
        })

        # Engineered features
        X["IR_Red_Ratio"] = (
            X["Infra Red (a.u)"] / X["Red (a.u)"]
        )

        X["Red_IR_Ratio"] = (
            X["Red (a.u)"] / X["Infra Red (a.u)"]
        )

        X["IR_Red_Diff"] = (
            X["Infra Red (a.u)"] - X["Red (a.u)"]
        )

        X["IR_Red_Sum"] = (
            X["Infra Red (a.u)"] + X["Red (a.u)"]
        )

        X["Normalized_Difference"] = (
            (X["Infra Red (a.u)"] - X["Red (a.u)"]) /
            (X["Infra Red (a.u)"] + X["Red (a.u)"])
        )

        X["Log_IR_Red_Ratio"] = np.log(
            X["Infra Red (a.u)"] / X["Red (a.u)"]
        )

        return X


# ============================================================
# LOAD MODEL
# ============================================================

MODEL_PATH = Path(__file__).parent / "hemoglobin_xgb_pipeline.pkl"

try:
    model = joblib.load(MODEL_PATH)
    MODEL_LOADED = True
    MODEL_ERROR = None

except Exception as e:
    model = None
    MODEL_LOADED = False
    MODEL_ERROR = str(e)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="MAX30102 Hemoglobin Estimation API",
    description="Hemoglobin prediction using Red + Infra Red PPG readings, age and gender.",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST MODEL
# ============================================================

class PredictRequest(BaseModel):

    red: float = Field(
        ...,
        gt=0,
        description="MAX30102 Red sensor value"
    )

    ir: float = Field(
        ...,
        gt=0,
        description="MAX30102 Infra Red sensor value"
    )

    age: int = Field(
        ...,
        ge=1,
        le=120,
        description="Age in years"
    )

    gender: int = Field(
        ...,
        ge=0,
        le=1,
        description="1 = Male, 0 = Female"
    )


# ============================================================
# RESPONSE MODEL
# ============================================================

class PredictResponse(BaseModel):

    predicted_hb_g_dl: float
    gender: str
    age: int

    red: float
    ir: float

    anemia_classification: str
    warning: str | None


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "message": "Hemoglobin Estimation API is running",
        "model_loaded": MODEL_LOADED,
        "docs": "/docs",
        "health": "/health"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    if MODEL_LOADED:

        return {
            "status": "healthy",
            "model_loaded": True
        }

    return {
        "status": "error",
        "model_loaded": False,
        "error": MODEL_ERROR
    }


# ============================================================
# MODEL INFORMATION
# ============================================================

@app.get("/model-info")
def model_info():

    return {
        "model_loaded": MODEL_LOADED,
        "model_type": "XGBoost Regressor",
        "sensor": "MAX30102",
        "raw_inputs": [
            "Red (a.u)",
            "Infra Red (a.u)",
            "Gender",
            "Age (year)"
        ],
        "engineered_features": [
            "IR_Red_Ratio",
            "Red_IR_Ratio",
            "IR_Red_Diff",
            "IR_Red_Sum",
            "Normalized_Difference",
            "Log_IR_Red_Ratio"
        ]
    }


# ============================================================
# ANEMIA CLASSIFICATION
# ============================================================

def classify_hemoglobin(hb: float, gender: int) -> str:

    if gender == 1:
        # Male
        if hb >= 13:
            return "Normal"
        elif hb >= 11:
            return "Mild Anemia"
        elif hb >= 8:
            return "Moderate Anemia"
        else:
            return "Severe Anemia"

    else:
        # Female
        if hb >= 12:
            return "Normal"
        elif hb >= 11:
            return "Mild Anemia"
        elif hb >= 8:
            return "Moderate Anemia"
        else:
            return "Severe Anemia"


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):

    if not MODEL_LOADED:

        raise HTTPException(
            status_code=500,
            detail=f"Model could not be loaded: {MODEL_ERROR}"
        )

    # Convert 1/0 to the gender names used during training
    gender_name = "Male" if request.gender == 1 else "Female"

    # Create input using EXACT column names used during training
    input_data = pd.DataFrame([
        {
            "Red (a.u)": request.red,
            "Infra Red (a.u)": request.ir,
            "Gender": gender_name,
            "Age (year)": request.age
        }
    ])

    try:

        # Pipeline performs feature engineering,
        # imputation and XGBoost prediction
        prediction = model.predict(input_data)

        hb = float(prediction[0])

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

    # Classification
    classification = classify_hemoglobin(
        hb,
        request.gender
    )

    # Warning for unusual prediction
    warning = None

    if hb < 4:
        warning = "Very low hemoglobin prediction. Verify sensor reading."

    elif hb > 20:
        warning = "Very high hemoglobin prediction. Verify sensor reading."

    return {
        "predicted_hb_g_dl": round(hb, 2),
        "gender": gender_name,
        "age": request.age,
        "red": request.red,
        "ir": request.ir,
        "anemia_classification": classification,
        "warning": warning
    }


# ============================================================
# RUN LOCALLY
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )