from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# IMPORTANT:
# HbFeatureEngineer is stored inside the trained .pkl pipeline.
# This import must exist before joblib.load().
from hb_features import HbFeatureEngineer


# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="Hemoglobin Estimation API",
    description="Hemoglobin prediction using MAX30102 Red + IR PPG data",
    version="1.0.0"
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# MODEL LOADING
# =========================================================

MODEL_PATH = Path(__file__).parent / "hemoglobin_xgb_pipeline.pkl"

model = None
MODEL_LOADED = False

try:
    model = joblib.load(MODEL_PATH)
    MODEL_LOADED = True
    print("Hemoglobin XGBoost model loaded successfully.")

except Exception as e:
    print("ERROR: Could not load hemoglobin model.")
    print(e)


# =========================================================
# REQUEST DATA MODEL
# =========================================================

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


# =========================================================
# ROOT ENDPOINT
# =========================================================

@app.get("/")
def root():

    return {
        "message": "Hemoglobin Estimation API is running",
        "model_loaded": MODEL_LOADED,
        "docs": "/docs",
        "health": "/health"
    }


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": MODEL_LOADED
    }


# =========================================================
# MODEL INFORMATION
# =========================================================

@app.get("/model-info")
def model_info():

    return {
        "model": "XGBoost Regressor",
        "input_sensor": "MAX30102",
        "prediction_unit": "g/dL",

        "inputs": {
            "red": "MAX30102 Red value",
            "ir": "MAX30102 Infra Red value",
            "age": "Age in years",
            "gender": "1 = Male, 0 = Female"
        },

        "gender_encoding": {
            "Male": 1,
            "Female": 0
        },

        "engineered_features": [
            "IR_Red_Ratio",
            "Red_IR_Ratio",
            "IR_Red_Diff",
            "IR_Red_Sum",
            "Normalized_Difference",
            "Log_IR_Red_Ratio"
        ],

        "output": "Predicted Hemoglobin"
    }


# =========================================================
# HEMOGLOBIN CLASSIFICATION
# =========================================================

def classify_hemoglobin(hb: float, gender: int) -> str:

    # Male
    if gender == 1:

        if hb >= 13:
            return "Normal"

        elif hb >= 11:
            return "Mild Anemia"

        elif hb >= 8:
            return "Moderate Anemia"

        else:
            return "Severe Anemia"

    # Female
    else:

        if hb >= 12:
            return "Normal"

        elif hb >= 11:
            return "Mild Anemia"

        elif hb >= 8:
            return "Moderate Anemia"

        else:
            return "Severe Anemia"


# =========================================================
# PREDICTION ENDPOINT
# =========================================================

@app.post("/predict")
def predict(request: PredictRequest):

    # -----------------------------------------------------
    # Check model
    # -----------------------------------------------------

    if not MODEL_LOADED or model is None:

        raise HTTPException(
            status_code=500,
            detail="Hemoglobin model is not loaded."
        )

    try:

        # -------------------------------------------------
        # Convert gender number to training dataset format
        # -------------------------------------------------

        if request.gender == 1:
            gender_text = "Male"
        else:
            gender_text = "Female"

        # -------------------------------------------------
        # Create input dataframe
        #
        # These column names MUST match the training dataset.
        # -------------------------------------------------

        input_data = pd.DataFrame([
            {
                "Red (a.u)": request.red,
                "Infra Red (a.u)": request.ir,
                "Gender": gender_text,
                "Age (year)": request.age
            }
        ])

        # -------------------------------------------------
        # Make prediction
        #
        # The saved pipeline handles:
        #
        # 1. Gender encoding
        # 2. Feature engineering
        # 3. Missing-value handling
        # 4. XGBoost prediction
        #
        # Do NOT manually calculate the engineered features
        # here.
        # -------------------------------------------------

        prediction = model.predict(input_data)

        hb = float(prediction[0])

        # -------------------------------------------------
        # Classification
        # -------------------------------------------------

        classification = classify_hemoglobin(
            hb,
            request.gender
        )

        # -------------------------------------------------
        # Warning
        # -------------------------------------------------

        warning = None

        if hb < 4 or hb > 20:

            warning = (
                "Predicted hemoglobin is outside the expected "
                "range. Please verify the sensor readings and "
                "consider laboratory confirmation."
            )

        # -------------------------------------------------
        # Return response
        # -------------------------------------------------

        return {
            "predicted_hb_g_dl": round(hb, 2),

            "gender": gender_text,

            "age": request.age,

            "red": request.red,

            "ir": request.ir,

            "anemia_classification": classification,

            "warning": warning
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )