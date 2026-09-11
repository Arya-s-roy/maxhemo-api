from pathlib import Path
from datetime import datetime, timezone

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# IMPORTANT:
# Required so joblib can load HbFeatureEngineer
from hb_features import HbFeatureEngineer


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Hemoglobin Estimation API",
    description="Hemoglobin prediction using MAX30102 Red + IR PPG data",
    version="2.0.0"
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
# MODEL
# ============================================================

MODEL_PATH = (
    Path(__file__).parent /
    "hemoglobin_xgb_pipeline.pkl"
)

model = None
MODEL_LOADED = False


try:

    model = joblib.load(MODEL_PATH)

    MODEL_LOADED = True

    print(
        "Hemoglobin XGBoost model loaded successfully."
    )

except Exception as e:

    print(
        "ERROR: Could not load hemoglobin model."
    )

    print(e)


# ============================================================
# TEMPORARY SENSOR STORAGE
# ============================================================
#
# For hackathon/demo use.
#
# ESP32 sends:
#
# {
#     "red": 52341,
#     "ir": 61782
# }
#
# Render stores the latest reading here.
#
# IMPORTANT:
# This is in-memory storage.
# It can reset if Render restarts.
#
# For a production system, use a database.
# ============================================================

latest_sensor_data = {
    "red": None,
    "ir": None,
    "timestamp": None,
    "device": None
}


# ============================================================
# REQUEST MODELS
# ============================================================

class SensorRequest(BaseModel):

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

    device: str = Field(
        default="ESP32",
        description="Sensor device name"
    )


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
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "message": "Hemoglobin Estimation API is running",

        "version": "2.0.0",

        "model_loaded": MODEL_LOADED,

        "docs": "/docs",

        "health": "/health",

        "endpoints": {
            "send_sensor": "POST /sensor",
            "latest_sensor": "GET /sensor/latest",
            "predict": "POST /predict"
        }
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": MODEL_LOADED
    }


# ============================================================
# MODEL INFO
# ============================================================

@app.get("/model-info")
def model_info():

    return {

        "model": "XGBoost Regressor",

        "input_sensor": "MAX30102",

        "prediction_unit": "g/dL",

        "inputs": {

            "red":
                "MAX30102 Red value",

            "ir":
                "MAX30102 Infra Red value",

            "age":
                "Age in years",

            "gender":
                "1 = Male, 0 = Female"
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

        "output":
            "Predicted Hemoglobin"
    }


# ============================================================
# RECEIVE SENSOR DATA FROM ESP32
# ============================================================

@app.post("/sensor")
def receive_sensor(data: SensorRequest):

    global latest_sensor_data

    latest_sensor_data = {

        "red": data.red,

        "ir": data.ir,

        "timestamp":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "device":
            data.device
    }

    print(
        f"Sensor received | "
        f"RED={data.red} | "
        f"IR={data.ir}"
    )

    return {

        "success": True,

        "message":
            "Sensor data received",

        "red":
            data.red,

        "ir":
            data.ir,

        "timestamp":
            latest_sensor_data["timestamp"],

        "device":
            data.device
    }


# ============================================================
# GET LATEST SENSOR DATA
# ============================================================

@app.get("/sensor/latest")
def get_latest_sensor():

    if (
        latest_sensor_data["red"] is None
        or
        latest_sensor_data["ir"] is None
    ):

        return {

            "success": False,

            "message":
                "No sensor data received yet",

            "sensor_ready":
                False
        }

    return {

        "success": True,

        "sensor_ready": True,

        "red":
            latest_sensor_data["red"],

        "ir":
            latest_sensor_data["ir"],

        "timestamp":
            latest_sensor_data["timestamp"],

        "device":
            latest_sensor_data["device"]
    }


# ============================================================
# HEMOGLOBIN CLASSIFICATION
# ============================================================

def classify_hemoglobin(
    hb: float,
    gender: int
) -> str:

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
# PREDICT
# ============================================================

@app.post("/predict")
def predict(request: PredictRequest):

    if not MODEL_LOADED or model is None:

        raise HTTPException(

            status_code=500,

            detail=
                "Hemoglobin model is not loaded."
        )

    try:

        # ----------------------------------------------------
        # Convert gender
        # ----------------------------------------------------

        if request.gender == 1:

            gender_text = "Male"

        else:

            gender_text = "Female"


        # ----------------------------------------------------
        # Create DataFrame
        # ----------------------------------------------------

        input_data = pd.DataFrame([

            {

                "Red (a.u)":
                    request.red,

                "Infra Red (a.u)":
                    request.ir,

                "Gender":
                    gender_text,

                "Age (year)":
                    request.age
            }

        ])


        # ----------------------------------------------------
        # Model prediction
        # ----------------------------------------------------

        prediction = model.predict(
            input_data
        )

        hb = float(
            prediction[0]
        )


        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        classification = classify_hemoglobin(

            hb,

            request.gender
        )


        # ----------------------------------------------------
        # Warning
        # ----------------------------------------------------

        warning = None

        if hb < 4 or hb > 20:

            warning = (

                "Predicted hemoglobin is outside "
                "the expected range. Please verify "
                "the sensor readings and consider "
                "laboratory confirmation."
            )


        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        return {

            "success": True,

            "predicted_hb_g_dl":
                round(hb, 2),

            "gender":
                gender_text,

            "age":
                request.age,

            "red":
                request.red,

            "ir":
                request.ir,

            "anemia_classification":
                classification,

            "warning":
                warning
        }


    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=
                f"Prediction failed: {str(e)}"
        )