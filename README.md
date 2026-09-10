# 🩸 Aura — Hemoglobin Estimation API

A machine-learning powered REST API for estimating **hemoglobin (Hb)** levels using **MAX30102 Red + Infrared (IR) PPG sensor data**, along with age and gender.

The API is built using **FastAPI** and uses a trained **XGBoost regression pipeline** to estimate hemoglobin levels.

---

## 🚀 Live API

### 🌐 Production API

**Aura Hemoglobin API**

https://aura-haemoglobin-api.onrender.com

### 📚 Interactive API Documentation

Swagger UI:

https://aura-haemoglobin-api.onrender.com/docs

### ❤️ Health Check

https://aura-haemoglobin-api.onrender.com/health

Expected response:

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

---

## ✨ Features

- 🩸 Hemoglobin estimation using Machine Learning
- ❤️ MAX30102 Red + IR PPG sensor support
- 🤖 XGBoost regression model
- ⚡ FastAPI REST API
- 📊 Automatic feature engineering
- 👤 Age and gender based prediction
- 🩺 Basic anemia classification
- ☁️ Deployed on Render
- 📖 Interactive Swagger API documentation
- 🔌 Ready for ESP32 integration
- 🚀 REST API accessible from IoT devices and applications

---

# 🏗️ System Architecture

```text
                  ┌──────────────────┐
                  │    MAX30102      │
                  │   PPG Sensor     │
                  └────────┬─────────┘
                           │
                    Red + IR Values
                           │
                           ▼
                  ┌──────────────────┐
                  │      ESP32       │
                  └────────┬─────────┘
                           │
                      HTTP POST
                           │
                           ▼
              ┌────────────────────────┐
              │    FastAPI Backend     │
              │                        │
              │      /predict          │
              └────────────┬───────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │  Feature Engineering   │
              └────────────┬───────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │   XGBoost Pipeline     │
              └────────────┬───────────┘
                           │
                           ▼
                 Hemoglobin Estimate
                           │
                           ▼
                Anemia Classification
```

---

# 🧠 Machine Learning Model

The backend uses an **XGBoost Regressor** trained on PPG sensor measurements and demographic information.

The model is stored as a serialized pipeline:

```text
hemoglobin_xgb_pipeline.pkl
```

The pipeline includes:

1. Custom feature engineering
2. Missing-value imputation
3. XGBoost regression

---

# 📥 Input Features

The API accepts four inputs:

| Input | Type | Description |
|---|---|---|
| `red` | Float | MAX30102 Red sensor value |
| `ir` | Float | MAX30102 Infrared sensor value |
| `age` | Integer | Age in years |
| `gender` | Integer | `1` = Male, `0` = Female |

Example:

```json
{
  "red": 52341,
  "ir": 61782,
  "age": 24,
  "gender": 1
}
```

---

# ⚙️ Feature Engineering

The API automatically generates additional features from the Red and IR sensor values.

The following features are generated internally:

```text
IR_Red_Ratio
Red_IR_Ratio
IR_Red_Diff
IR_Red_Sum
Normalized_Difference
Log_IR_Red_Ratio
```

The client or ESP32 **does not need to calculate these features manually**.

The trained machine-learning pipeline performs the feature engineering automatically.

---

# 👤 Gender Encoding

Gender is encoded as follows:

```text
Male   → 1
Female → 0
```

For example:

```json
{
  "gender": 1
}
```

represents a male subject.

And:

```json
{
  "gender": 0
}
```

represents a female subject.

---

# 📡 API Endpoints

## `GET /`

Returns basic information about the API.

### Request

```text
GET https://aura-haemoglobin-api.onrender.com/
```

### Example Response

```json
{
  "message": "Hemoglobin Estimation API is running",
  "model_loaded": true,
  "docs": "/docs",
  "health": "/health"
}
```

---

## `GET /health`

Checks whether the API is running and whether the machine-learning model has been successfully loaded.

### Request

```text
GET https://aura-haemoglobin-api.onrender.com/health
```

### Response

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

---

## `GET /model-info`

Returns information about the deployed model, inputs, feature engineering, and output.

### Request

```text
GET https://aura-haemoglobin-api.onrender.com/model-info
```

---

# 🔮 Hemoglobin Prediction

## `POST /predict`

The `/predict` endpoint accepts Red, IR, age, and gender and returns an estimated hemoglobin value.

### Endpoint

```text
POST https://aura-haemoglobin-api.onrender.com/predict
```

### Request Body

```json
{
  "red": 52341,
  "ir": 61782,
  "age": 24,
  "gender": 1
}
```

### Example Response

```json
{
  "predicted_hb_g_dl": 14.36,
  "gender": "Male",
  "age": 24,
  "red": 52341,
  "ir": 61782,
  "anemia_classification": "Normal",
  "warning": null
}
```

> The value above is an example prediction for the example input. Actual predictions depend on the sensor measurements provided to the model.

---

# 💻 Testing with cURL

You can test the production API directly using cURL.

```bash
curl -X POST "https://aura-haemoglobin-api.onrender.com/predict" \
-H "Content-Type: application/json" \
-d '{
  "red": 52341,
  "ir": 61782,
  "age": 24,
  "gender": 1
}'
```

---

# 🐍 Testing with Python

```python
import requests

url = "https://aura-haemoglobin-api.onrender.com/predict"

data = {
    "red": 52341,
    "ir": 61782,
    "age": 24,
    "gender": 1
}

response = requests.post(url, json=data)

print(response.json())
```

---

# 📚 Swagger Documentation

The API provides an interactive Swagger interface.

Open:

```text
https://aura-haemoglobin-api.onrender.com/docs
```

From Swagger UI you can:

- View available endpoints
- View request parameters
- Send test requests
- Test the `/predict` endpoint
- View API responses
- Test the health endpoint

---

# 🩺 Anemia Classification

The API provides a basic anemia classification based on the predicted hemoglobin value and gender.

## Male

| Hemoglobin | Classification |
|---:|---|
| ≥ 13 g/dL | Normal |
| 11–12.99 g/dL | Mild Anemia |
| 8–10.99 g/dL | Moderate Anemia |
| < 8 g/dL | Severe Anemia |

## Female

| Hemoglobin | Classification |
|---:|---|
| ≥ 12 g/dL | Normal |
| 11–11.99 g/dL | Mild Anemia |
| 8–10.99 g/dL | Moderate Anemia |
| < 8 g/dL | Severe Anemia |

> These thresholds are implemented as basic project classification rules and are not intended to provide a clinical diagnosis.

---

# 🔌 MAX30102 + ESP32 Integration

The intended hardware workflow is:

```text
          MAX30102
             │
       ┌─────┴─────┐
       │           │
      Red          IR
       │           │
       └─────┬─────┘
             │
             ▼
           ESP32
             │
             │ HTTP POST
             ▼
    Aura Hemoglobin API
             │
             ▼
      XGBoost Pipeline
             │
             ▼
     Hemoglobin Estimate
```

The ESP32 needs to send:

```text
Red
IR
Age
Gender
```

to the `/predict` endpoint.

Example payload:

```json
{
  "red": 52341,
  "ir": 61782,
  "age": 24,
  "gender": 1
}
```

---

# ⚠️ Sensor Data Compatibility

The MAX30102 readings used during deployment should have a similar **scale and representation** to the values used to train the model.

For example, if the training data contains values such as:

```text
Red ≈ 50,000
IR  ≈ 60,000
```

the deployed MAX30102 should provide readings in a comparable representation.

If the sensor data is normalized, scaled, filtered differently, or otherwise transformed compared with the training data, the model predictions may become unreliable.

The same preprocessing assumptions used during model training should therefore be maintained during deployment.

---

# 🛠️ Local Development

## 1. Clone the repository

```bash
git clone <YOUR-GITHUB-REPOSITORY>
cd <YOUR-GITHUB-REPOSITORY>
```

---

## 2. Create a virtual environment

### Windows

```powershell
python -m venv .venv
```

Activate:

```powershell
.\.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Run the API

```bash
uvicorn main:app --reload
```

The local API will be available at:

```text
http://127.0.0.1:8000
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

Health check:

```text
http://127.0.0.1:8000/health
```

---

# 📁 Project Structure

```text
.
├── main.py
├── hb_features.py
├── hemoglobin_xgb_pipeline.pkl
├── requirements.txt
├── render.yaml
└── .gitignore
```

### `main.py`

The main FastAPI application.

It contains:

- API configuration
- Model loading
- Request validation
- `/predict`
- `/health`
- `/model-info`
- Hemoglobin classification

### `hb_features.py`

Contains the custom `HbFeatureEngineer` transformer used inside the trained machine-learning pipeline.

This file is required when loading the serialized `.pkl` model.

### `hemoglobin_xgb_pipeline.pkl`

The trained XGBoost pipeline used for hemoglobin prediction.

### `requirements.txt`

Contains the Python dependencies required to run the API.

### `render.yaml`

Contains the Render deployment configuration.

### `.gitignore`

Prevents unnecessary local files such as virtual environments and Python cache files from being committed.

---

# 📦 Requirements

The API uses the following major Python packages:

```text
FastAPI
Uvicorn
Scikit-learn
XGBoost
Pandas
NumPy
Joblib
```

---

# ☁️ Render Deployment

The API is deployed using **Render**.

### Build Command

```bash
pip install -r requirements.txt
```

### Start Command

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

### Production URL

```text
https://aura-haemoglobin-api.onrender.com
```

### Render Configuration

The repository contains:

```text
render.yaml
```

with the deployment configuration.

Example:

```yaml
services:
  - type: web
    name: hemoglobin-xgboost-api
    runtime: python
    plan: free

    buildCommand: pip install -r requirements.txt

    startCommand: uvicorn main:app --host 0.0.0.0 --port $PORT

    envVars:
      - key: PYTHON_VERSION
        value: "3.13.4"
```

---

# 🔐 API Input Validation

The API validates incoming requests.

### Red

```text
Must be greater than 0
```

### IR

```text
Must be greater than 0
```

### Age

```text
1–120 years
```

### Gender

```text
0 = Female
1 = Male
```

---

# 🧪 Example Workflow

A typical prediction workflow looks like:

```text
1. Place finger on MAX30102
          ↓
2. MAX30102 captures Red + IR PPG
          ↓
3. ESP32 reads sensor values
          ↓
4. ESP32 sends HTTP POST request
          ↓
5. FastAPI receives the data
          ↓
6. Feature engineering is performed
          ↓
7. XGBoost predicts hemoglobin
          ↓
8. API returns Hb estimate
          ↓
9. Basic anemia classification is returned
```

---

# 🚀 Future Improvements

Possible future improvements include:

- 🔌 Real-time ESP32 integration
- 📱 Mobile application
- 🌐 Web dashboard
- 📈 Real-time PPG visualization
- ❤️ Signal-quality detection
- 🧠 Model improvement with larger datasets
- 🧪 Validation against laboratory hemoglobin measurements
- 🔐 API authentication
- 🗄️ Database integration
- 📊 Prediction history
- 📡 Real-time IoT monitoring
- 🎯 Model confidence/uncertainty estimation
- 🏥 Clinical validation

---

# ⚠️ Medical Disclaimer

This project is intended for **research, educational, and prototype purposes**.

The hemoglobin values generated by this API should **not be considered a replacement for a laboratory blood test or professional medical diagnosis**.

The system requires appropriate validation against laboratory measurements before any clinical application.

If a person has concerns about anemia or their hemoglobin level, they should consult a qualified healthcare professional and use appropriate laboratory testing.

---

# 🌟 Project Overview

**Aura** combines IoT, biomedical sensing, machine learning, and cloud deployment to create a prototype hemoglobin estimation system.

```text
MAX30102
    +
ESP32
    +
FastAPI
    +
XGBoost
    +
Render
    ↓
Hemoglobin Estimation
```

---

# 🌐 Live Links

### 🚀 API

https://aura-haemoglobin-api.onrender.com

### 📚 Swagger Documentation

https://aura-haemoglobin-api.onrender.com/docs

### ❤️ Health Check

https://aura-haemoglobin-api.onrender.com/health

### ℹ️ Model Information

https://aura-haemoglobin-api.onrender.com/model-info

---

## 🩸 Aura

**AI-powered hemoglobin estimation using PPG sensing and machine learning.**

Built with ❤️ using **MAX30102 + ESP32 + FastAPI + XGBoost + Render**.
