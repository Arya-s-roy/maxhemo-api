# TCS34725 Hemoglobin Estimation API

FastAPI service that serves the XGBoost hemoglobin model trained by
`haemoglo.py` in this repository. Deployed on Render.

## Endpoints

| Method | Path         | Purpose                                            |
|--------|--------------|----------------------------------------------------|
| GET    | `/health`    | Liveness + artifact check                          |
| GET    | `/model-info`| Feature list, importances, scaler statistics       |
| POST   | `/predict`   | Predict hemoglobin from raw TCS34725 readings      |
| GET    | `/docs`      | Interactive Swagger UI (auto-generated)             |

## Predict request

```json
POST /predict
{
  "readings": [
    {"r": 13125, "g": 13825, "b": 8050, "c": 35000},
    {"r": 13200, "g": 13800, "b": 8070, "c": 35100}
  ],
  "patient": {"bmi": 22.5, "age": 24, "sex": 1}
}
```

- `readings`: 1-256 raw TCS34725 samples; the normalized ratios (r/c, g/c, b/c)
  are averaged server-side to damp sensor noise before predicting.
- `sex`: `1` = male, `0` = female (must match the training encoding).

Response:

```json
{
  "predicted_hb_g_dl": 10.83,
  "readings_used": 2,
  "mean_norm": {"Norm_Red": 0.3761, "Norm_Green": 0.3944, "Norm_Blue": 0.2298},
  "anemia_classification": "mild_anemia",
  "warning": null
}
```

`anemia_classification` uses WHO-style cutoffs (normal >= 13 g/dL men,
>= 12 g/dL women; severe < 8 g/dL).

## Run locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload
# API at http://localhost:8000, Swagger UI at http://localhost:8000/docs
```

## Deploy on Render

1. Push this repository to GitHub (the `.pkl` artifacts are tracked).
2. In Render, choose **New -> Blueprint** and point it at the repo
   (the included `render.yaml` configures the service), **or** manually create
   a **Web Service** with:
   - Runtime: Python
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
3. After the first deploy, note the public URL (`https://<service>.onrender.com`).

From an ESP32, POST the JSON body above to `https://<service>.onrender.com/predict`.

## After retraining

Run `python haemoglo.py` - it overwrites `tcs34725_hb_model.pkl` and
`tcs34725_scaler.pkl` in the repo root, which `main.py` loads. Commit and push;
Render auto-deploys.

## Caveats

- The model was trained on 15 (synthetic-looking) rows - see the project
  analysis. Predictions are demo-grade until real paired data is collected.
- `requirements.txt` pins the exact sklearn/xgboost/pandas versions used in
  training; bump them only alongside retraining.
