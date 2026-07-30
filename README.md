# AQI Predictor - Internship MLOps Project

Step-by-step build. Only Step 1 (etl/extract.py) is implemented so far -- everything
else is an empty placeholder to be filled in on later days.

## Setup
1. Fill in `.env` with your API keys
2. `pip install -r requirements.txt`
3. `python etl/extract.py`

## Roadmap
1. Extract (done)
2. Feature engineering (transform.py)
3. Load to Hopsworks Feature Store (load.py)
4. Backfill via Airflow catchup=True
5. Train models (train.py)
6. Evaluate models (evaluate.py)
7. Model registry (registry.py)
8. Airflow DAGs (feature + training pipelines)
9. Flask prediction API (app.py)
10. Streamlit dashboard (streamlit_app.py)
11. SHAP explainability (explain.py)
12. Alerts (AQI > 300)
