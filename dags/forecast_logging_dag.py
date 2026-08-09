"""
Forecast Logging DAG (TaskFlow API)
----------------------------------------
Runs once a day: predicts +24h/+48h/+72h for all 4 cities using the
current best registered model, and logs each prediction to Hopsworks'
forecast_log feature group. This is what makes real-world accuracy
checking possible (see analytics/forecast_log.py) -- without this
running daily, there's nothing to compare "what the model predicted"
against "what actually happened" once time passes.
"""

from airflow.decorators import dag, task
from datetime import datetime

from analytics.forecast_log import log_todays_forecasts


@dag(
    dag_id="forecast_logging",
    start_date=datetime(2026, 8, 2),
    schedule="@daily",
    catchup=False,
    is_paused_upon_creation=False,
    tags=["aqi-predictor", "forecast-logging"],
)
def forecast_logging_dag():

    @task
    def log_forecasts():
        log_todays_forecasts()

    log_forecasts()


forecast_logging_dag()