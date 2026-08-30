
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