"""
Step 8 - Training Pipeline DAG (TaskFlow API)
--------------------------------------------------
Runs the full training pipeline once a day: fetch features from Hopsworks,
train Random Forest + Ridge Regression (multi-horizon: +24h/+48h/+72h),
evaluate both, register the winner in the Model Registry.

This is ONE task, not split into fetch/train/evaluate/register tasks like
the feature pipeline DAG -- a fitted model is a Python object, and Airflow's
XCom (used to pass data between TaskFlow tasks) is meant for small
JSON-like data, not model binaries. run_training_pipeline() runs the whole
thing in a single process instead.
"""

from airflow.decorators import dag, task
from datetime import datetime

from training.registry import run_training_pipeline


@dag(
    dag_id="training_pipeline",
    start_date=datetime(2026, 8, 2),
    schedule="@daily",
    catchup=False,
    is_paused_upon_creation=False,
    tags=["aqi-predictor", "training-pipeline"],
)
def training_pipeline_dag():

    @task
    def train_and_register():
        run_training_pipeline()

    train_and_register()


training_pipeline_dag()