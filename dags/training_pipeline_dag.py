
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