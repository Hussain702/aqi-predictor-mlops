"""
Step 7 - Model Registry
--------------------------
Registers the winning multi-horizon model from evaluate.py in the
Hopsworks Model Registry (versioned: v1, v2, v3... so you can roll back
if a future retrain performs worse).

Also exposes run_training_pipeline() -- the full daily pipeline
(fetch -> train -> evaluate -> register) as ONE function. This is what
the Airflow training_pipeline_dag.py calls as a single task: a fitted
model is a Python object, not something you can pass between separate
Airflow tasks via XCom (XCom is for small JSON-like data, not model
binaries) -- so the whole thing runs in one process, one task.

Run standalone (does the full daily pipeline once, right now):
    python -m training.registry
"""

import os
import shutil
import time

import joblib

from utils.hopsworks_client import get_project

MODEL_NAME = "aqi_predictor_model"
MODEL_DIR = "model_artifact"
MAX_UPLOAD_RETRIES = 4


def flatten_metrics(nested_metrics: dict) -> dict:
    """
    Hopsworks expects a flat {name: number} metrics dict, but our evaluate.py
    returns nested {horizon: {rmse, mae, r2}}. Flatten to e.g.
    {"rmse_24h": .., "mae_24h": .., "rmse_48h": .., ...} so Hopsworks can
    store it (and so get_best_model() can later query by e.g. "rmse_24h").
    """
    flat = {}
    for horizon, metrics in nested_metrics.items():
        for metric_name, value in metrics.items():
            flat[f"{metric_name}_{horizon}"] = value
    return flat


def save_model_locally(model, feature_names: list, model_name: str) -> str:
    """
    Dump the model + its feature list to a local folder Hopsworks can upload.

    Branches by framework: sklearn models (Ridge, Random Forest) pickle
    fine with joblib; the Keras neural network doesn't -- it needs
    TensorFlow's own model.save() format instead. A framework.txt marker
    file tells the dashboard (and any future loader) which one it's
    looking at without needing to guess from file extensions.

    model_name.txt records the SPECIFIC winning algorithm (e.g.
    "random_forest") -- framework alone can't distinguish Ridge from
    Random Forest, since both are "sklearn".
    """
    if os.path.exists(MODEL_DIR):
        shutil.rmtree(MODEL_DIR)
    os.makedirs(MODEL_DIR)

    framework = getattr(model, "framework", "sklearn")
    with open(os.path.join(MODEL_DIR, "framework.txt"), "w") as f:
        f.write(framework)
    with open(os.path.join(MODEL_DIR, "model_name.txt"), "w") as f:
        f.write(model_name)

    if framework == "tensorflow":
        model.model.save(os.path.join(MODEL_DIR, "keras_model.keras"))
    else:
        joblib.dump(model, os.path.join(MODEL_DIR, "model.pkl"))

    with open(os.path.join(MODEL_DIR, "feature_names.txt"), "w") as f:
        f.write("\n".join(feature_names))

    return MODEL_DIR


def register_model(model, feature_names: list, flat_metrics: dict, project, model_name: str):
    """
    Upload the model folder to the Hopsworks Model Registry.

    Wrapped in a retry loop with backoff: on a slow/unstable connection,
    a multi-minute upload can get dropped partway through (ConnectionReset,
    RemoteDisconnected). create_model() only registers metadata client-side
    -- the actual files aren't persisted until .save() completes -- so it's
    safe to just retry .save() again on the same model object.
    """
    model_registry = project.get_model_registry()
    model_dir = save_model_locally(model, feature_names, model_name)
    framework = getattr(model, "framework", "sklearn")

    description = (
        "AQI multi-horizon forecast model (predicts +24h/+48h/+72h AQI; "
        f"best of Ridge Regression / Random Forest / Neural Network -- winner: {model_name}"
    )

    if framework == "tensorflow":
        aqi_model = model_registry.tensorflow.create_model(
            name=MODEL_NAME, metrics=flat_metrics, description=description,
        )
    else:
        aqi_model = model_registry.sklearn.create_model(
            name=MODEL_NAME, metrics=flat_metrics, description=description,
        )

    for attempt in range(1, MAX_UPLOAD_RETRIES + 1):
        try:
            aqi_model.save(model_dir)
            break
        except Exception as e:
            print(f"Upload attempt {attempt}/{MAX_UPLOAD_RETRIES} failed: {e}")
            if attempt == MAX_UPLOAD_RETRIES:
                raise
            wait_seconds = 15 * attempt
            print(f"Retrying in {wait_seconds}s...")
            time.sleep(wait_seconds)

    print(f"Registered '{MODEL_NAME}' version {aqi_model.version} ({model_name}, {framework}) in the Model Registry.")
    return aqi_model


def run_training_pipeline():
    """Full pipeline: fetch features -> train -> evaluate -> register the best."""
    from sklearn.model_selection import train_test_split
    from training.train import load_training_data, prepare_features, train_all
    from training.evaluate import evaluate_all, pick_best_model

    df = load_training_data()
    print(f"Loaded {len(df)} historical rows from Hopsworks.")

    X, Y = prepare_features(df)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    fitted_models = train_all(X_train, Y_train)
    results = evaluate_all(fitted_models, X_test, Y_test)
    best_name = pick_best_model(results)

    best_model = fitted_models[best_name]
    flat_metrics = flatten_metrics(results[best_name])

    project = get_project()
    register_model(best_model, X.columns.tolist(), flat_metrics, project, best_name)


if __name__ == "__main__":
    run_training_pipeline()