"""
Step 6 - Evaluate trained models (per horizon)
--------------------------------------------------
Scores every fitted model on the held-out test set using RMSE, MAE, R2 --
SEPARATELY for each forecast horizon (24h/48h/72h), since accuracy
typically drops the further out you forecast. Picks the best model by
average RMSE across all 3 horizons.

Run standalone (loads data, trains, evaluates, prints a winner):
    python -m training.evaluate
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from training.train import TARGET_HORIZONS


def evaluate_model(model, X_test, Y_test) -> dict:
    """Returns {horizon: {rmse, mae, r2}} for one fitted model."""
    y_pred = np.asarray(model.predict(X_test))

    metrics = {}
    for i, horizon in enumerate(TARGET_HORIZONS):
        y_true_h = Y_test.iloc[:, i]
        y_pred_h = y_pred[:, i]

        metrics[horizon] = {
            "rmse": float(np.sqrt(mean_squared_error(y_true_h, y_pred_h))),
            "mae": float(mean_absolute_error(y_true_h, y_pred_h)),
            "r2": float(r2_score(y_true_h, y_pred_h)),
        }
    return metrics


def evaluate_all(fitted_models: dict, X_test, Y_test) -> dict:
    """Returns {model_name: {horizon: {rmse, mae, r2}}}, printing as it goes."""
    results = {}
    for name, model in fitted_models.items():
        metrics = evaluate_model(model, X_test, Y_test)
        results[name] = metrics
        print(f"\n{name}:")
        for horizon, m in metrics.items():
            print(f"  {horizon}: RMSE={m['rmse']:.2f}  MAE={m['mae']:.2f}  R2={m['r2']:.3f}")
    return results


def pick_best_model(results: dict) -> str:
    """Lower AVERAGE RMSE across the 3 horizons = better. Returns the winning model's name."""
    avg_rmse = {
        name: np.mean([m["rmse"] for m in horizons.values()])
        for name, horizons in results.items()
    }
    best_name = min(avg_rmse, key=avg_rmse.get)
    print(f"\nBest model: {best_name} (avg RMSE across horizons = {avg_rmse[best_name]:.2f})")
    return best_name


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from training.train import load_training_data, prepare_features, train_all

    df = load_training_data()
    print(f"Loaded {len(df)} historical rows from Hopsworks.")

    X, Y = prepare_features(df)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    fitted_models = train_all(X_train, Y_train)
    results = evaluate_all(fitted_models, X_test, Y_test)
    pick_best_model(results)