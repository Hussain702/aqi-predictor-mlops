"""
Step 11 - SHAP Explainability
---------------------------------
Explains an individual AQI prediction: which features pushed the forecast
up or down, and by how much. This is what answers "why did the model
predict 170?" for a supervisor or anyone reading the dashboard.

Uses shap.TreeExplainer for the Random Forest (fast, exact) and falls
back to shap.KernelExplainer for anything else (Ridge, the neural
network) -- slower and approximate, but model-agnostic.

Used by the dashboard's per-city "Why this forecast?" section.
"""

import numpy as np
import pandas as pd
import shap


def get_explainer(model, background_X: pd.DataFrame):
    """
    Pick the right SHAP explainer for the model type.
    background_X: a sample of feature rows used as the reference
    distribution for KernelExplainer (TreeExplainer ignores it).
    """
    if type(model).__name__ == "RandomForestRegressor":
        return shap.TreeExplainer(model)

    background = shap.sample(background_X, min(50, len(background_X)))
    return shap.KernelExplainer(model.predict, background)


def explain_prediction(
    model, X_row: pd.DataFrame, background_X: pd.DataFrame, horizon_index: int = 0
) -> pd.DataFrame:
    """
    Returns a DataFrame of {feature, shap_value} for one prediction, for
    ONE forecast horizon (0 = +24h, 1 = +48h, 2 = +72h), sorted by impact
    (largest absolute contribution first).
    """
    explainer = get_explainer(model, background_X)
    shap_values = explainer.shap_values(X_row)

    # Multi-output SHAP results come back in different shapes depending on
    # the explainer/model -- normalize the common cases here.
    if isinstance(shap_values, list):
        values = np.asarray(shap_values[horizon_index])[0]
    elif np.asarray(shap_values).ndim == 3:
        values = np.asarray(shap_values)[0, :, horizon_index]
    else:
        values = np.asarray(shap_values)[0]

    result = pd.DataFrame({"feature": X_row.columns, "shap_value": values})
    result["abs_impact"] = result["shap_value"].abs()
    return result.sort_values("abs_impact", ascending=False).drop(columns="abs_impact")