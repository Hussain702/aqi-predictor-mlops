"""
Step 10 - Streamlit Dashboard (v3: EDA + SHAP + hazard alerts)
------------------------------------------------------------------
Adds, on top of the v2 tabs/header/footer layout:
    - "EDA" tab: AQI trends, distribution, hourly/monthly patterns,
      feature correlation heatmap (analytics/eda.py)
    - "Why this forecast?" SHAP feature-importance chart per city
      (serving/explain.py) -- answers "why did the model predict X?"
    - Hazardous AQI alert banners (per city + an aggregate warning on
      the Overview tab) when any forecast day crosses Unhealthy/Hazardous
    - Model loading now handles BOTH sklearn (Ridge/Random Forest) and
      TensorFlow (the neural network candidate) registered models,
      picking the loader based on the framework.txt marker registry.py
      saves alongside the model

How "verify results" works here: "Today" is the REAL, actual AQI value
from your live hourly pipeline's most recent row -- not a prediction.
Tomorrow/Day+2/Day+3 are the model's +24h/+48h/+72h forecasts (see
training/train.py's docstring for how that forecasting works).

Run:
    streamlit run dashboard/streamlit_app.py
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

# Streamlit only adds this file's own folder (dashboard/) to sys.path, not
# the project root -- unlike `python -m etl.load`, which does. Without this,
# `from utils...` / `from training...` fail with ModuleNotFoundError.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.hopsworks_client import get_project
from training.train import FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION, prepare_features
from serving.explain import explain_prediction
from serving.predict import load_latest_model, predict_for_city
from analytics.forecast_log import compute_accuracy
from analytics.eda import (
    aqi_trend_figure,
    aqi_distribution_figure,
    hourly_pattern_figure,
    monthly_pattern_figure,
    correlation_heatmap_figure,
)

CITIES = ["lahore", "islamabad", "karachi", "peshawar"]
CITY_TAB_ICONS = {"lahore": "\U0001F3DB", "islamabad": "\U0001F3DB", "karachi": "\U0001F30A", "peshawar": "\U0001F54C"}
HORIZON_LABELS = ["Today", "Tomorrow", "Day +2", "Day +3"]
HAZARD_THRESHOLD = 200  # "Unhealthy" and above triggers an alert banner

st.set_page_config(page_title="AeroWatch | AQI Forecast", page_icon="\U0001F32C", layout="wide")


# ---------------- AQI category helpers (standard US AQI breakpoints) ----------------
AQI_CATEGORIES = [
    (50, "Good", "#2ecc71", "\U0001F7E2"),
    (100, "Moderate", "#f1c40f", "\U0001F7E1"),
    (150, "Unhealthy (Sensitive)", "#e67e22", "\U0001F7E0"),
    (200, "Unhealthy", "#e74c3c", "\U0001F534"),
    (300, "Very Unhealthy", "#8e44ad", "\U0001F7E3"),
    (float("inf"), "Hazardous", "#7f1d1d", "\u26AB"),
]


def aqi_category(aqi: float):
    for threshold, label, color, emoji in AQI_CATEGORIES:
        if aqi <= threshold:
            return label, color, emoji
    return AQI_CATEGORIES[-1][1:]


# ---------------- Cached loaders (avoid re-connecting to Hopsworks on every rerun) ----------------
@st.cache_resource(show_spinner="Connecting to Hopsworks and loading the model...")
def load_model_and_data():
    project = get_project()

    model, feature_names, winning_model_name, framework, model_version = load_latest_model(project)

    fs = project.get_feature_store()
    fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    raw_df = fg.read()
    raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"])
    latest_per_city = raw_df.sort_values("timestamp").groupby("city").tail(1)

    # A background sample for SHAP's KernelExplainer (ignored by TreeExplainer,
    # used when the winning model isn't Random Forest -- Ridge or the NN).
    X_full, _ = prepare_features(raw_df)
    X_background = X_full.sample(min(200, len(X_full)), random_state=42)

    return model, feature_names, latest_per_city, model_version, raw_df, X_background, framework, winning_model_name


# ---------------- UI building blocks ----------------
def render_header():
    st.markdown(
        """
        <div style="background:linear-gradient(90deg,#1e3c72,#2a5298);
                    border-radius:16px;padding:24px 32px;margin-bottom:22px;
                    display:flex;align-items:center;gap:18px;">
            <div style="font-size:46px;">\U0001F32C\uFE0F</div>
            <div>
                <div style="font-size:30px;font-weight:800;color:white;letter-spacing:0.5px;">
                    AeroWatch
                </div>
                <div style="font-size:14px;color:#cbd8f0;">
                    Pakistan Air Quality Forecast &mdash; Lahore, Islamabad, Karachi, Peshawar
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer(model_version, model_name, framework):
    st.divider()
    st.markdown(
        f"""
        <div style="text-align:center;color:#888;font-size:12px;padding:8px 0;">
            AeroWatch &bull; Data: Hopsworks Feature Store &bull;
            Model: v{model_version} &mdash; {model_name} ({framework}) &bull;
            Dashboard refreshed {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(model_version, model_name, framework):
    st.sidebar.markdown("### \U0001F32C AeroWatch")
    st.sidebar.caption(f"Model version: {model_version} \u2014 {model_name} ({framework})")

    if st.sidebar.button("\U0001F504 Refresh Data", use_container_width=True):
        load_model_and_data.clear()
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("**AQI Legend**")
    for threshold, label, color, emoji in AQI_CATEGORIES:
        st.sidebar.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">'
            f'<div style="width:14px;height:14px;border-radius:4px;background:{color};"></div>'
            f'<div style="font-size:13px;">{label}</div></div>',
            unsafe_allow_html=True,
        )


def render_day_cards(values, labels=HORIZON_LABELS):
    cols = st.columns(len(values))
    for col, label, value in zip(cols, labels, values):
        category, color, emoji = aqi_category(value)
        with col:
            st.markdown(
                f"""
                <div style="background-color:{color}22;border:2px solid {color};
                            border-radius:14px;padding:18px 8px;text-align:center;">
                    <div style="font-size:13px;color:#888;">{label}</div>
                    <div style="font-size:34px;line-height:1.2;">{emoji}</div>
                    <div style="font-size:30px;font-weight:800;">{value:.0f}</div>
                    <div style="font-size:12px;color:{color};font-weight:700;">{category}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_trend_chart(values, labels=HORIZON_LABELS, height=220):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=labels, y=values, mode="lines+markers",
        line=dict(color="#3498db", width=3), marker=dict(size=10),
        fill="tozeroy", fillcolor="rgba(52,152,219,0.08)",
    ))
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=10, b=10),
        yaxis_title="AQI", showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_hazard_banner(city_label: str, values: list, labels=HORIZON_LABELS):
    """Shows a warning banner if any forecast day is Unhealthy (>200) or worse."""
    hazard_days = [
        (label, value) for label, value in zip(labels, values) if value >= HAZARD_THRESHOLD
    ]
    if not hazard_days:
        return
    worst_label, worst_value = max(hazard_days, key=lambda pair: pair[1])
    category, _, emoji = aqi_category(worst_value)
    st.error(
        f"{emoji} **Hazardous air alert &mdash; {city_label}**: "
        f"{category} air quality forecast on **{worst_label}** (AQI {worst_value:.0f}). "
        "Sensitive groups should limit outdoor exposure.",
        icon="\u26A0\uFE0F",
    )


def render_shap_section(model, X_row: pd.DataFrame, X_background: pd.DataFrame):
    """'Why this forecast?' -- SHAP feature importance for the +24h prediction."""
    st.markdown("##### Why this forecast? (feature impact on +24h AQI)")
    try:
        with st.spinner("Computing feature importance..."):
            shap_df = explain_prediction(model, X_row, X_background, horizon_index=0).head(8)

        fig = go.Figure(go.Bar(
            x=shap_df["shap_value"], y=shap_df["feature"], orientation="h",
            marker_color=["#e74c3c" if v > 0 else "#2ecc71" for v in shap_df["shap_value"]],
        ))
        fig.update_layout(
            height=280, margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Impact on predicted AQI (+24h)",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Red bars push the forecast up; green bars pull it down.")
    except Exception as e:
        st.info(f"Feature importance unavailable for this model right now ({e}).")


def render_overview_tab(latest_per_city, model, feature_names):
    st.subheader("Today's AQI \u2014 all cities")

    predictions_by_city = {}
    cols = st.columns(len(CITIES))
    for col, city in zip(cols, CITIES):
        city_rows = latest_per_city[latest_per_city["city"] == city]
        with col:
            if city_rows.empty:
                st.warning(f"No data yet for {city.title()}.")
                continue
            values, as_of, _ = predict_for_city(city_rows.iloc[0], model, feature_names)
            predictions_by_city[city] = values
            today_aqi = values[0]
            category, color, emoji = aqi_category(today_aqi)
            st.markdown(
                f"""
                <div style="background-color:{color}22;border:2px solid {color};
                            border-radius:16px;padding:22px 10px;text-align:center;">
                    <div style="font-size:15px;font-weight:700;">{city.title()}</div>
                    <div style="font-size:38px;">{emoji}</div>
                    <div style="font-size:34px;font-weight:800;">{today_aqi:.0f}</div>
                    <div style="font-size:12px;color:{color};font-weight:700;">{category}</div>
                    <div style="font-size:11px;color:#888;margin-top:6px;">
                        {as_of:%b %d, %H:%M} UTC
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    hazard_cities = [
        city for city, values in predictions_by_city.items() if max(values) >= HAZARD_THRESHOLD
    ]
    if hazard_cities:
        names = ", ".join(c.title() for c in hazard_cities)
        st.error(
            f"\u26A0\uFE0F **{len(hazard_cities)} of {len(CITIES)} cities** forecast Unhealthy "
            f"or worse air quality in the next 3 days: **{names}**.",
        )

    st.markdown("")
    st.caption("Select a city tab above for its full 3-day forecast, trend chart, and feature importance.")


def render_city_tab(city, latest_per_city, model, feature_names, X_background):
    city_rows = latest_per_city[latest_per_city["city"] == city]
    if city_rows.empty:
        st.warning(f"No data found yet for {city.title()}.")
        return

    values, as_of, X_row = predict_for_city(city_rows.iloc[0], model, feature_names)

    st.subheader(f"{city.title()} \u2014 3-day AQI forecast")
    st.caption(f"Latest reading: {as_of:%Y-%m-%d %H:%M} UTC")

    render_hazard_banner(city.title(), values)
    render_day_cards(values)
    render_trend_chart(values)
    st.markdown("")
    render_shap_section(model, X_row, X_background)


def render_eda_tab(raw_df: pd.DataFrame):
    st.subheader("Exploratory Data Analysis")
    st.caption("Trends and patterns across all historical data in the feature store.")

    city = st.selectbox("City for trend chart", CITIES, format_func=str.title, key="eda_city")
    st.plotly_chart(aqi_trend_figure(raw_df, city), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(hourly_pattern_figure(raw_df), use_container_width=True)
    with col2:
        st.plotly_chart(monthly_pattern_figure(raw_df), use_container_width=True)

    st.plotly_chart(aqi_distribution_figure(raw_df), use_container_width=True)
    st.plotly_chart(correlation_heatmap_figure(raw_df), use_container_width=True)


def render_accuracy_tab():
    """
    Real-world accuracy: what the model predicted vs. what actually
    happened, once enough time has passed for logged forecasts' target
    dates to have arrived. Different from the Model Performance numbers
    in the README, which come from a held-out TEST SPLIT of historical
    data -- this is the live model being checked against reality.
    """
    st.subheader("Forecast Accuracy \u2014 predicted vs. actual")
    st.caption(
        "Requires the daily forecast_logging DAG to have been running for at least "
        "a day or two, so some logged +24h predictions have had time to actually happen."
    )

    try:
        with st.spinner("Comparing logged forecasts against actual outcomes..."):
            accuracy_df = compute_accuracy()
    except Exception as e:
        st.info(f"Forecast log not available yet ({e}). Run the `forecast_logging` DAG first.")
        return

    if accuracy_df.empty:
        st.info(
            "No matched predictions yet \u2014 check back in a day or two once some "
            "logged forecasts' target dates have actually passed."
        )
        return

    summary = (
        accuracy_df.groupby(["horizon", "model_name"])["abs_error"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "MAE", "count": "n_predictions"})
    )
    st.dataframe(summary, use_container_width=True, hide_index=True)
    st.caption(
        "MAE = mean absolute error in AQI points, on REAL predictions the model "
        "already made and that have since come true \u2014 not a test-set estimate."
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=accuracy_df["aqi"], y=accuracy_df["predicted_aqi"], mode="markers",
        marker=dict(color="#3498db", size=7, opacity=0.6),
        text=accuracy_df["city"] + " \u00b7 " + accuracy_df["horizon"],
    ))
    max_val = max(accuracy_df["aqi"].max(), accuracy_df["predicted_aqi"].max()) + 10
    fig.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val], mode="lines",
        line=dict(color="#999", dash="dash"), showlegend=False,
    ))
    fig.update_layout(
        height=380, margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Actual AQI", yaxis_title="Predicted AQI",
        title="Predicted vs. actual (closer to the dashed line = better)",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)


def main():
    render_header()

    try:
        (model, feature_names, latest_per_city, model_version,
         raw_df, X_background, framework, winning_model_name) = load_model_and_data()
    except Exception as e:
        st.error(f"Could not load model/data from Hopsworks: {e}")
        return

    render_sidebar(model_version, winning_model_name, framework)

    tab_labels = ["\U0001F3E0 Overview"] + [
        f"{CITY_TAB_ICONS[c]} {c.title()}" for c in CITIES
    ] + ["\U0001F4CA EDA", "\U0001F3AF Accuracy"]
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        render_overview_tab(latest_per_city, model, feature_names)

    for tab, city in zip(tabs[1:-2], CITIES):
        with tab:
            render_city_tab(city, latest_per_city, model, feature_names, X_background)

    with tabs[-2]:
        render_eda_tab(raw_df)

    with tabs[-1]:
        render_accuracy_tab()

    render_footer(model_version, winning_model_name, framework)


if __name__ == "__main__":
    main()