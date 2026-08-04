"""
Step 10 - Streamlit Dashboard (v2: tabs + header/footer + branding)
------------------------------------------------------------------------
Loads the best registered model + the latest feature-store row per city
from Hopsworks, and displays a weather-forecast-style AQI outlook
(Today / Tomorrow / Day +2 / Day +3).

Layout, for a supervisor demo:
    - Header banner with logo/wordmark
    - Tabs: "Overview" (all 4 cities at a glance, no scrolling) + one tab
      per city (full detail: 4 day-cards + trend chart) -- this replaces
      the old stacked layout that made you scroll through every city
    - Sidebar: AQI color-code legend, model version, manual refresh button
    - Footer with data source / last-updated info

How "verify results" works here: "Today" is the REAL, actual AQI value
from your live hourly pipeline's most recent row -- not a prediction.
Tomorrow/Day+2/Day+3 are the model's +24h/+48h/+72h forecasts (see
training/train.py's docstring for how that forecasting works).

Run:
    streamlit run dashboard/streamlit_app.py
"""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Streamlit only adds this file's own folder (dashboard/) to sys.path, not
# the project root -- unlike `python -m etl.load`, which does. Without this,
# `from utils...` / `from training...` fail with ModuleNotFoundError.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.hopsworks_client import get_project
from training.registry import MODEL_NAME
from training.train import (
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
    FEATURE_COLUMNS,
    CATEGORICAL_COLUMN,
)

CITIES = ["lahore", "islamabad", "karachi", "peshawar"]
CITY_TAB_ICONS = {"lahore": "\U0001F3DB", "islamabad": "\U0001F3DB", "karachi": "\U0001F30A", "peshawar": "\U0001F54C"}
HORIZON_LABELS = ["Today", "Tomorrow", "Day +2", "Day +3"]

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
def load_model_and_latest_features():
    project = get_project()

    mr = project.get_model_registry()
    try:
        model_meta = mr.get_best_model(MODEL_NAME, "rmse_24h", "min")
    except Exception:
        models = mr.get_models(MODEL_NAME)
        if not models:
            raise RuntimeError(
                f"No registered model named '{MODEL_NAME}' found. "
                "Run `python -m training.registry` first."
            )
        model_meta = max(models, key=lambda m: m.version)

    model_dir = model_meta.download()
    model = joblib.load(os.path.join(model_dir, "model.pkl"))
    with open(os.path.join(model_dir, "feature_names.txt")) as f:
        feature_names = f.read().splitlines()

    fs = project.get_feature_store()
    fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    df = fg.read()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    latest_per_city = df.sort_values("timestamp").groupby("city").tail(1)

    return model, feature_names, latest_per_city, model_meta.version


def build_feature_row(city_row: pd.Series, feature_names: list) -> pd.DataFrame:
    """Build a single-row DataFrame matching the model's expected column order."""
    row = {col: city_row[col] for col in FEATURE_COLUMNS}
    for name in feature_names:
        if name.startswith("city_"):
            row[name] = 1 if name == f"city_{city_row[CATEGORICAL_COLUMN]}" else 0
    return pd.DataFrame([row])[feature_names]


def predict_for_city(city_row: pd.Series, model, feature_names: list):
    """Returns (values=[today, +24h, +48h, +72h], as_of timestamp)."""
    X = build_feature_row(city_row, feature_names)
    predictions = np.asarray(model.predict(X)).flatten()
    values = [float(city_row["aqi"])] + list(predictions)
    return values, city_row["timestamp"]


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


def render_footer(model_version):
    st.divider()
    st.markdown(
        f"""
        <div style="text-align:center;color:#888;font-size:12px;padding:8px 0;">
            AeroWatch &bull; Data: Hopsworks Feature Store &bull;
            Model: v{model_version} (Random Forest / Ridge Regression) &bull;
            Dashboard refreshed {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(model_version):
    st.sidebar.markdown("### \U0001F32C AeroWatch")
    st.sidebar.caption(f"Model version: {model_version}")

    if st.sidebar.button("\U0001F504 Refresh Data", use_container_width=True):
        load_model_and_latest_features.clear()
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


def render_overview_tab(latest_per_city, model, feature_names):
    st.subheader("Today's AQI \u2014 all cities")
    cols = st.columns(len(CITIES))

    for col, city in zip(cols, CITIES):
        city_rows = latest_per_city[latest_per_city["city"] == city]
        with col:
            if city_rows.empty:
                st.warning(f"No data yet for {city.title()}.")
                continue
            values, as_of = predict_for_city(city_rows.iloc[0], model, feature_names)
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

    st.markdown("")
    st.caption(
        "Select a city tab above for its full 3-day forecast and trend chart."
    )


def render_city_tab(city, latest_per_city, model, feature_names):
    city_rows = latest_per_city[latest_per_city["city"] == city]
    if city_rows.empty:
        st.warning(f"No data found yet for {city.title()}.")
        return

    values, as_of = predict_for_city(city_rows.iloc[0], model, feature_names)

    st.subheader(f"{city.title()} \u2014 3-day AQI forecast")
    st.caption(f"Latest reading: {as_of:%Y-%m-%d %H:%M} UTC")

    render_day_cards(values)
    render_trend_chart(values)


def main():
    render_header()

    try:
        model, feature_names, latest_per_city, model_version = load_model_and_latest_features()
    except Exception as e:
        st.error(f"Could not load model/data from Hopsworks: {e}")
        return

    render_sidebar(model_version)

    tab_labels = ["\U0001F3E0 Overview"] + [
        f"{CITY_TAB_ICONS[c]} {c.title()}" for c in CITIES
    ]
    tabs = st.tabs(tab_labels)

    with tabs[0]:
        render_overview_tab(latest_per_city, model, feature_names)

    for tab, city in zip(tabs[1:], CITIES):
        with tab:
            render_city_tab(city, latest_per_city, model, feature_names)

    render_footer(model_version)


if __name__ == "__main__":
    main()