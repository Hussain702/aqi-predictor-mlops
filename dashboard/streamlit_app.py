
"""
AeroWatch - Streamlit Dashboard
--------------------------------

Optimized version:
- Fast initial dashboard
- Hopsworks connection/model cached separately
- Historical Feature Store data is NOT loaded on startup
- SHAP is calculated only when explicitly requested
- EDA is loaded only when the EDA tab is opened
- Accuracy is loaded only when requested
- Avoids doing expensive work for every tab during initial render
"""

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# -------------------------------------------------------------------
# Project root
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# -------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.hopsworks_client import get_project

from training.train import (
    FEATURE_GROUP_NAME,
    FEATURE_GROUP_VERSION,
)

from serving.predict import (
    load_latest_model,
    predict_for_city,
)

from serving.explain import explain_prediction

from analytics.forecast_log import compute_accuracy

from analytics.eda import (
    aqi_trend_figure,
    aqi_distribution_figure,
    hourly_pattern_figure,
    monthly_pattern_figure,
    correlation_heatmap_figure,
)


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

CITIES = [
    "lahore",
    "islamabad",
    "karachi",
    "peshawar",
]

CITY_TAB_ICONS = {
    "lahore": "🏛",
    "islamabad": "🏛",
    "karachi": "🌊",
    "peshawar": "🕌",
}

HORIZON_LABELS = [
    "Today",
    "Tomorrow",
    "Day +2",
    "Day +3",
]

HAZARD_THRESHOLD = 200

st.set_page_config(
    page_title="AeroWatch | AQI Forecast",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -------------------------------------------------------------------
# AQI categories
# -------------------------------------------------------------------

AQI_CATEGORIES = [
    (50, "Good", "#2ecc71", "🟢"),
    (100, "Moderate", "#f1c40f", "🟡"),
    (150, "Unhealthy (Sensitive)", "#e67e22", "🟠"),
    (200, "Unhealthy", "#e74c3c", "🔴"),
    (300, "Very Unhealthy", "#8e44ad", "🟣"),
    (float("inf"), "Hazardous", "#7f1d1d", "⚫"),
]


def aqi_category(aqi: float):
    for threshold, label, color, emoji in AQI_CATEGORIES:
        if aqi <= threshold:
            return label, color, emoji

    return AQI_CATEGORIES[-1][1:]


# -------------------------------------------------------------------
# Performance helper
# -------------------------------------------------------------------

def timed_message(label, start):
    elapsed = time.perf_counter() - start
    return f"{label}: {elapsed:.2f}s"


# -------------------------------------------------------------------
# Hopsworks connection
# -------------------------------------------------------------------

@st.cache_resource(
    ttl=3600,
    show_spinner="Connecting to Hopsworks..."
)
def get_cached_project():
    """
    Hopsworks project connection.

    Cached for one hour so Streamlit reruns don't repeatedly
    authenticate against Hopsworks.
    """
    start = time.perf_counter()

    project = get_project()

    elapsed = time.perf_counter() - start

    return project


# -------------------------------------------------------------------
# Model loading
# -------------------------------------------------------------------

@st.cache_resource(
    ttl=3600,
    show_spinner="Loading latest AQI model..."
)
def load_cached_model():
    """
    Load the latest registered model only once.

    This is intentionally separated from Feature Store loading.
    """

    project = get_cached_project()

    model, feature_names, winning_model_name, framework, model_version = (
        load_latest_model(project)
    )

    return (
        model,
        feature_names,
        winning_model_name,
        framework,
        model_version,
    )


# -------------------------------------------------------------------
# Feature Store
# -------------------------------------------------------------------

@st.cache_resource(ttl=3600)
def get_cached_feature_group():
    """
    Return the Hopsworks Feature Group object.

    The actual dataframe is loaded separately using cache_data.
    """

    project = get_cached_project()

    fs = project.get_feature_store()

    fg = fs.get_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
    )

    return fg


# -------------------------------------------------------------------
# Latest city data
# -------------------------------------------------------------------

@st.cache_data(
    ttl=600,
    show_spinner="Loading latest AQI readings..."
)
def load_latest_city_data():
    """
    Load Feature Store data and retain only the newest row per city.

    This result is cached for 10 minutes.
    """

    start = time.perf_counter()

    fg = get_cached_feature_group()

    raw_df = fg.read()

    if raw_df.empty:
        raise RuntimeError(
            "The Hopsworks Feature Group returned no data."
        )

    if "timestamp" not in raw_df.columns:
        raise RuntimeError(
            "Feature Group does not contain a 'timestamp' column."
        )

    raw_df["timestamp"] = pd.to_datetime(
        raw_df["timestamp"],
        errors="coerce",
    )

    raw_df = raw_df.dropna(
        subset=["timestamp"]
    )

    if "city" not in raw_df.columns:
        raise RuntimeError(
            "Feature Group does not contain a 'city' column."
        )

    latest_per_city = (
        raw_df
        .sort_values("timestamp")
        .groupby("city")
        .tail(1)
        .copy()
    )

    return latest_per_city


# -------------------------------------------------------------------
# Historical data
# -------------------------------------------------------------------

@st.cache_data(
    ttl=1800,
    show_spinner="Loading historical AQI data..."
)
def load_historical_data():
    """
    Expensive operation.

    IMPORTANT:
    This is NOT called during initial dashboard loading.

    It only runs when the EDA section needs historical data.
    """

    fg = get_cached_feature_group()

    raw_df = fg.read()

    if raw_df.empty:
        raise RuntimeError(
            "No historical data found in the Feature Store."
        )

    raw_df["timestamp"] = pd.to_datetime(
        raw_df["timestamp"],
        errors="coerce",
    )

    raw_df = raw_df.dropna(
        subset=["timestamp"]
    )

    return raw_df


# -------------------------------------------------------------------
# SHAP background data
# -------------------------------------------------------------------

@st.cache_data(
    ttl=1800,
    show_spinner="Preparing SHAP background data..."
)
def load_shap_background():
    """
    Expensive operation.

    Only called when user requests SHAP.

    We intentionally don't calculate this at application startup.
    """

    from training.train import prepare_features

    raw_df = load_historical_data()

    X_full, _ = prepare_features(raw_df)

    if X_full.empty:
        raise RuntimeError(
            "Could not create SHAP background data."
        )

    return X_full.sample(
        min(100, len(X_full)),
        random_state=42,
    )


# -------------------------------------------------------------------
# Header
# -------------------------------------------------------------------

def render_header():

    st.markdown(
        """
        <div style="
            background:linear-gradient(90deg,#1e3c72,#2a5298);
            border-radius:16px;
            padding:22px 30px;
            margin-bottom:20px;
            display:flex;
            align-items:center;
            gap:18px;
        ">

            <div style="font-size:44px;">
                🌬️
            </div>

            <div>
                <div style="
                    font-size:30px;
                    font-weight:800;
                    color:white;
                ">
                    AeroWatch
                </div>

                <div style="
                    font-size:14px;
                    color:#cbd8f0;
                ">
                    Pakistan Air Quality Forecast
                    — Lahore, Islamabad, Karachi, Peshawar
                </div>
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


# -------------------------------------------------------------------
# Footer
# -------------------------------------------------------------------

def render_footer(
    model_version,
    model_name,
    framework,
):

    st.divider()

    st.markdown(
        f"""
        <div style="
            text-align:center;
            color:#888;
            font-size:12px;
            padding:8px 0;
        ">

            AeroWatch • Hopsworks Feature Store •
            Model: v{model_version} —
            {model_name} ({framework}) •

            Refreshed:
            {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC

        </div>
        """,
        unsafe_allow_html=True,
    )


# -------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------

def render_sidebar(
    model_version,
    model_name,
    framework,
):

    st.sidebar.markdown("### 🌬️ AeroWatch")

    st.sidebar.caption(
        f"Model v{model_version} — "
        f"{model_name} ({framework})"
    )

    if st.sidebar.button(
        "🔄 Refresh Data",
        use_container_width=True,
    ):

        get_cached_project.clear()
        load_cached_model.clear()
        get_cached_feature_group.clear()
        load_latest_city_data.clear()
        load_historical_data.clear()
        load_shap_background.clear()

        st.rerun()

    st.sidebar.markdown("---")

    st.sidebar.markdown("**AQI Legend**")

    for threshold, label, color, emoji in AQI_CATEGORIES:

        st.sidebar.markdown(
            f"""
            <div style="
                display:flex;
                align-items:center;
                gap:8px;
                margin-bottom:5px;
            ">

                <div style="
                    width:14px;
                    height:14px;
                    border-radius:4px;
                    background:{color};
                ">
                </div>

                <div style="font-size:13px;">
                    {emoji} {label}
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    st.sidebar.markdown("---")

    st.sidebar.caption(
        "Data cache: 10 minutes\n\n"
        "Model cache: 1 hour"
    )


# -------------------------------------------------------------------
# Day cards
# -------------------------------------------------------------------

def render_day_cards(
    values,
    labels=HORIZON_LABELS,
):

    cols = st.columns(len(values))

    for col, label, value in zip(
        cols,
        labels,
        values,
    ):

        category, color, emoji = aqi_category(value)

        with col:

            st.markdown(
                f"""
                <div style="
                    background-color:{color}22;
                    border:2px solid {color};
                    border-radius:14px;
                    padding:18px 8px;
                    text-align:center;
                ">

                    <div style="
                        font-size:13px;
                        color:#888;
                    ">
                        {label}
                    </div>

                    <div style="font-size:34px;">
                        {emoji}
                    </div>

                    <div style="
                        font-size:30px;
                        font-weight:800;
                    ">
                        {value:.0f}
                    </div>

                    <div style="
                        font-size:12px;
                        color:{color};
                        font-weight:700;
                    ">
                        {category}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )


# -------------------------------------------------------------------
# Forecast chart
# -------------------------------------------------------------------

def render_trend_chart(
    values,
    labels=HORIZON_LABELS,
):

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=labels,
            y=values,
            mode="lines+markers",
            line=dict(
                color="#3498db",
                width=3,
            ),
            marker=dict(
                size=10,
            ),
            fill="tozeroy",
            fillcolor="rgba(52,152,219,0.08)",
        )
    )

    fig.update_layout(
        height=260,
        margin=dict(
            l=10,
            r=10,
            t=10,
            b=10,
        ),
        yaxis_title="AQI",
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# -------------------------------------------------------------------
# Hazard warning
# -------------------------------------------------------------------

def render_hazard_banner(
    city_label,
    values,
    labels=HORIZON_LABELS,
):

    hazard_days = [
        (label, value)
        for label, value in zip(labels, values)
        if value >= HAZARD_THRESHOLD
    ]

    if not hazard_days:
        return

    worst_label, worst_value = max(
        hazard_days,
        key=lambda x: x[1],
    )

    category, _, emoji = aqi_category(
        worst_value
    )

    st.error(
        f"{emoji} **Air quality alert — {city_label}**\n\n"
        f"{category} air quality forecast on "
        f"**{worst_label}** "
        f"(AQI {worst_value:.0f}). "
        f"Sensitive groups should limit outdoor exposure."
    )


# -------------------------------------------------------------------
# Overview
# -------------------------------------------------------------------

def render_overview_tab(
    latest_per_city,
    model,
    feature_names,
):

    st.subheader(
        "Today's AQI — all cities"
    )

    predictions_by_city = {}

    cols = st.columns(
        len(CITIES)
    )

    for col, city in zip(
        cols,
        CITIES,
    ):

        city_rows = latest_per_city[
            latest_per_city["city"] == city
        ]

        with col:

            if city_rows.empty:

                st.warning(
                    f"No data yet for {city.title()}."
                )

                continue

            values, as_of, _ = predict_for_city(
                city_rows.iloc[0],
                model,
                feature_names,
            )

            predictions_by_city[city] = values

            today_aqi = values[0]

            category, color, emoji = (
                aqi_category(today_aqi)
            )

            st.markdown(
                f"""
                <div style="
                    background-color:{color}22;
                    border:2px solid {color};
                    border-radius:16px;
                    padding:20px 10px;
                    text-align:center;
                ">

                    <div style="
                        font-size:15px;
                        font-weight:700;
                    ">
                        {city.title()}
                    </div>

                    <div style="
                        font-size:38px;
                    ">
                        {emoji}
                    </div>

                    <div style="
                        font-size:34px;
                        font-weight:800;
                    ">
                        {today_aqi:.0f}
                    </div>

                    <div style="
                        font-size:12px;
                        color:{color};
                        font-weight:700;
                    ">
                        {category}
                    </div>

                    <div style="
                        font-size:11px;
                        color:#888;
                        margin-top:6px;
                    ">
                        {as_of:%b %d, %H:%M} UTC
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

    # ---------------------------------------------------------------
    # Aggregate warning
    # ---------------------------------------------------------------

    hazard_cities = [
        city
        for city, values
        in predictions_by_city.items()
        if max(values) >= HAZARD_THRESHOLD
    ]

    if hazard_cities:

        names = ", ".join(
            c.title()
            for c in hazard_cities
        )

        st.error(
            f"⚠️ **{len(hazard_cities)} of {len(CITIES)} cities** "
            f"are forecast Unhealthy or worse during the next "
            f"3 days: **{names}**."
        )

    st.markdown("")

    st.info(
        "💡 Select a city from the City Forecast page "
        "to see the detailed forecast and optionally calculate SHAP."
    )


# -------------------------------------------------------------------
# City forecast
# -------------------------------------------------------------------

def render_city_tab(
    city,
    latest_per_city,
    model,
    feature_names,
):

    city_rows = latest_per_city[
        latest_per_city["city"] == city
    ]

    if city_rows.empty:

        st.warning(
            f"No data found yet for {city.title()}."
        )

        return

    values, as_of, X_row = predict_for_city(
        city_rows.iloc[0],
        model,
        feature_names,
    )

    st.subheader(
        f"{city.title()} — 3-day AQI forecast"
    )

    st.caption(
        f"Latest reading: "
        f"{as_of:%Y-%m-%d %H:%M} UTC"
    )

    render_hazard_banner(
        city.title(),
        values,
    )

    render_day_cards(values)

    render_trend_chart(values)

    st.markdown("---")

    # ---------------------------------------------------------------
    # SHAP is now OPTIONAL
    # ---------------------------------------------------------------

    st.markdown(
        "### 🔎 Why this forecast?"
    )

    st.caption(
        "SHAP analysis is computationally expensive. "
        "Click the button only when you want to inspect "
        "which features influenced the +24h forecast."
    )

    if st.button(
        f"Calculate feature impact for {city.title()}",
        key=f"shap_{city}",
        use_container_width=True,
    ):

        try:

            with st.spinner(
                "Calculating SHAP feature importance..."
            ):

                X_background = (
                    load_shap_background()
                )

                shap_df = explain_prediction(
                    model,
                    X_row,
                    X_background,
                    horizon_index=0,
                ).head(8)

            fig = go.Figure(
                go.Bar(
                    x=shap_df["shap_value"],
                    y=shap_df["feature"],
                    orientation="h",
                    marker_color=[
                        "#e74c3c"
                        if value > 0
                        else "#2ecc71"
                        for value
                        in shap_df["shap_value"]
                    ],
                )
            )

            fig.update_layout(
                height=300,
                margin=dict(
                    l=10,
                    r=10,
                    t=10,
                    b=10,
                ),
                xaxis_title=(
                    "Impact on predicted AQI (+24h)"
                ),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

            st.caption(
                "Red = pushes forecast higher. "
                "Green = pushes forecast lower."
            )

        except Exception as e:

            st.warning(
                "SHAP calculation failed."
            )

            st.exception(e)


# -------------------------------------------------------------------
# EDA
# -------------------------------------------------------------------

def render_eda_tab():

    st.subheader(
        "📊 Exploratory Data Analysis"
    )

    st.caption(
        "Historical Feature Store analytics."
    )

    # Historical data is loaded ONLY here.

    with st.spinner(
        "Loading historical data from Hopsworks..."
    ):

        raw_df = load_historical_data()

    st.success(
        f"Loaded {len(raw_df):,} historical records."
    )

    city = st.selectbox(
        "City for trend chart",
        CITIES,
        format_func=str.title,
        key="eda_city",
    )

    st.plotly_chart(
        aqi_trend_figure(
            raw_df,
            city,
        ),
        use_container_width=True,
    )

    col1, col2 = st.columns(2)

    with col1:

        st.plotly_chart(
            hourly_pattern_figure(raw_df),
            use_container_width=True,
        )

    with col2:

        st.plotly_chart(
            monthly_pattern_figure(raw_df),
            use_container_width=True,
        )

    st.plotly_chart(
        aqi_distribution_figure(raw_df),
        use_container_width=True,
    )

    st.plotly_chart(
        correlation_heatmap_figure(raw_df),
        use_container_width=True,
    )


# -------------------------------------------------------------------
# Accuracy
# -------------------------------------------------------------------

def render_accuracy_tab():

    st.subheader(
        "🎯 Forecast Accuracy"
    )

    st.caption(
        "Predicted vs actual AQI from logged live forecasts."
    )

    if not st.button(
        "Load Forecast Accuracy",
        use_container_width=True,
    ):
        st.info(
            "Click the button to query the forecast log."
        )
        return

    try:

        with st.spinner(
            "Comparing forecasts with actual outcomes..."
        ):

            accuracy_df = compute_accuracy()

    except Exception as e:

        st.warning(
            "Forecast log is not available yet."
        )

        st.exception(e)

        return

    if accuracy_df.empty:

        st.info(
            "No matched predictions yet. "
            "Run the forecast logging DAG and wait "
            "for the forecast targets to occur."
        )

        return

    summary = (
        accuracy_df
        .groupby(
            ["horizon", "model_name"]
        )["abs_error"]
        .agg(
            ["mean", "count"]
        )
        .reset_index()
        .rename(
            columns={
                "mean": "MAE",
                "count": "n_predictions",
            }
        )
    )

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True,
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=accuracy_df["aqi"],
            y=accuracy_df["predicted_aqi"],
            mode="markers",
            marker=dict(
                size=7,
                opacity=0.6,
            ),
            text=(
                accuracy_df["city"]
                + " · "
                + accuracy_df["horizon"]
            ),
        )
    )

    max_val = max(
        accuracy_df["aqi"].max(),
        accuracy_df["predicted_aqi"].max(),
    ) + 10

    fig.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            line=dict(
                color="#999",
                dash="dash",
            ),
            showlegend=False,
        )
    )

    fig.update_layout(
        height=380,
        margin=dict(
            l=10,
            r=10,
            t=30,
            b=10,
        ),
        xaxis_title="Actual AQI",
        yaxis_title="Predicted AQI",
        title=(
            "Predicted vs actual "
            "(closer to dashed line = better)"
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():

    render_header()

    # ---------------------------------------------------------------
    # Load model
    # ---------------------------------------------------------------

    try:

        (
            model,
            feature_names,
            winning_model_name,
            framework,
            model_version,
        ) = load_cached_model()

    except Exception as e:

        st.error(
            "Could not load the AQI model from Hopsworks."
        )

        st.exception(e)

        return

    # ---------------------------------------------------------------
    # Load latest city data
    # ---------------------------------------------------------------

    try:

        latest_per_city = (
            load_latest_city_data()
        )

    except Exception as e:

        st.error(
            "Could not load latest AQI data "
            "from Hopsworks."
        )

        st.exception(e)

        return

    # ---------------------------------------------------------------
    # Sidebar
    # ---------------------------------------------------------------

    render_sidebar(
        model_version,
        winning_model_name,
        framework,
    )

    # ---------------------------------------------------------------
    # Main navigation
    #
    # IMPORTANT:
    # We intentionally do NOT use st.tabs().
    #
    # st.tabs() renders all tab content during a script run,
    # which defeats lazy loading for expensive EDA/SHAP operations.
    # ---------------------------------------------------------------

    page = st.radio(
        "Navigation",
        [
            "🏠 Overview",
            "🏙️ City Forecast",
            "📊 EDA",
            "🎯 Accuracy",
        ],
        horizontal=True,
    )

    st.markdown("---")

    # ---------------------------------------------------------------
    # Overview
    # ---------------------------------------------------------------

    if page == "🏠 Overview":

        render_overview_tab(
            latest_per_city,
            model,
            feature_names,
        )

    # ---------------------------------------------------------------
    # City forecast
    # ---------------------------------------------------------------

    elif page == "🏙️ City Forecast":

        city = st.selectbox(
            "Select city",
            CITIES,
            format_func=str.title,
        )

        render_city_tab(
            city,
            latest_per_city,
            model,
            feature_names,
        )

    # ---------------------------------------------------------------
    # EDA
    # ---------------------------------------------------------------

    elif page == "📊 EDA":

        render_eda_tab()

    # ---------------------------------------------------------------
    # Accuracy
    # ---------------------------------------------------------------

    elif page == "🎯 Accuracy":

        render_accuracy_tab()

    # ---------------------------------------------------------------
    # Footer
    # ---------------------------------------------------------------

    render_footer(
        model_version,
        winning_model_name,
        framework,
    )


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------

if __name__ == "__main__":
    main()

