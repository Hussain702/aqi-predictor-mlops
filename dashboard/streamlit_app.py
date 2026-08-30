
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.hopsworks_client import get_project
from training.train import FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION
from serving.predict import load_latest_model, predict_for_city
from serving.explain import explain_prediction
from analytics.forecast_log import compute_accuracy, get_forecast_log_stats
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
HAZARD_THRESHOLD = 200

st.set_page_config(
    page_title="AeroWatch | AQI Forecast",
    page_icon="\U0001F32C",
    layout="wide",
    initial_sidebar_state="expanded",
)

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


@st.cache_resource(ttl=3600, show_spinner="Connecting to Hopsworks...")
def get_cached_project():
    """Hopsworks project connection, cached for one hour."""
    return get_project()


@st.cache_resource(ttl=3600, show_spinner="Loading latest AQI model...")
def load_cached_model():
    project = get_cached_project()
    model, feature_names, winning_model_name, framework, model_version = load_latest_model(project)
    return model, feature_names, winning_model_name, framework, model_version


@st.cache_resource(ttl=3600)
def get_cached_feature_group():
    project = get_cached_project()
    fs = project.get_feature_store()
    return fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)


@st.cache_data(ttl=600, show_spinner="Loading latest AQI readings...")
def load_latest_city_data():
    fg = get_cached_feature_group()
    raw_df = fg.read()

    if raw_df.empty:
        raise RuntimeError("The Hopsworks Feature Group returned no data.")
    if "timestamp" not in raw_df.columns:
        raise RuntimeError("Feature Group does not contain a 'timestamp' column.")

    raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"], errors="coerce")
    raw_df = raw_df.dropna(subset=["timestamp"])

    if "city" not in raw_df.columns:
        raise RuntimeError("Feature Group does not contain a 'city' column.")

    return raw_df.sort_values("timestamp").groupby("city").tail(1).copy()


@st.cache_data(ttl=1800, show_spinner="Loading historical AQI data...")
def load_historical_data():
    """Expensive. NOT called during initial load -- only when EDA page opens."""
    fg = get_cached_feature_group()
    raw_df = fg.read()

    if raw_df.empty:
        raise RuntimeError("No historical data found in the Feature Store.")

    raw_df["timestamp"] = pd.to_datetime(raw_df["timestamp"], errors="coerce")
    return raw_df.dropna(subset=["timestamp"])


@st.cache_data(ttl=1800, show_spinner="Preparing SHAP background data...")
def load_shap_background():
    """Expensive. Only called when the user requests SHAP."""
    from training.train import prepare_features
    raw_df = load_historical_data()
    X_full, _ = prepare_features(raw_df)
    if X_full.empty:
        raise RuntimeError("Could not create SHAP background data.")
    return X_full.sample(min(100, len(X_full)), random_state=42)


def render_header():
    st.markdown(
        """
        <div style="background:linear-gradient(90deg,#1e3c72,#2a5298);
                    border-radius:16px;padding:22px 30px;margin-bottom:20px;
                    display:flex;align-items:center;gap:18px;">
            <div style="font-size:44px;">\U0001F32C\uFE0F</div>
            <div>
                <div style="font-size:30px;font-weight:800;color:white;">AeroWatch</div>
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
            AeroWatch &bull; Hopsworks Feature Store &bull;
            Model: v{model_version} &mdash; {model_name} ({framework}) &bull;
            Refreshed: {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(model_version, model_name, framework):
    st.sidebar.markdown("### \U0001F32C AeroWatch")
    st.sidebar.caption(f"Model v{model_version} \u2014 {model_name} ({framework})")

    if st.sidebar.button("\U0001F504 Refresh Data", width="stretch"):
        get_cached_project.clear()
        load_cached_model.clear()
        get_cached_feature_group.clear()
        load_latest_city_data.clear()
        load_historical_data.clear()
        load_shap_background.clear()
        # Also clear cached SHAP results -- they were computed against
        # now-stale data/model.
        for key in list(st.session_state.keys()):
            if key.startswith("shap_result_"):
                del st.session_state[key]
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("**AQI Legend**")
    for threshold, label, color, emoji in AQI_CATEGORIES:
        st.sidebar.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:5px;">'
            f'<div style="width:14px;height:14px;border-radius:4px;background:{color};"></div>'
            f'<div style="font-size:13px;">{emoji} {label}</div></div>',
            unsafe_allow_html=True,
        )
    st.sidebar.markdown("---")
    st.sidebar.caption("Data cache: 10 minutes\n\nModel cache: 1 hour")


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
                    <div style="font-size:34px;">{emoji}</div>
                    <div style="font-size:30px;font-weight:800;">{value:.0f}</div>
                    <div style="font-size:12px;color:{color};font-weight:700;">{category}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_trend_chart(values, labels=HORIZON_LABELS):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=labels, y=values, mode="lines+markers",
        line=dict(color="#3498db", width=3), marker=dict(size=10),
        fill="tozeroy", fillcolor="rgba(52,152,219,0.08)",
    ))
    fig.update_layout(
        height=260, margin=dict(l=10, r=10, t=10, b=10),
        yaxis_title="AQI", showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, width="stretch")


def render_hazard_banner(city_label, values, labels=HORIZON_LABELS):
    hazard_days = [(l, v) for l, v in zip(labels, values) if v >= HAZARD_THRESHOLD]
    if not hazard_days:
        return
    worst_label, worst_value = max(hazard_days, key=lambda x: x[1])
    category, _, emoji = aqi_category(worst_value)
    st.error(
        f"{emoji} **Air quality alert \u2014 {city_label}**\n\n"
        f"{category} air quality forecast on **{worst_label}** "
        f"(AQI {worst_value:.0f}). Sensitive groups should limit outdoor exposure."
    )


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
                            border-radius:16px;padding:20px 10px;text-align:center;">
                    <div style="font-size:15px;font-weight:700;">{city.title()}</div>
                    <div style="font-size:38px;">{emoji}</div>
                    <div style="font-size:34px;font-weight:800;">{today_aqi:.0f}</div>
                    <div style="font-size:12px;color:{color};font-weight:700;">{category}</div>
                    <div style="font-size:11px;color:#888;margin-top:6px;">{as_of:%b %d, %H:%M} UTC</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    hazard_cities = [c for c, v in predictions_by_city.items() if max(v) >= HAZARD_THRESHOLD]
    if hazard_cities:
        names = ", ".join(c.title() for c in hazard_cities)
        st.error(
            f"\u26A0\uFE0F **{len(hazard_cities)} of {len(CITIES)} cities** are forecast "
            f"Unhealthy or worse during the next 3 days: **{names}**."
        )

    st.markdown("")
    st.info(
        "\U0001F4A1 Select a city from the City Forecast page to see the detailed "
        "forecast and optionally calculate SHAP."
    )


def render_city_tab(city, latest_per_city, model, feature_names):
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
    st.markdown("---")

    # SHAP result is stashed in session_state so it doesn't vanish the next
    # time the script reruns for an unrelated reason (st.button() only
    # returns True on the exact run it was clicked -- without
    # session_state, the chart disappears on the very next interaction).
    st.markdown("### \U0001F50E Why this forecast?")
    st.caption(
        "SHAP analysis is computationally expensive. Click the button only "
        "when you want to inspect which features influenced the +24h forecast."
    )

    shap_state_key = f"shap_result_{city}"

    if st.button(f"Calculate feature impact for {city.title()}", key=f"shap_btn_{city}", width="stretch"):
        try:
            with st.spinner("Calculating SHAP feature importance..."):
                X_background = load_shap_background()
                shap_df = explain_prediction(model, X_row, X_background, horizon_index=0).head(8)
            st.session_state[shap_state_key] = shap_df
        except Exception as e:
            st.warning("SHAP calculation failed.")
            st.exception(e)
            st.session_state.pop(shap_state_key, None)

    if shap_state_key in st.session_state:
        shap_df = st.session_state[shap_state_key]
        fig = go.Figure(go.Bar(
            x=shap_df["shap_value"], y=shap_df["feature"], orientation="h",
            marker_color=["#e74c3c" if v > 0 else "#2ecc71" for v in shap_df["shap_value"]],
        ))
        fig.update_layout(
            height=300, margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Impact on predicted AQI (+24h)",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, width="stretch")
        st.caption("Red = pushes forecast higher. Green = pushes forecast lower.")


def render_eda_tab():
    st.subheader("\U0001F4CA Exploratory Data Analysis")
    st.caption("Historical Feature Store analytics.")

    with st.spinner("Loading historical data from Hopsworks..."):
        raw_df = load_historical_data()

    st.success(f"Loaded {len(raw_df):,} historical records.")

    city = st.selectbox("City for trend chart", CITIES, format_func=str.title, key="eda_city")
    st.plotly_chart(aqi_trend_figure(raw_df, city), width="stretch")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(hourly_pattern_figure(raw_df), width="stretch")
    with col2:
        st.plotly_chart(monthly_pattern_figure(raw_df), width="stretch")

    st.plotly_chart(aqi_distribution_figure(raw_df), width="stretch")
    st.plotly_chart(correlation_heatmap_figure(raw_df), width="stretch")


def render_accuracy_tab():
    st.subheader("\U0001F3AF Forecast Accuracy")
    st.caption("Predicted vs actual AQI from logged live forecasts.")

    if not st.button("Load Forecast Accuracy", width="stretch"):
        st.info("Click the button to query the forecast log.")
        return

    try:
        with st.spinner("Comparing forecasts with actual outcomes..."):
            accuracy_df = compute_accuracy()
    except Exception as e:
        st.warning("Forecast log is not available yet.")
        st.exception(e)
        return

    if accuracy_df.empty:
        stats = get_forecast_log_stats()

        if stats["total_logged"] == 0:
            st.warning(
                "No forecasts have been logged at all yet. Check that the "
                "`forecast_logging` DAG actually ran successfully -- open its "
                "task logs in the Airflow UI and confirm it printed "
                "\"Logged N forecasts...\" rather than an error."
            )
        else:
            now = pd.Timestamp.now(tz="UTC")
            earliest = stats["earliest_target"]
            time_remaining = earliest - now

            if time_remaining.total_seconds() > 0:
                hours_left = time_remaining.total_seconds() / 3600
                st.info(
                    f"{stats['total_logged']} forecast(s) logged so far, but none of "
                    f"their target dates have arrived yet. Earliest target: "
                    f"**{earliest:%Y-%m-%d %H:%M} UTC** (about {hours_left:.1f}h from now). "
                    "Check back after that -- this is expected right after the first run."
                )
            else:
                st.info(
                    f"{stats['total_logged']} forecast(s) logged, and the earliest target "
                    f"date ({earliest:%Y-%m-%d %H:%M} UTC) has already passed, but no matching "
                    "actual reading was found within the tolerance window. This can happen if "
                    "the hourly feature_pipeline had a gap right around that time -- should "
                    "resolve itself as more hourly data accumulates."
                )
        return

    summary = (
        accuracy_df.groupby(["horizon", "model_name"])["abs_error"]
        .agg(["mean", "count"]).reset_index()
        .rename(columns={"mean": "MAE", "count": "n_predictions"})
    )
    st.dataframe(summary, width="stretch", hide_index=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=accuracy_df["aqi"], y=accuracy_df["predicted_aqi"], mode="markers",
        marker=dict(size=7, opacity=0.6),
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
        title="Predicted vs actual (closer to dashed line = better)",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, width="stretch")


def main():
    render_header()

    try:
        (model, feature_names, winning_model_name, framework,
         model_version) = load_cached_model()
    except Exception as e:
        st.error("Could not load the AQI model from Hopsworks.")
        st.exception(e)
        return

    try:
        latest_per_city = load_latest_city_data()
    except Exception as e:
        st.error("Could not load latest AQI data from Hopsworks.")
        st.exception(e)
        return

    render_sidebar(model_version, winning_model_name, framework)

    # st.radio (not st.tabs) is what makes this actually lazy: st.tabs
    # renders every tab's code on every rerun regardless of which is
    # visually selected, defeating the whole point of lazy-loading EDA/SHAP.
    page = st.radio(
        "Navigation",
        ["\U0001F3E0 Overview", "\U0001F3D9\uFE0F City Forecast", "\U0001F4CA EDA", "\U0001F3AF Accuracy"],
        horizontal=True,
    )
    st.markdown("---")

    if page == "\U0001F3E0 Overview":
        render_overview_tab(latest_per_city, model, feature_names)
    elif page == "\U0001F3D9\uFE0F City Forecast":
        city = st.selectbox("Select city", CITIES, format_func=str.title)
        render_city_tab(city, latest_per_city, model, feature_names)
    elif page == "\U0001F4CA EDA":
        render_eda_tab()
    elif page == "\U0001F3AF Accuracy":
        render_accuracy_tab()

    render_footer(model_version, winning_model_name, framework)


if __name__ == "__main__":
    main()