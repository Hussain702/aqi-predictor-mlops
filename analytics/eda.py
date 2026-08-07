"""
Exploratory Data Analysis helpers.
Each function takes the raw feature-group dataframe (from Hopsworks) and
returns a Plotly figure. Used by the dashboard's "EDA" tab -- kept as
plain functions (not notebook cells) so the same trend/pattern logic is
reusable anywhere, not just inside Streamlit.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

DEFAULT_CORRELATION_COLUMNS = [
    "temperature", "humidity", "wind_speed", "pm25", "pm10", "aqi",
]


def aqi_trend_figure(df: pd.DataFrame, city: str):
    """Line chart of AQI over time for one city."""
    city_df = df[df["city"] == city].sort_values("timestamp")
    fig = px.line(city_df, x="timestamp", y="aqi", title=f"{city.title()}: AQI over time")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def aqi_distribution_figure(df: pd.DataFrame):
    """Histogram of AQI values, overlaid per city -- shows how often each city hits each severity band."""
    fig = px.histogram(
        df, x="aqi", color="city", nbins=40, barmode="overlay", opacity=0.6,
        title="AQI distribution by city",
    )
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def hourly_pattern_figure(df: pd.DataFrame):
    """Average AQI by hour of day, per city -- reveals daily pollution cycles (e.g. traffic/industry hours)."""
    hourly = df.groupby(["city", "hour"])["aqi"].mean().reset_index()
    fig = px.line(
        hourly, x="hour", y="aqi", color="city", markers=True,
        title="Average AQI by hour of day",
    )
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def monthly_pattern_figure(df: pd.DataFrame):
    """Average AQI by month, per city -- reveals seasonal patterns (e.g. winter smog)."""
    monthly = df.groupby(["city", "month"])["aqi"].mean().reset_index()
    fig = px.line(
        monthly, x="month", y="aqi", color="city", markers=True,
        title="Average AQI by month",
    )
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def correlation_heatmap_figure(df: pd.DataFrame, columns=None):
    """Correlation heatmap across the core numeric readings."""
    columns = columns or DEFAULT_CORRELATION_COLUMNS
    corr = df[columns].corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale="RdBu", zmin=-1, zmax=1,
        text=corr.round(2).values, texttemplate="%{text}",
    ))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=20, b=10))
    return fig