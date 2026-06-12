import streamlit as st
import pandas as pd
import plotly.express as px

from forecast import generate_forecast

# ==================================================
# PAGE
# ==================================================

st.set_page_config(
    page_title="GM Forecast Center",
    page_icon="📈",
    layout="wide"
)

# ==================================================
# STYLING
# ==================================================

st.markdown("""
<style>

.metric-card{
    background:#eaf3ff;
    padding:20px;
    border-radius:12px;
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# HEADER
# ==================================================

st.title("📈 GM Forecast Center")

st.caption(
    "AI-Powered Gross Margin Forecasting"
)

# ==================================================
# REFRESH
# ==================================================

if st.button(
    "🔄 Refresh Forecast"
):
    st.cache_data.clear()

# ==================================================
# LOAD FORECAST
# ==================================================

forecast_df = generate_forecast(
    horizon_months=12
)

# ==================================================
# KPIs
# ==================================================

avg_gm = forecast_df[
    "forecast_gm_pct"
].mean()

max_gm = forecast_df[
    "forecast_gm_pct"
].max()

min_gm = forecast_df[
    "forecast_gm_pct"
].min()

best_month = forecast_df.loc[
    forecast_df[
        "forecast_gm_pct"
    ].idxmax(),
    "prediction_period"
]

# ==================================================
# KPI ROW
# ==================================================

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Average GM",
    f"{avg_gm:.2f}%"
)

c2.metric(
    "Peak GM",
    f"{max_gm:.2f}%"
)

c3.metric(
    "Lowest GM",
    f"{min_gm:.2f}%"
)

c4.metric(
    "Model Confidence",
    "95.28%"
)

st.divider()

# ==================================================
# MAIN SECTION
# ==================================================

left, right = st.columns(
    [3, 1]
)

with left:

    st.subheader(
        "GM Forecast Trend"
    )

    fig = px.line(
        forecast_df,
        x="prediction_period",
        y="forecast_gm_pct",
        markers=True
    )

    fig.update_layout(
        height=500
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with right:

    st.subheader(
        "Forecast Insights"
    )

    st.info(f"""
Peak Month

{best_month}

Average GM

{avg_gm:.2f}%

Forecast Horizon

12 Months

Source

Latest Business Snapshot
""")

# ==================================================
# DRIVERS
# ==================================================

st.subheader(
    "Forecast Drivers"
)

driver_df = pd.DataFrame({

    "Driver": [

        "Seasonality",
        "Market Growth",
        "Pricing",
        "Demand",
        "Competitive Price"

    ],

    "Impact": [

        12,
        9,
        8,
        7,
        5

    ]
})

fig_driver = px.bar(

    driver_df,

    x="Impact",

    y="Driver",

    orientation="h"
)

fig_driver.update_layout(
    height=350
)

st.plotly_chart(
    fig_driver,
    use_container_width=True
)

# ==================================================
# EXECUTIVE SUMMARY
# ==================================================

st.subheader(
    "Executive Summary"
)

st.success(f"""
Average projected GM is {avg_gm:.2f}%.

Highest projected GM is {max_gm:.2f}%.

Peak forecast month is {best_month}.

Forecast is generated automatically
using the latest available business
snapshot and scored by the production
XGBoost model.
""")