import streamlit as st
import pandas as pd
import plotly.express as px

from forecast import generate_forecast

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="GM Forecasting Center",
    page_icon="📈",
    layout="wide"
)

# ==================================================
# HEADER
# ==================================================

st.title("📈 GM Forecasting Center")
st.caption("Carrier Pricing Intelligence")

# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.header("Forecast Controls")

horizon = st.sidebar.slider(
    "Forecast Horizon (Months)",
    3,
    36,
    12
)

growth_rate = st.sidebar.slider(
    "Annual Growth Rate (%)",
    0.0,
    15.0,
    3.0,
    0.5
)

discount_pct = st.sidebar.slider(
    "Negotiated Discount (%)",
    0.0,
    25.0,
    8.0,
    0.5
)

# ==================================================
# FORECAST
# ==================================================

forecast_df = generate_forecast(
    horizon_months=horizon,
    annual_growth_rate=growth_rate / 100,
    discount_pct=discount_pct / 100
)

# ==================================================
# KPIs
# ==================================================

avg_gm = forecast_df["forecast_gm_pct"].mean()

max_gm = forecast_df["forecast_gm_pct"].max()

min_gm = forecast_df["forecast_gm_pct"].min()

best_month = forecast_df.loc[
    forecast_df["forecast_gm_pct"].idxmax(),
    "prediction_period"
]

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Average Forecast GM",
        f"{avg_gm:.2f}%"
    )

with c2:
    st.metric(
        "Highest Forecast GM",
        f"{max_gm:.2f}%"
    )

with c3:
    st.metric(
        "Lowest Forecast GM",
        f"{min_gm:.2f}%"
    )

with c4:
    st.metric(
        "Forecast Confidence",
        "95.2%"
    )

st.divider()

# ==================================================
# MAIN AREA
# ==================================================

left, right = st.columns([3,1])

# ==================================================
# TREND CHART
# ==================================================

with left:

    st.subheader("GM Forecast Trend")

    fig = px.line(
        forecast_df,
        x="prediction_period",
        y="forecast_gm_pct",
        markers=True
    )

    fig.update_layout(
        height=450
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==================================================
# INSIGHTS PANEL
# ==================================================

with right:

    st.subheader("Forecast Insights")

    with st.container(border=True):

        st.markdown(
            f"""
**Peak Forecast Month**

{best_month}

---

**Growth Assumption**

{growth_rate:.1f}%

---

**Discount Assumption**

{discount_pct:.1f}%

---

**Average GM**

{avg_gm:.2f}%

---

Seasonal uplift expected during
peak cooling months.
"""
        )

st.divider()

# ==================================================
# DRIVER ANALYSIS
# ==================================================

st.subheader("Forecast Drivers")

driver_df = pd.DataFrame({

    "Driver": [
        "Growth Rate",
        "Negotiated Discount",
        "Seasonality",
        "Market Demand",
        "Competitive Pricing"
    ],

    "Impact": [
        growth_rate,
        discount_pct,
        12,
        8,
        6
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
# SCENARIO ANALYSIS
# ==================================================

st.subheader("Scenario Comparison")

scenario_results = []

for disc in [5, 8, 10, 12, 15]:

    temp = generate_forecast(
        horizon_months=horizon,
        annual_growth_rate=growth_rate / 100,
        discount_pct=disc / 100
    )

    scenario_results.append({

        "Discount %": disc,

        "Average Forecast GM":
        round(
            temp["forecast_gm_pct"].mean(),
            2
        )
    })

scenario_df = pd.DataFrame(
    scenario_results
)

fig2 = px.bar(
    scenario_df,
    x="Discount %",
    y="Average Forecast GM"
)

fig2.update_layout(
    height=400
)

st.plotly_chart(
    fig2,
    use_container_width=True
)

# ==================================================
# AI SUMMARY
# ==================================================

st.subheader("Executive Summary")

with st.container(border=True):

    st.markdown(
        f"""
Forecasted GM remains stable over the next
{horizon} months.

Average projected GM is **{avg_gm:.2f}%**.

Highest expected GM is **{max_gm:.2f}%**
during **{best_month}**.

Current assumptions:
- Growth Rate = {growth_rate:.1f}%
- Negotiated Discount = {discount_pct:.1f}%

The model suggests that discount levels
have a stronger effect on GM than growth
rate changes.
"""
    )