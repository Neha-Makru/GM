import pandas as pd
import numpy as np
import joblib

# =====================================================
# LOAD ARTIFACTS
# =====================================================

MODEL_PATH = "gm_margin_xgb_final.pkl"
PREPROCESSOR_PATH = "gm_margin_preprocessor_final.pkl"
FEATURE_PATH = "gm_margin_feature_order.pkl"

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)
feature_order = joblib.load(FEATURE_PATH)

# =====================================================
# DEFAULT BUSINESS SCENARIO
# =====================================================

DEFAULTS = {

    "prediction_period": "2025-06",

    "dealer_tier": "Gold",
    "dealer_type": "Distributor",
    "contract_type": "Annual",

    "category_l1": "Commercial HVAC",
    "category_l2": "Chiller",

    "country": "USA",
    "country_code": "US",

    "region": "North",
    "climate_zone": "Temperate",
    "market_density": "Urban",

    "peer_group_id": "PG001",

    "annual_units_sold": 1500,
    "annual_revenue_usd": 2500000,

    "premium_product_mix_pct": 0.35,
    "commercial_project_mix_pct": 0.20,

    "churn_risk_score": 0.10,
    "nps_score": 70,

    "dealer_tier_enc": 2,
    "region_enc": 1,
    "climate_enc": 1,

    "avg_gm_pct_3m": 0.37,
    "avg_gm_pct_6m": 0.37,
    "avg_gm_pct_12m": 0.37,

    "avg_discount_depth_pct": 0.10,
    "target_achievement_rate_12m": 0.95,

    "list_price_usd": 4500,
    "landed_cost_usd": 3200,

    "capacity_tr": 3.5,
    "efficiency_rating": 8,

    "wifi_enabled": 1,
    "is_bundle_eligible": 1,

    "days_in_market": 250,
    "inventory_weeks_on_hand": 4,

    "demand_trend_index": 1.1,

    "is_eol_product": 0,

    "dealer_product_units_sold": 350,

    "competitor_avg_price_usd": 4700,
    "price_elasticity_index": 0.8,
    "market_share_pct": 0.25,

    "demand_index": 1.05,
    "seasonality_index": 1.10,

    "cooling_degree_days": 1500,

    "market_growth_rate_pct": 0.04,

    "fiscal_year": 2025,
    "fiscal_quarter": 2,
    "fiscal_month": 6,

    "is_peak_season": 0,
    "days_to_quarter_end": 30,

    "is_promo_period": 0,
    "cooling_season_flag": 1,

    "peer_group_median_gm_pct": 0.37,
    "peer_group_avg_discount_pct": 0.11,

    "negotiated_discount_pct": 0.08,
    "off_invoice_pct": 0.02,
    "rebate_pct": 0.01,
    "coop_fund_pct": 0.01,
    "freight_pct": 0.03,
    "warranty_reserve_pct": 0.01,

    "inventory_weeks_on_hand_curr": 4,

    "txn_count": 40
}

# =====================================================
# FORECAST FUNCTION
# =====================================================

def generate_forecast(
        horizon_months=12,
        annual_growth_rate=0.03,
        discount_pct=0.08
):

    forecast_rows = []

    start_date = pd.Timestamp.today().replace(day=1)

    for i in range(horizon_months):

        row = DEFAULTS.copy()

        future_date = start_date + pd.DateOffset(months=i)

        row["prediction_period"] = future_date.strftime("%Y-%m")

        row["fiscal_year"] = future_date.year
        row["fiscal_month"] = future_date.month
        row["fiscal_quarter"] = ((future_date.month - 1)//3) + 1

        growth_factor = (1 + annual_growth_rate) ** (i/12)

        row["annual_revenue_usd"] *= growth_factor

        row["competitor_avg_price_usd"] *= growth_factor

        row["list_price_usd"] *= growth_factor

        row["market_growth_rate_pct"] = annual_growth_rate

        row["negotiated_discount_pct"] = discount_pct

        # ---------------------------------
        # Seasonality Logic
        # ---------------------------------

        if future_date.month in [5, 6, 7, 8]:

            row["seasonality_index"] = 1.20
            row["is_peak_season"] = 1

        else:

            row["seasonality_index"] = 0.95
            row["is_peak_season"] = 0

        forecast_rows.append(row)

    forecast_df = pd.DataFrame(forecast_rows)

    # =================================================
    # ALIGN TO TRAINING FEATURES
    # =================================================

    for col in feature_order:

        if col not in forecast_df.columns:
            forecast_df[col] = 0

    forecast_df = forecast_df[feature_order]

    # =================================================
    # PREPROCESS
    # =================================================

    processed = preprocessor.transform(forecast_df)

    # =================================================
    # PREDICT
    # =================================================

    predictions = model.predict(processed)

    results = forecast_df.copy()

    results["forecast_gm_pct"] = predictions * 100

    return results


# =====================================================
# TEST RUN
# =====================================================

if __name__ == "__main__":

    forecast = generate_forecast(
        horizon_months=12,
        annual_growth_rate=0.03,
        discount_pct=0.08
    )

    print("\nForecast Preview:\n")

    print(
        forecast[
            [
                "prediction_period",
                "forecast_gm_pct"
            ]
        ]
    )