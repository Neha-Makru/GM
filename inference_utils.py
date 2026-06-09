import joblib
import pandas as pd

MODEL = joblib.load("gm_margin_xgb_final.pkl")
PREPROCESSOR = joblib.load("gm_margin_preprocessor_final.pkl")
FEATURE_ORDER = joblib.load("gm_margin_feature_order.pkl")

CATEGORICAL_FEATURES = [
    "prediction_period",
    "dealer_tier",
    "dealer_type",
    "contract_type",
    "category_l1",
    "category_l2",
    "country",
    "country_code",
    "region",
    "climate_zone",
    "market_density",
    "peer_group_id"
]

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
    "list_price_usd": 4500,
    "landed_cost_usd": 3200,
    "capacity_tr": 3.5,
    "efficiency_rating": 8,
    "wifi_enabled": 1,
    "is_bundle_eligible": 1,
    "days_in_market": 250,
    "inventory_weeks_on_hand": 4,
    "demand_trend_index": 1.1,
    "competitor_avg_price_usd": 4700,
    "price_elasticity_index": 0.8,
    "market_share_pct": 0.25,
    "demand_index": 1.05,
    "seasonality_index": 1.1,
    "cooling_degree_days": 1500,
    "market_growth_rate_pct": 0.04,
    "fiscal_year": 2025,
    "fiscal_quarter": 2,
    "fiscal_month": 6,
    "is_peak_season": 1,
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


def create_feature_row(overrides=None):

    row = {}

    for col in FEATURE_ORDER:

        if col in DEFAULTS:
            row[col] = DEFAULTS[col]

        elif col in CATEGORICAL_FEATURES:
            row[col] = "Unknown"

        else:
            row[col] = 0

    if overrides:
        row.update(overrides)

    return row


def predict_df(df):

    df = df.copy()

    missing_cols = []

    filled_values = 0

    for col in FEATURE_ORDER:

        if col not in df.columns:

            missing_cols.append(col)

            if col in DEFAULTS:
                df[col] = DEFAULTS[col]

            elif col in CATEGORICAL_FEATURES:
                df[col] = "Unknown"

            else:
                df[col] = 0

    for col in FEATURE_ORDER:

        before = df[col].isna().sum()

        if col in DEFAULTS:
            df[col] = df[col].fillna(DEFAULTS[col])

        elif col in CATEGORICAL_FEATURES:
            df[col] = df[col].fillna("Unknown")

        else:
            df[col] = df[col].fillna(0)

        filled_values += before

    df = df[FEATURE_ORDER]

    processed = PREPROCESSOR.transform(df)

    preds = MODEL.predict(processed)

    df["predicted_gm_margin_pct"] = preds * 100

    return df, len(missing_cols), filled_values