import pandas as pd
import joblib


# ==========================================
# LOAD SAVED ARTIFACTS
# ==========================================

print("Loading artifacts...")

model = joblib.load(
    "gm_margin_xgb_final.pkl"
)

preprocessor = joblib.load(
    "gm_margin_preprocessor_final.pkl"
)

feature_order = joblib.load(
    "gm_margin_feature_order.pkl"
)

print("Artifacts Loaded")


# ==========================================
# DEFINE CATEGORICAL FEATURES
# ==========================================

categorical_features = [

    "prediction_period",

    "dealer_tier",

    "dealer_type",

    "contract_type",

    "product_tier",

    "product_series",

    "category_l1",

    "category_l2",

    "country",

    "country_code",

    "region",

    "climate_zone",

    "market_density",

    "peer_group_id"

]


# ==========================================
# INITIALIZE INPUT DICTIONARY
# ==========================================

input_dict = {}

for feature in feature_order:

    if feature in categorical_features:

        input_dict[feature] = "Unknown"

    else:

        input_dict[feature] = 0


# ==========================================
# OVERRIDE WITH SAMPLE VALUES
# Modify these however you want
# ==========================================

input_dict["prediction_period"] = "2025-06"

input_dict["dealer_tier"] = "Gold"

input_dict["dealer_type"] = "Distributor"

input_dict["contract_type"] = "Annual"

input_dict["product_tier"] = "Premium"

input_dict["product_series"] = "Infinity"

input_dict["category_l1"] = "Commercial HVAC"

input_dict["category_l2"] = "Chiller"

input_dict["country"] = "USA"

input_dict["country_code"] = "US"

input_dict["region"] = "North"

input_dict["climate_zone"] = "Temperate"

input_dict["market_density"] = "Urban"

input_dict["peer_group_id"] = "PG001"


# numerical features

input_dict["annual_units_sold"] = 1500

input_dict["annual_revenue_usd"] = 2500000

input_dict["premium_product_mix_pct"] = 0.35

input_dict["commercial_project_mix_pct"] = 0.20

input_dict["churn_risk_score"] = 0.10

input_dict["nps_score"] = 70

input_dict["list_price_usd"] = 4500

input_dict["landed_cost_usd"] = 3200

input_dict["capacity_tr"] = 3.5

input_dict["efficiency_rating"] = 8

input_dict["wifi_enabled"] = 1

input_dict["is_bundle_eligible"] = 1

input_dict["days_in_market"] = 250

input_dict["inventory_weeks_on_hand"] = 4

input_dict["demand_trend_index"] = 1.1

input_dict["competitor_avg_price_usd"] = 4700

input_dict["price_elasticity_index"] = 0.8

input_dict["market_share_pct"] = 0.25

input_dict["demand_index"] = 1.05

input_dict["seasonality_index"] = 1.1

input_dict["cooling_degree_days"] = 1500

input_dict["market_growth_rate_pct"] = 0.04

input_dict["fiscal_year"] = 2025

input_dict["fiscal_quarter"] = 2

input_dict["fiscal_month"] = 6

input_dict["is_peak_season"] = 1

input_dict["days_to_quarter_end"] = 30

input_dict["is_promo_period"] = 0

input_dict["cooling_season_flag"] = 1

input_dict["peer_group_median_gm_pct"] = 0.37

input_dict["peer_group_avg_discount_pct"] = 0.11

input_dict["negotiated_discount_pct"] = 0.08

input_dict["off_invoice_pct"] = 0.02

input_dict["rebate_pct"] = 0.01

input_dict["coop_fund_pct"] = 0.01

input_dict["freight_pct"] = 0.03

input_dict["warranty_reserve_pct"] = 0.01

input_dict["inventory_weeks_on_hand_curr"] = 4

input_dict["units_sold"] = 350

input_dict["txn_count"] = 40

input_dict["below_floor_count"] = 2


# ==========================================
# BUILD DATAFRAME
# ==========================================

new_data = pd.DataFrame(
    [input_dict]
)

new_data = new_data[
    feature_order
]

print("\nInput Data Ready")

print(
    new_data.head()
)


# ==========================================
# PREPROCESS
# ==========================================

processed = preprocessor.transform(
    new_data
)


# ==========================================
# PREDICT
# ==========================================

prediction = model.predict(
    processed
)


# ==========================================
# OUTPUT
# ==========================================

predicted_margin = prediction[0] * 100

print("\n======================")

print(
    f"Predicted GM Margin: {predicted_margin:.2f}%"
)

print("======================")