# =====================================================
# GM MARGIN PREDICTION CONFIGURATION
# =====================================================

# -----------------------------
# DATA
# -----------------------------

DATA_FILE = "carrier_hvac_global_v3_enriched.xlsx"

DATA_SHEET = "ml_training_dataset"

TARGET_COLUMN = "target_actual_gm_pct"

SPLIT_COLUMN = "dataset_split"


# -----------------------------
# MODEL ARTIFACTS
# -----------------------------

MODEL_FILE = "gm_margin_xgb_final.pkl"

PREPROCESSOR_FILE = "gm_margin_preprocessor_final.pkl"

FEATURE_FILE = "gm_margin_feature_order.pkl"

FEATURE_IMPORTANCE_FILE = "feature_importance.xlsx"


# -----------------------------
# RANDOM FOREST
# -----------------------------

RF_PARAMS = {

    "n_estimators": 100,

    "max_depth": 10,

    "min_samples_leaf": 5,

    "min_samples_split": 10,

    "random_state": 42,

    "n_jobs": -1
}


# -----------------------------
# FINAL XGBOOST
# -----------------------------

XGB_PARAMS = {

    "n_estimators": 600,

    "learning_rate": 0.03,

    "max_depth": 5,

    "subsample": 0.85,

    "colsample_bytree": 0.85,

    "min_child_weight": 3,

    "reg_alpha": 0.1,

    "reg_lambda": 1.5,

    "objective": "reg:squarederror",

    "random_state": 42,

    "n_jobs": -1
}


# -----------------------------
# LEAKAGE / EXCLUDED FEATURES
# -----------------------------

REMOVE_COLUMNS = [

    "ml_record_id",
    "dealer_id",
    "product_id",
    "geography_id",

    "target_actual_pocket_margin_pct",
    "target_discount_pct_actual",
    "target_gm_pct_product",

    "dealer_product_avg_gm_pct_6m",
    "dealer_vs_peer_gm_delta",

    "total_revenue_usd",
    "units_sold",
    "below_floor_count",

    "standard_gm_pct",
    "product_tier_enc",
    "product_tier",
    "product_series",
    "dealer_product_avg_discount_pct",

    "product_lifecycle_stage",
    "lifecycle_enc"
]


# -----------------------------
# REPORTING
# -----------------------------

RANDOM_STATE = 42

TOP_FEATURE_COUNT = 20