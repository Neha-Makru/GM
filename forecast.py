import pandas as pd
import numpy as np
import joblib

# =====================================================
# CONFIG
# =====================================================

DATA_FILE = "carrier_hvac_global_v3_enriched.xlsx"
DATA_SHEET = "ml_training_dataset"

MODEL_PATH = "gm_margin_xgb_final.pkl"
PREPROCESSOR_PATH = "gm_margin_preprocessor_final.pkl"
FEATURE_PATH = "gm_margin_feature_order.pkl"

# =====================================================
# LOAD ARTIFACTS
# =====================================================

model = joblib.load(MODEL_PATH)

preprocessor = joblib.load(
    PREPROCESSOR_PATH
)

feature_order = joblib.load(
    FEATURE_PATH
)

# =====================================================
# LOAD LATEST BUSINESS SNAPSHOT
# =====================================================

def get_latest_business_snapshot():

    df = pd.read_excel(
        DATA_FILE,
        sheet_name=DATA_SHEET
    )

    df["prediction_period"] = pd.to_datetime(
        df["prediction_period"],
        format="%Y-%m",
        errors="coerce"
    )

    latest_period = df[
        "prediction_period"
    ].max()

    latest_row = df[
        df["prediction_period"] == latest_period
    ].iloc[0]

    snapshot = latest_row.to_dict()

    return snapshot

# =====================================================
# FORECAST ENGINE
# =====================================================

def generate_forecast(
        horizon_months=12
):

    base_row = get_latest_business_snapshot()

    forecast_rows = []

    start_date = pd.Timestamp.today().replace(
        day=1
    )

    for i in range(horizon_months):

        row = base_row.copy()

        future_date = start_date + pd.DateOffset(
            months=i
        )

        row["prediction_period"] = (
            future_date.strftime("%Y-%m")
        )

        row["fiscal_year"] = future_date.year

        row["fiscal_month"] = future_date.month

        row["fiscal_quarter"] = (
            (future_date.month - 1)//3
        ) + 1

        # ==================================
        # GROWTH ASSUMPTIONS
        # ==================================

        growth_factor = (
            1.03
        ) ** (i / 12)

        if "annual_revenue_usd" in row:
            row["annual_revenue_usd"] *= growth_factor

        if "list_price_usd" in row:
            row["list_price_usd"] *= growth_factor

        if "competitor_avg_price_usd" in row:
            row["competitor_avg_price_usd"] *= growth_factor

        row["market_growth_rate_pct"] = 0.03

        # ==================================
        # SEASONALITY
        # ==================================

        if future_date.month in [5, 6, 7, 8]:

            row["seasonality_index"] = 1.20

            row["is_peak_season"] = 1

            row["cooling_season_flag"] = 1

        else:

            row["seasonality_index"] = 0.95

            row["is_peak_season"] = 0

            row["cooling_season_flag"] = 0

        forecast_rows.append(row)

    forecast_df = pd.DataFrame(
        forecast_rows
    )

    # ==================================
    # ALIGN TO TRAINING FEATURES
    # ==================================

    for col in feature_order:

        if col not in forecast_df.columns:
            forecast_df[col] = np.nan

    forecast_df = forecast_df[
        feature_order
    ]

    forecast_df = forecast_df.fillna(0)

    # ==================================
    # PREPROCESS
    # ==================================

    processed = preprocessor.transform(
        forecast_df
    )

    # ==================================
    # PREDICT
    # ==================================

    predictions = model.predict(
        processed
    )

    forecast_df[
        "forecast_gm_pct"
    ] = predictions * 100

    return forecast_df

# =====================================================
# TEST
# =====================================================

if __name__ == "__main__":

    forecast = generate_forecast()

    print(
        forecast[
            [
                "prediction_period",
                "forecast_gm_pct"
            ]
        ]
    )