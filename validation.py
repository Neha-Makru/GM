import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

# =====================================================
# LOAD DATA
# =====================================================

DATA_PATH = "ml_training_dataset.xlsx"

print("Loading Dataset...")

df = pd.read_excel(DATA_PATH)

print(df.shape)

# =====================================================
# LOAD ARTIFACTS
# =====================================================

print("\nLoading Artifacts...")

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

# =====================================================
# TARGET
# =====================================================

TARGET = "target_actual_gm_pct"

# =====================================================
# USE TEST DATA ONLY
# =====================================================

test_df = df[
    df["dataset_split"] == "Test"
].copy()

print(
    "\nTest Records:",
    len(test_df)
)

# =====================================================
# PREPARE FEATURES
# =====================================================

X_test = test_df[
    feature_order
].copy()

y_test = test_df[
    TARGET
].copy()

# =====================================================
# TRANSFORM
# =====================================================

print("\nPreprocessing...")

X_test_processed = preprocessor.transform(
    X_test
)

# =====================================================
# PREDICT
# =====================================================

print("\nRunning Predictions...")

preds = model.predict(
    X_test_processed
)

# =====================================================
# METRICS
# =====================================================

r2 = r2_score(
    y_test,
    preds
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        preds
    )
)

mae = mean_absolute_error(
    y_test,
    preds
)

print("\n==========================")
print("MODEL VALIDATION RESULTS")
print("==========================")

print(
    f"R²   : {r2:.6f}"
)

print(
    f"RMSE : {rmse:.6f}"
)

print(
    f"MAE  : {mae:.6f}"
)

# =====================================================
# SAVE PREDICTIONS
# =====================================================

results = test_df.copy()

results["actual_gm_pct"] = y_test

results["predicted_gm_pct"] = preds

results["absolute_error"] = (
    abs(
        results["actual_gm_pct"]
        -
        results["predicted_gm_pct"]
    )
)

results.to_excel(
    "validation_results.xlsx",
    index=False
)

print(
    "\nSaved: validation_results.xlsx"
)

# =====================================================
# FEATURE IMPORTANCE
# =====================================================

try:

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    importance_df = pd.DataFrame({

        "feature": feature_names,

        "importance":
        model.feature_importances_
    })

    importance_df = (
        importance_df
        .sort_values(
            "importance",
            ascending=False
        )
    )

    importance_df.to_excel(
        "feature_importance.xlsx",
        index=False
    )

    print(
        "Saved: feature_importance.xlsx"
    )

except Exception as e:

    print(
        "\nFeature importance export skipped."
    )

# =====================================================
# ERROR DISTRIBUTION
# =====================================================

results["error_pct_points"] = (
    (
        results["predicted_gm_pct"]
        -
        results["actual_gm_pct"]
    ) * 100
)

print("\nError Summary")

print(
    results[
        "error_pct_points"
    ].describe()
)