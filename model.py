# =====================================================
# GM MARGIN PREDICTION MODEL
# =====================================================
#
# Purpose:
# Predict Gross Margin % for HVAC transactions
#
# Final Production Model:
# XGBoost Regressor
#
# Output Artifacts:
# - gm_margin_xgb_final.pkl
# - gm_margin_preprocessor_final.pkl
# - gm_margin_feature_order.pkl
#
# =====================================================

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)

# =====================================================
# LOAD DATA
# =====================================================

FILE_PATH = "carrier_hvac_global_v3_enriched.xlsx"

print("Loading dataset...")

df = pd.read_excel(
    FILE_PATH,
    sheet_name="ml_training_dataset"
)

print(f"Dataset Shape: {df.shape}")

# =====================================================
# BASIC DATA REVIEW
# =====================================================

print("\nDataset Info\n")

print(df.info())

print("\nTarget Summary\n")

print(
    df["target_actual_gm_pct"].describe()
)

print("\nDataset Split Distribution\n")

print(
    df["dataset_split"].value_counts()
)

# =====================================================
# TARGET DISTRIBUTION
# =====================================================

plt.figure(figsize=(8,4))

plt.hist(
    df["target_actual_gm_pct"],
    bins=40
)

plt.xlabel("Gross Margin %")

plt.ylabel("Count")

plt.title("Target Distribution")

plt.show()

# =====================================================
# REMOVE LEAKAGE FEATURES
# =====================================================
#
# These variables either:
#
# 1. Reveal the target directly
# 2. Contain future information
# 3. Are post-transaction measures
# 4. Were removed after feature-review
# 5. Were excluded during ablation testing
#
# Final deployed model excludes all columns below.
#
# =====================================================

REMOVE_COLUMNS = [

    # ----------------------------------
    # Technical Identifiers
    # ----------------------------------

    "ml_record_id",
    "dealer_id",
    "product_id",
    "geography_id",

    # ----------------------------------
    # Direct Target Leakage
    # ----------------------------------

    "target_actual_pocket_margin_pct",
    "target_discount_pct_actual",
    "target_gm_pct_product",

    # ----------------------------------
    # Historical GM Leakage
    # ----------------------------------

    "dealer_product_avg_gm_pct_6m",
    "dealer_vs_peer_gm_delta",

    # ----------------------------------
    # Post-Transaction Metrics
    # ----------------------------------

    "total_revenue_usd",
    "units_sold",
    "below_floor_count",

    # ----------------------------------
    # Dominant Business Features
    # Removed after importance review
    # ----------------------------------

    "standard_gm_pct",
    "product_tier_enc",
    "product_tier",
    "product_series",
    "dealer_product_avg_discount_pct",

    # ----------------------------------
    # Ablation Study Removal
    # ----------------------------------

    "product_lifecycle_stage",
    "lifecycle_enc"
]

df_model = df.drop(
    columns=REMOVE_COLUMNS,
    errors="ignore"
)

print(
    f"\nModel Dataset Shape: {df_model.shape}"
)

# =====================================================
# DEFINE TARGET
# =====================================================

TARGET = "target_actual_gm_pct"

X = df_model.drop(
    columns=[TARGET]
)

y = df_model[TARGET]

print(f"Feature Matrix : {X.shape}")
print(f"Target Vector  : {y.shape}")

# =====================================================
# TRAIN / VALIDATION / TEST SPLIT
# =====================================================
#
# Dataset already contains predefined splits.
#
# =====================================================

train_mask = (
    df_model["dataset_split"] == "Train"
)

val_mask = (
    df_model["dataset_split"] == "Validation"
)

test_mask = (
    df_model["dataset_split"] == "Test"
)

X_train = X[train_mask].drop(
    columns=["dataset_split"]
)

y_train = y[train_mask]

X_val = X[val_mask].drop(
    columns=["dataset_split"]
)

y_val = y[val_mask]

X_test = X[test_mask].drop(
    columns=["dataset_split"]
)

y_test = y[test_mask]

print("\nTrain Shape:", X_train.shape)
print("Validation Shape:", X_val.shape)
print("Test Shape:", X_test.shape)

# =====================================================
# PREPROCESSING
# =====================================================
#
# Categorical variables:
# One Hot Encoding
#
# Numeric variables:
# Passed through unchanged
#
# Tree-based models do not require scaling.
#
# =====================================================

categorical_cols = X_train.select_dtypes(
    include=["object"]
).columns.tolist()

numeric_cols = X_train.select_dtypes(
    exclude=["object"]
).columns.tolist()

print(
    f"\nCategorical Features: {len(categorical_cols)}"
)

print(
    f"Numeric Features: {len(numeric_cols)}"
)

preprocessor = ColumnTransformer(

    transformers=[

        (
            "cat",

            OneHotEncoder(
                handle_unknown="ignore"
            ),

            categorical_cols
        ),

        (
            "num",

            "passthrough",

            numeric_cols
        )
    ]
)

X_train_p = preprocessor.fit_transform(
    X_train
)

X_val_p = preprocessor.transform(
    X_val
)

X_test_p = preprocessor.transform(
    X_test
)

feature_names = (
    preprocessor
    .get_feature_names_out()
)

print(
    f"\nEngineered Features: {len(feature_names)}"
)

# =====================================================
# RANDOM FOREST BASELINE
# =====================================================
#
# Random Forest is used as a benchmark model.
#
# This helps determine whether XGBoost provides
# meaningful performance improvements.
#
# =====================================================

rf = RandomForestRegressor(

    n_estimators=100,

    max_depth=10,

    min_samples_leaf=5,

    min_samples_split=10,

    random_state=42,

    n_jobs=-1
)

print("\nTraining Random Forest...")

rf.fit(
    X_train_p,
    y_train
)

# =====================================================
# RANDOM FOREST EVALUATION
# =====================================================

rf_val_pred = rf.predict(
    X_val_p
)

rf_mae = mean_absolute_error(
    y_val,
    rf_val_pred
)

rf_rmse = np.sqrt(
    mean_squared_error(
        y_val,
        rf_val_pred
    )
)

rf_r2 = r2_score(
    y_val,
    rf_val_pred
)

rf_train_pred = rf.predict(
    X_train_p
)

rf_train_r2 = r2_score(
    y_train,
    rf_train_pred
)

print("\n==========================")
print("RANDOM FOREST RESULTS")
print("==========================")

print(f"Validation MAE  : {rf_mae:.6f}")
print(f"Validation RMSE : {rf_rmse:.6f}")
print(f"Validation R²   : {rf_r2:.6f}")
print(f"Training R²     : {rf_train_r2:.6f}")

# =====================================================
# FINAL XGBOOST MODEL
# =====================================================
#
# Tuned configuration selected after
# multiple experiments.
#
# This model is deployed in:
#
# - Streamlit App
# - Batch Inference
# - Validation Pipeline
# - Forecasting Engine
#
# =====================================================

xgb_final = XGBRegressor(

    n_estimators=600,

    learning_rate=0.03,

    max_depth=5,

    subsample=0.85,

    colsample_bytree=0.85,

    min_child_weight=3,

    reg_alpha=0.1,

    reg_lambda=1.5,

    objective="reg:squarederror",

    random_state=42,

    n_jobs=-1
)

print("\nTraining Final XGBoost Model...")

xgb_final.fit(

    X_train_p,

    y_train,

    eval_set=[

        (X_val_p, y_val)

    ],

    verbose=False
)

# =====================================================
# VALIDATION PERFORMANCE
# =====================================================

val_pred = xgb_final.predict(
    X_val_p
)

val_r2 = r2_score(
    y_val,
    val_pred
)

val_rmse = np.sqrt(
    mean_squared_error(
        y_val,
        val_pred
    )
)

val_mae = mean_absolute_error(
    y_val,
    val_pred
)

train_pred = xgb_final.predict(
    X_train_p
)

train_r2 = r2_score(
    y_train,
    train_pred
)

print("\n==========================")
print("XGBOOST VALIDATION RESULTS")
print("==========================")

print(f"Validation R²   : {val_r2:.6f}")
print(f"Validation RMSE : {val_rmse:.6f}")
print(f"Validation MAE  : {val_mae:.6f}")
print(f"Training R²     : {train_r2:.6f}")

# =====================================================
# TEST SET PERFORMANCE
# =====================================================
#
# Final performance on completely unseen data.
#
# This metric is used for project reporting.
#
# =====================================================

test_pred = xgb_final.predict(
    X_test_p
)

test_r2 = r2_score(
    y_test,
    test_pred
)

test_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        test_pred
    )
)

test_mae = mean_absolute_error(
    y_test,
    test_pred
)

print("\n==========================")
print("FINAL TEST RESULTS")
print("==========================")

print(f"Test R²   : {test_r2:.6f}")
print(f"Test RMSE : {test_rmse:.6f}")
print(f"Test MAE  : {test_mae:.6f}")

# =====================================================
# RESIDUAL ANALYSIS
# =====================================================
#
# Residuals = Actual - Predicted
#
# Ideally centered around zero.
#
# =====================================================

residuals = y_test - test_pred

plt.figure(figsize=(8,4))

plt.hist(
    residuals,
    bins=40
)

plt.xlabel("Residual")

plt.ylabel("Count")

plt.title(
    "Test Residual Distribution"
)

plt.show()

# =====================================================
# FEATURE IMPORTANCE
# =====================================================

importance_df = pd.DataFrame({

    "feature":
    preprocessor.get_feature_names_out(),

    "importance":
    xgb_final.feature_importances_
})

importance_df = importance_df.sort_values(

    "importance",

    ascending=False
)

print("\nTop 20 Features")

print(
    importance_df.head(20)
)

importance_df.to_excel(

    "feature_importance.xlsx",

    index=False
)

# =====================================================
# SAVE DEPLOYMENT ARTIFACTS
# =====================================================
#
# These files power:
#
# - app.py
# - inference_utils.py
# - validate_model.py
# - forecast.py
# - forecast_app.py
#
# =====================================================

joblib.dump(

    preprocessor,

    "gm_margin_preprocessor_final.pkl"
)

joblib.dump(

    xgb_final,

    "gm_margin_xgb_final.pkl"
)

joblib.dump(

    X_train.columns.tolist(),

    "gm_margin_feature_order.pkl"
)

print("\nArtifacts Saved Successfully")

# =====================================================
# TRAINING SUMMARY
# =====================================================

print("\n==========================")
print("PROJECT SUMMARY")
print("==========================")

print(f"Training Records   : {len(X_train)}")
print(f"Validation Records : {len(X_val)}")
print(f"Test Records       : {len(X_test)}")

print(f"\nFinal Test R²      : {test_r2:.4f}")
print(f"Final Test RMSE    : {test_rmse:.4f}")
print(f"Final Test MAE     : {test_mae:.4f}")

print("\nModel Saved:")
print("gm_margin_xgb_final.pkl")

print("\nPreprocessor Saved:")
print("gm_margin_preprocessor_final.pkl")

print("\nFeature Schema Saved:")
print("gm_margin_feature_order.pkl")