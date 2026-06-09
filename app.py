import streamlit as st
import pandas as pd
import io

from inference_utils import (
    create_feature_row,
    predict_df
)

st.set_page_config(
    page_title="GM Margin Predictor",
    layout="wide"
)

st.title("GM Margin Prediction Dashboard")

tab1, tab2 = st.tabs(
    ["Single Prediction", "Batch Prediction"]
)

# ==================================================
# TAB 1
# ==================================================

with tab1:

    st.subheader("Single Prediction")

    dealer_tier = st.selectbox(
        "Dealer Tier",
        ["Silver", "Gold", "Platinum"]
    )

    dealer_type = st.selectbox(
        "Dealer Type",
        ["Distributor", "Retailer"]
    )

    category = st.selectbox(
        "Category",
        [
            "Commercial HVAC",
            "Refrigeration",
            "Indoor Air Quality",
            "Thermostat"
        ]
    )

    list_price = st.number_input(
        "List Price",
        value=4500.0
    )

    landed_cost = st.number_input(
        "Landed Cost",
        value=3200.0
    )

    discount = st.slider(
        "Discount %",
        0.0,
        0.30,
        0.08
    )

    if st.button("Predict"):

        row = create_feature_row({

            "dealer_tier": dealer_tier,

            "dealer_type": dealer_type,

            "category_l1": category,

            "list_price_usd": list_price,

            "landed_cost_usd": landed_cost,

            "negotiated_discount_pct": discount

        })

        df = pd.DataFrame([row])

        result_df, _, _ = predict_df(df)

        prediction = result_df[
            "predicted_gm_margin_pct"
        ].iloc[0]

        st.success(
            f"Predicted GM Margin: {prediction:.2f}%"
        )

        excel_buffer = io.BytesIO()

        with pd.ExcelWriter(
            excel_buffer,
            engine="openpyxl"
        ) as writer:

            result_df.to_excel(
                writer,
                index=False
            )

        st.download_button(
            "Download Prediction",
            excel_buffer.getvalue(),
            file_name="prediction.xlsx"
        )

# ==================================================
# TAB 2
# ==================================================

with tab2:

    st.subheader("Batch Prediction")

    st.info(
        "Required columns: dealer_tier, dealer_type, category_l1, list_price_usd, landed_cost_usd, negotiated_discount_pct"
    )

    uploaded_file = st.file_uploader(
        "Upload Excel File",
        type=["xlsx"]
    )

    if uploaded_file:

        df = pd.read_excel(
            uploaded_file
        )

        required_cols = [
            "dealer_tier",
            "dealer_type",
            "category_l1",
            "list_price_usd",
            "landed_cost_usd",
            "negotiated_discount_pct"
        ]

        missing_required = [

            c

            for c in required_cols

            if c not in df.columns

        ]

        if missing_required:

            st.error(
                f"Missing required columns: {missing_required}"
            )

        else:

            result_df, created_cols, filled_vals = predict_df(df)

            st.success(
                f"Rows processed: {len(result_df)}"
            )

            st.write(
                f"Missing columns created: {created_cols}"
            )

            st.write(
                f"Missing values filled: {filled_vals}"
            )

            st.dataframe(
                result_df.head()
            )

            excel_buffer = io.BytesIO()

            with pd.ExcelWriter(
                excel_buffer,
                engine="openpyxl"
            ) as writer:

                result_df.to_excel(
                    writer,
                    index=False
                )

            st.download_button(
                "Download Predictions",
                excel_buffer.getvalue(),
                file_name="batch_predictions.xlsx"
            )