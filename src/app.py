"""
Streamlit app for single and batch predictions.

Run:
    streamlit run src.app
"""

# --- Ensures project root is on sys.path ---
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # parent of src/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from typing import Any, Dict

import pandas as pd
import streamlit as st

from src.config import DATA_FILE, TARGET_COL
from src.deployment import predict_single, predict_batch
from src.preprocessing import data_load



@st.cache_data
def load_sample_data() -> pd.DataFrame:
    return data_load(DATA_FILE)


def build_input_form(df_sample: pd.DataFrame) -> Dict[str, Any]:
    feature_cols = [c for c in df_sample.columns if c != TARGET_COL]
    input_data: Dict[str, Any] = {}
    for col in feature_cols:
        series = df_sample[col]
        if pd.api.types.is_numeric_dtype(series):
            default = float(series.median()) if not series.isna().all() else 0.0
            input_data[col] = st.number_input(label=col, value=default)
        else:
            options = series.dropna().unique().tolist()
            if not options:
                options = ["Unknown"]
            input_data[col] = st.selectbox(label=col, options=options, index=0)
    return input_data


def main() -> None:
    st.title("Customer Churn Prediction")
    st.markdown("Model trained with pipelines (imputer + scaler for numeric, imputer + OHE for categorical).")

    df_sample = load_sample_data()

    tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

    with tab1:
        input_data = build_input_form(df_sample)
        if st.button("Predict Churn"):
            result = predict_single(input_data)
            st.write("**Prediction:**", result["prediction"])
            st.write("**Churn probability:**", f"{result['churn_probability']:.4f}")
            if result["churn_probability"] > 0.5:
                st.error("High risk of churn")
            else:
                st.success("Low risk of churn")

    with tab2:
        st.subheader("Upload CSV for batch prediction")
        uploaded = st.file_uploader("Choose cleaned CSV", type=["csv"])
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            st.write("Preview")
            st.dataframe(df.head())
            if st.button("Run batch prediction"):
                result_df = predict_batch(df)
                st.dataframe(result_df.head())
                csv = result_df.to_csv(index=False).encode("utf-8")
                st.download_button("Download predictions", data=csv, file_name="churn_predictions.csv", mime="text/csv")


if __name__ == "__main__":
    main()
