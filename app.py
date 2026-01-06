import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# Config
PROJECT_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = PROJECT_DIR / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "model.joblib"
META_PATH = ARTIFACT_DIR / "metadata.json"


@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing {MODEL_PATH}. Run the last cell in the notebook to generate artifacts/."
        )
    if not META_PATH.exists():
        raise FileNotFoundError(
            f"Missing {META_PATH}. Run the last cell in the notebook to generate artifacts/."
        )
    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    return model, meta


def build_input_form(meta: dict) -> pd.DataFrame:
    st.subheader("Input features")

    numeric_features = meta["numeric_features"]
    categorical_features = meta["categorical_features"]
    categories = meta.get("categories", {})
    numeric_cfg = meta.get("numeric_input_config", {})

    values = {}

    # Categorical inputs
    for col in categorical_features:
        options = categories.get(col, [])
        # Fallback: allow free text if options were not saved
        if options:
            values[col] = st.selectbox(col, options=options, index=0)
        else:
            values[col] = st.text_input(col, value="")

    # Numeric inputs
    for col in numeric_features:
        cfg = numeric_cfg.get(col, {})
        vmin = float(cfg.get("min", 0.0))
        vmax = float(cfg.get("max", max(vmin + 1.0, 1.0)))
        vmed = float(cfg.get("median", (vmin + vmax) / 2.0))

        # Use int input for integer-like features (year/age/household_size)
        if float(vmin).is_integer() and float(vmax).is_integer():
            values[col] = st.number_input(col, min_value=int(vmin), max_value=int(vmax), value=int(vmed), step=1)
        else:
            values[col] = st.number_input(col, min_value=vmin, max_value=vmax, value=vmed)

    # Return a single-row DataFrame
    return pd.DataFrame([values])


def main():
    st.title("Financial Inclusion: Bank Account Prediction")
    st.write("Predict whether an individual is likely to have a bank account based on demographic features.")

    model, meta = load_artifacts()
    threshold = float(meta.get("decision_threshold", 0.5))

    with st.form("predict_form"):
        X_input = build_input_form(meta)
        submitted = st.form_submit_button("Validate & Predict")

    if submitted:
        # Model is a full preprocessing+classifier pipeline
        try:
            proba = float(model.predict_proba(X_input)[:, 1][0])
        except Exception:
            # Fallback if estimator doesn't expose predict_proba
            proba = float(model.decision_function(X_input)[0])
            # map to (0,1) via sigmoid-ish scaling
            proba = float(1 / (1 + np.exp(-proba)))

        pred_label = "Yes" if proba >= threshold else "No"

        st.subheader("Prediction")
        st.metric("Predicted bank account (Yes/No)", pred_label)
        st.metric("Predicted probability (Yes)", f"{proba:.3f}")
        st.caption(f"Decision threshold used: {threshold:.3f}")


if __name__ == "__main__":
    main()
