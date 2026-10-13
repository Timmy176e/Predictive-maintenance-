"""
Streamlit app: AI4I 2020 predictive-maintenance failure-risk estimator.

Loads the trained Random Forest pipeline from model_artifacts/ and lets
someone enter a machine's sensor readings to get a failure probability.

Run locally:
    streamlit run app.py
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st
import joblib

APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "model_artifacts" / "random_forest_pipeline.joblib"
METADATA_PATH = APP_DIR / "model_artifacts" / "model_metadata.json"

st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="🛠️",
    layout="centered",
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    if METADATA_PATH.exists():
        with open(METADATA_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


# ---- Load artifacts -----------------------------------------------------

if not MODEL_PATH.exists():
    st.error(
        "No trained model found at `model_artifacts/random_forest_pipeline.joblib`.\n\n"
        "Run `python train_model.py` first (or copy the artifacts you already "
        "saved to Google Drive into the `model_artifacts/` folder), then restart "
        "the app."
    )
    st.stop()

model = load_model()
metadata = load_metadata()

preprocessor = model.named_steps["preprocessor"]
feature_names = list(preprocessor.feature_names_in_)
cat_encoder = preprocessor.named_transformers_["categorical"]
type_categories = list(cat_encoder.categories_[0])

threshold = float(metadata.get("classification_threshold", 0.50))


def guess_column(feature_names, *keywords):
    """Find the trained column name that contains one of the given keywords,
    so the UI doesn't break if the exact column spelling (e.g. units in
    brackets) differs slightly from what's hard-coded here."""
    for col in feature_names:
        lc = col.lower()
        if any(kw in lc for kw in keywords):
            return col
    return None


air_col = guess_column(feature_names, "air")
process_col = guess_column(feature_names, "process")
speed_col = guess_column(feature_names, "rotation", "speed", "rpm")
torque_col = guess_column(feature_names, "torque")
wear_col = guess_column(feature_names, "wear")

# ---- UI -------------------------------------------------------------------

st.title("🛠️ Predictive Maintenance")
st.caption(
    f"{metadata.get('model', 'Random Forest Classifier')} trained on the "
    f"{metadata.get('dataset', 'AI4I 2020 Predictive Maintenance Dataset')}"
)

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        machine_type = st.selectbox(
            "Product quality type",
            type_categories,
            help="L = Low, M = Medium, H = High quality variant",
        )
        air_temp = st.number_input(
            "Air temperature [K]", min_value=0.0, value=300.0, step=0.1, format="%.1f"
        )
        process_temp = st.number_input(
            "Process temperature [K]", min_value=0.0, value=310.0, step=0.1, format="%.1f"
        )

    with col2:
        rpm = st.number_input(
            "Rotational speed [rpm]", min_value=0, value=1500, step=10
        )
        torque = st.number_input(
            "Torque [Nm]", min_value=0.0, value=40.0, step=0.1, format="%.1f"
        )
        tool_wear = st.number_input(
            "Tool wear [min]", min_value=0, value=100, step=1
        )

    submitted = st.form_submit_button("Predict failure risk", use_container_width=True)

if submitted:
    row = {"Type": machine_type}
    if air_col:
        row[air_col] = air_temp
    if process_col:
        row[process_col] = process_temp
    if speed_col:
        row[speed_col] = rpm
    if torque_col:
        row[torque_col] = torque
    if wear_col:
        row[wear_col] = tool_wear

    missing = [c for c in feature_names if c not in row]
    if missing:
        st.error(f"Couldn't map these trained features to a form field: {missing}")
        st.stop()

    input_df = pd.DataFrame([row])[feature_names]
    proba = float(model.predict_proba(input_df)[0, 1])
    is_high_risk = proba >= threshold

    st.divider()
    st.metric("Estimated failure probability", f"{proba:.1%}")
    st.progress(min(proba, 1.0))

    if is_high_risk:
        st.error(
            f"⚠️ **High risk** — probability is at or above the "
            f"{threshold:.0%} alert threshold. Flag this machine for inspection."
        )
    else:
        st.success(
            f"✅ **Low risk** — probability is below the {threshold:.0%} alert threshold."
        )

with st.expander("About this model"):
    if metadata:
        st.write(f"**Target:** {metadata.get('target', 'Machine failure')}")
        st.write(f"**Decision threshold:** {threshold:.0%}")
        tm = metadata.get("test_metrics", {})
        if tm:
            c1, c2, c3 = st.columns(3)
            c1.metric("Precision", f"{tm.get('precision_failure', 0):.0%}")
            c2.metric("Recall", f"{tm.get('recall_failure', 0):.0%}")
            c3.metric("ROC-AUC", f"{tm.get('roc_auc', 0):.3f}")
            st.caption(
                f"Measured on a held-out test set — "
                f"{tm.get('true_positive', '?')} true positives, "
                f"{tm.get('false_negative', '?')} missed failures, "
                f"{tm.get('false_positive', '?')} false alarms."
            )
        limitations = metadata.get("limitations", [])
        if limitations:
            st.markdown("**Limitations**")
            for lim in limitations:
                st.caption(f"• {lim}")
    else:
        st.write("No `model_metadata.json` found alongside the model.")
