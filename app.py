"""
Telecom Customer Churn Prediction - Demo App
Srikakulam District Survey (Bavajipeta / Dusipeta / Dusi villages)

This Streamlit app trains a Logistic Regression model on the synthetic
dataset (built to match the real survey statistics) and lets you enter
a hypothetical customer's details to get a churn prediction.
"""

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Telecom Churn Predictor",
    page_icon="📶",
    layout="centered",
)

st.title("📶 Telecom Customer Churn Predictor")
st.caption(
    "Based on an 82-user field survey in Bavajipeta, Dusipeta, and Dusi "
    "villages, Srikakulam district, Andhra Pradesh (2023-2026)."
)


# ----------------------------------------------------------------------
# Load data & train model (cached so it only runs once)
# ----------------------------------------------------------------------
@st.cache_resource
def load_model():
    df = pd.read_csv("telecom_synthetic_dataset.csv")

    X = df.drop(columns=["user_id", "churned", "complaint_count"])
    y = df["churned"]

    categorical_cols = ["operator", "plan_type", "years_using_sim", "age_group"]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ],
        remainder="passthrough",
    )

    model = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])

    model.fit(X_train, y_train)
    test_accuracy = accuracy_score(y_test, model.predict(X_test))

    return model, test_accuracy, numeric_cols, categorical_cols


model, test_accuracy, numeric_cols, categorical_cols = load_model()

st.sidebar.header("About this model")
st.sidebar.write(
    "**Algorithm:** Logistic Regression\n\n"
    f"**Held-out test accuracy:** {test_accuracy:.0%}\n\n"
    "**Data:** Synthetic 82-row dataset constructed to match the "
    "percentages reported in the original field survey. "
    "See the project README for full details and limitations."
)


# ----------------------------------------------------------------------
# Input form
# ----------------------------------------------------------------------
st.subheader("Enter customer details")

col1, col2 = st.columns(2)

with col1:
    operator = st.selectbox("Operator", ["Jio", "Airtel", "BSNL"])
    plan_type = st.selectbox("Plan type", ["5G", "4G"])
    age_group = st.selectbox("Age group", ["18-25", "26-35", "36-50", "50+"])
    years_using_sim = st.selectbox("Years using this SIM", ["<1", "1-3", "3-5", "5+"])
    download_speed_mbps = st.slider(
        "Download speed (Mbps)", min_value=0.0, max_value=25.0, value=5.0, step=0.1
    )

with col2:
    st.markdown("**Complaint indicators**")
    got_4g_instead_of_5g = st.checkbox("Paying for 5G but getting 4G-level speed")
    post_expiry_calls_3x = st.checkbox("Receives 3+ promo calls/day after balance expiry")
    signal_problems = st.checkbox("Frequent signal / network problems")
    sits_under_sunlight_for_network = st.checkbox("Has to go outside / under sunlight for signal")
    phonepe_extra_charge = st.checkbox("Charged ₹3 extra for PhonePe recharge")
    call_cut_zero_balance = st.checkbox("Incoming calls cut soon after zero balance")
    no_1gb_per_day_plan = st.checkbox("No 1GB/day plan available")
    data_loan_available = st.checkbox("Data loan facility available")

st.divider()

if st.button("Predict churn risk", type="primary", use_container_width=True):
    input_dict = {
        "operator": operator,
        "plan_type": plan_type,
        "download_speed_mbps": download_speed_mbps,
        "got_4g_instead_of_5g": int(got_4g_instead_of_5g),
        "post_expiry_calls_3x": int(post_expiry_calls_3x),
        "signal_problems": int(signal_problems),
        "sits_under_sunlight_for_network": int(sits_under_sunlight_for_network),
        "phonepe_extra_charge": int(phonepe_extra_charge),
        "call_cut_zero_balance": int(call_cut_zero_balance),
        "no_1gb_per_day_plan": int(no_1gb_per_day_plan),
        "data_loan_available": int(data_loan_available),
        "years_using_sim": years_using_sim,
        "age_group": age_group,
    }

    input_df = pd.DataFrame([input_dict])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    churn_prob = probability[1]

    if prediction == 1:
        st.error(f"⚠️ **Likely to CHURN** (probability: {churn_prob:.0%})")
    else:
        st.success(f"✅ **Likely to STAY** (churn probability: {churn_prob:.0%})")

    st.progress(float(churn_prob))

    st.caption(
        "Note: This prediction comes from a model trained on a small, "
        "synthetic dataset built to mirror survey-level statistics. "
        "It is a demonstration of the methodology, not a production-grade "
        "prediction system."
    )

st.divider()
st.caption(
    "Project: Telecom Customer Churn Prediction (Srikakulam District Survey) | "
    "Author: Kranthi Kumar Palina"
)
