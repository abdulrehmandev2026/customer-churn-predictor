import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# Load model, scaler and feature lists
model = joblib.load("churn_model.joblib")
scaler = joblib.load("scaler.joblib")
feature_columns = joblib.load("feature_columns.joblib")
numeric_cols = joblib.load("numeric_cols.joblib")

st.title("📉 Customer Churn Predictor")
st.write("Enter customer details — the model will predict the churn risk. This is for learning purposes only, not business advice.")

# Show date and time in the sidebar
now = datetime.now()
st.sidebar.write("📅 Date:", now.strftime("%d %b %Y"))
st.sidebar.write("🕐 Time:", now.strftime("%I:%M %p"))

# Prediction history list — create empty on first run
if "history" not in st.session_state:
    st.session_state.history = []

# Inputs — values match the dataset exactly
gender = st.selectbox("Gender", ["Female", "Male"])
senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
partner = st.selectbox("Partner", ["No", "Yes"])
dependents = st.selectbox("Dependents", ["No", "Yes"])
tenure = st.slider("Tenure (months)", 0, 72, 12)
phone = st.selectbox("Phone Service", ["No", "Yes"])
multiple_lines = st.selectbox("Multiple Lines", ["No", "No phone service", "Yes"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["No", "No internet service", "Yes"])
online_backup = st.selectbox("Online Backup", ["No", "No internet service", "Yes"])
device_protection = st.selectbox("Device Protection", ["No", "No internet service", "Yes"])
tech_support = st.selectbox("Tech Support", ["No", "No internet service", "Yes"])
streaming_tv = st.selectbox("Streaming TV", ["No", "No internet service", "Yes"])
streaming_movies = st.selectbox("Streaming Movies", ["No", "No internet service", "Yes"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless = st.selectbox("Paperless Billing", ["No", "Yes"])
payment = st.selectbox("Payment Method", ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"])
monthly = st.slider("Monthly Charges ($)", 18.0, 119.0, 65.0)
total = st.slider("Total Charges ($)", 18.0, 8700.0, 1000.0)

if st.button("Predict Churn Risk"):
    data = pd.DataFrame([{
        "gender": gender, "SeniorCitizen": senior, "Partner": partner,
        "Dependents": dependents, "tenure": tenure, "PhoneService": phone,
        "MultipleLines": multiple_lines, "InternetService": internet,
        "OnlineSecurity": online_security, "OnlineBackup": online_backup,
        "DeviceProtection": device_protection, "TechSupport": tech_support,
        "StreamingTV": streaming_tv, "StreamingMovies": streaming_movies,
        "Contract": contract, "PaperlessBilling": paperless,
        "PaymentMethod": payment, "MonthlyCharges": monthly,
        "TotalCharges": total,
    }])
    data = pd.get_dummies(data)
    data = data.reindex(columns=feature_columns, fill_value=0)
    data[numeric_cols] = scaler.transform(data[numeric_cols])
    risk = model.predict_proba(data)[0][1] * 100

    st.write(f"Churn Risk: **{risk:.1f}%**")
    st.progress(int(risk))

    if risk < 30:
        band = "Low"
        st.success("🟢 Low Risk — this customer is likely to stay.")
    elif risk < 70:
        band = "Moderate"
        st.warning("🟡 Moderate Risk — offer a retention deal or better plan.")
    else:
        band = "High"
        st.error("🔴 High Risk — act now, this customer is likely to leave.")

    # Save this prediction in history
    st.session_state.history.append({
        "Time": datetime.now().strftime("%I:%M %p"),
        "Tenure": tenure,
        "Contract": contract,
        "Monthly $": monthly,
        "Risk %": round(risk, 1),
        "Band": band,
    })

# Show prediction history
st.subheader("📋 Prediction History")
if st.session_state.history:
    st.dataframe(pd.DataFrame(st.session_state.history))
else:
    st.write("No predictions yet — make your first prediction.")
