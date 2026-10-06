import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Diabetes Risk Prediction", page_icon="🩺")

model = joblib.load("diabetes_risk_model.pkl")
features = joblib.load("diabetes_risk_features.pkl")

st.title("🩺 Diabetes Risk Prediction")
st.write("Enter patient details to predict diabetes risk.")

c1, c2, c3 = st.columns(3)

with c1:
    age = st.number_input("Age", 18, 80, 40)
    gender = st.selectbox("Gender", ["Male", "Female"])
    bmi = st.number_input("BMI", 15.0, 40.8, 23.0, 0.1)
    sleep = st.number_input("Hours Sleep", 3.0, 12.0, 7.0, 0.1)
    stress = st.slider("Stress Level", 1, 10, 5)

with c2:
    sugar = st.number_input("Fasting Blood Sugar", 72, 400, 100)
    hba1c = st.number_input("HbA1c Level", 4.8, 13.4, 5.5, 0.1)
    systolic = st.number_input("Systolic BP", 80, 200, 120)
    diastolic = st.number_input("Diastolic BP", 50, 120, 80)
    waist = st.number_input("Waist Circumference (cm)", 73.2, 148.2, 90.0, 0.1)

with c3:
    family = st.selectbox("Family History", ["Yes", "No"])
    activity = st.selectbox("Physical Activity", ["Low", "Moderate", "High"])
    diet = st.selectbox("Diet Type", ["Healthy", "Moderate", "Unhealthy"])
    smoking = st.selectbox("Smoking Status", ["Never", "Former", "Current"])
    alcohol = st.selectbox("Alcohol Consumption", ["None", "Moderate", "High"])
    income = st.selectbox("Income Bracket", ["Low", "Middle", "High"])
    city = st.text_input("City", "Hyderabad")

def bmi_cat(x):
    if x < 18.5: return 0
    if x < 25: return 1
    if x < 30: return 2
    if x < 35: return 3
    if x < 40: return 4
    return 5

def hba1c_cat(x):
    if x < 5.7: return 0
    if x < 6.5: return 1
    return 2

def bp_cat(s, d):
    if s > 180 or d > 120: return 5
    if s >= 140 or d >= 90: return 4
    if s >= 130 or d >= 80: return 3
    if s >= 120 and d < 80: return 2
    if s < 90 and d < 60: return 0
    return 1

if st.button("🔍 Predict Diabetes Risk", use_container_width=True):

    data = pd.DataFrame({
        "age": [age],
        "bmi": [bmi],
        "hours_sleep_per_night": [sleep],
        "stress_level": [stress],
        "fasting_blood_sugar": [sugar],
        "hba1c_level": [hba1c],
        "blood_pressure_systolic": [systolic],
        "blood_pressure_diastolic": [diastolic],
        "waist_circumference_cm": [waist],
        "gender": [gender],
        "city": [city],
        "family_history_diabetes": [family],
        "physical_activity_level": [activity],
        "diet_type": [diet],
        "smoking_status": [smoking],
        "alcohol_consumption": [alcohol],
        "income_bracket": [income],
        "bmi_category": [bmi_cat(bmi)],
        "hba1c_category": [hba1c_cat(hba1c)],
        "bp_category": [bp_cat(systolic, diastolic)]
    })

    data = pd.get_dummies(data, drop_first=True, dtype=int)
    data = data.reindex(columns=features, fill_value=0)

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0]

    names = {0: "Low", 1: "Moderate", 2: "High"}
    risk = names[prediction]

    st.divider()
    st.subheader("Prediction Result")

    if risk == "Low":
        st.success("🟢 Low Diabetes Risk")
    elif risk == "Moderate":
        st.warning("🟡 Moderate Diabetes Risk")
    else:
        st.error("🔴 High Diabetes Risk")

    result = pd.DataFrame({
        "Risk Level": ["Low", "Moderate", "High"],
        "Probability (%)": probability * 100
    })

    result["Probability (%)"] = result["Probability (%)"].round(2)

    st.dataframe(result, use_container_width=True, hide_index=True)
    st.write(f"Model Confidence: **{probability[prediction] * 100:.2f}%**")