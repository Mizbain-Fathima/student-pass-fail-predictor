import streamlit as st
import pandas as pd
import joblib

# Load trained pipeline model
model = joblib.load("best_pass_fail_model.joblib")

st.set_page_config(page_title="🎓 Student Pass/Fail Predictor", page_icon="📘", layout="centered")

st.title("🎓 Student Pass/Fail Predictor")
st.write("Use this app to predict whether a student is likely to **pass or fail** based on academic and personal factors.")

st.markdown("---")
st.subheader("🧍 Student Information")

with st.form("student_form"):

    # ===================== Basic Info =====================
    st.markdown("### 📘 General Information")
    school = st.selectbox("School (GP: Gabriel Pereira, MS: Mousinho da Silveira)", ["GP", "MS"])
    sex = st.selectbox("Gender", ["F (Female)", "M (Male"])
    age = st.slider("Age (in years)", 15, 22, 17)
    address = st.selectbox("Address Type (U: Urban, R: Rural)", ["U", "R"])
    famsize = st.selectbox("Family Size (LE3: ≤3 members, GT3: >3 members)", ["LE3", "GT3"])
    Pstatus = st.selectbox("Parent Cohabitation Status (T: Together, A: Apart)", ["T", "A"])

    # ===================== Education =====================
    st.markdown("### 🧠 Education Background")
    Medu = st.slider("Mother’s Education (0: None – 4: Higher Education)", 0, 4, 2)
    Fedu = st.slider("Father’s Education (0: None – 4: Higher Education)", 0, 4, 2)
    Mjob = st.selectbox("Mother’s Job", ["teacher", "health", "services", "at_home", "other"])
    Fjob = st.selectbox("Father’s Job", ["teacher", "health", "services", "at_home", "other"])
    reason = st.selectbox("Reason for Choosing School", ["home", "reputation", "course", "other"])
    guardian = st.selectbox("Main Guardian", ["mother", "father", "other"])

    # ===================== Study Info =====================
    st.markdown("### 📚 Study & Behavior Factors")
    traveltime = st.slider("Travel Time (1: <15min, 2: 15–30min, 3: 30–60min, 4: >60min)", 1, 4, 1)
    studytime = st.slider("Weekly Study Time (1: <2h, 2: 2–5h, 3: 5–10h, 4: >10h)", 1, 4, 2)
    failures = st.slider("Past Class Failures (0–4)", 0, 4, 0)
    absences = st.number_input("Total Absences", 0, 100, 5)

    # ===================== Support & Lifestyle =====================
    st.markdown("### 💬 Support & Lifestyle")
    schoolsup = st.selectbox("Extra Educational Support", ["yes", "no"])
    famsup = st.selectbox("Family Support", ["yes", "no"])
    paid = st.selectbox("Extra Paid Classes", ["yes", "no"])
    activities = st.selectbox("Extracurricular Activities", ["yes", "no"])
    nursery = st.selectbox("Attended Nursery School", ["yes", "no"])
    higher = st.selectbox("Wants Higher Education", ["yes", "no"])
    internet = st.selectbox("Has Internet Access at Home", ["yes", "no"])
    romantic = st.selectbox("In a Relationship", ["yes", "no"])

    # ===================== Grades =====================
    st.markdown("### 🧾 Academic Performance")
    G1 = st.slider("First Period Grade (G1: 0–20)", 0, 20, 12)
    G2 = st.slider("Second Period Grade (G2: 0–20)", 0, 20, 13)
    famrel = st.slider("Family Relationship Quality (1–5)", 1, 5, 4)
    freetime = st.slider("Free Time After School (1–5)", 1, 5, 3)
    goout = st.slider("Going Out Frequency (1–5)", 1, 5, 2)
    Dalc = st.slider("Workday Alcohol Consumption (1–5)", 1, 5, 1)
    Walc = st.slider("Weekend Alcohol Consumption (1–5)", 1, 5, 1)
    health = st.slider("Current Health Status (1–5)", 1, 5, 4)

    submitted = st.form_submit_button("🎯 Predict Pass/Fail")

# ===================== Prediction =====================
if submitted:
    # Prepare DataFrame
    new_data = pd.DataFrame([{
        "school": school.split()[0],  # Extract GP/MS from label
        "sex": sex[0],  # Extract F/M
        "age": age, "address": address, "famsize": famsize, "Pstatus": Pstatus,
        "Medu": Medu, "Fedu": Fedu, "Mjob": Mjob, "Fjob": Fjob, "reason": reason,
        "guardian": guardian, "traveltime": traveltime, "studytime": studytime,
        "failures": failures, "schoolsup": schoolsup, "famsup": famsup, "paid": paid,
        "activities": activities, "nursery": nursery, "higher": higher,
        "internet": internet, "romantic": romantic, "famrel": famrel,
        "freetime": freetime, "goout": goout, "Dalc": Dalc, "Walc": Walc,
        "health": health, "absences": absences, "G1": G1, "G2": G2
    }])

    # Predict using full pipeline (includes encoder)
    pred = model.predict(new_data)[0]
    prob = model.predict_proba(new_data)[0][pred]

    if pred == 1:
        st.success(f"✅ Prediction: PASS (Confidence: {prob*100:.1f}%)")
    else:
        st.error(f"❌ Prediction: FAIL (Confidence: {prob*100:.1f}%)")

    st.info("Note: This prediction is based on historical patterns in student performance data and is for educational purposes only.")
