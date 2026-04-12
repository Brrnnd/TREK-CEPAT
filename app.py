import streamlit as st
import joblib
import numpy as np

# LOAD MODEL
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

st.title(" Prediksi Depresi Mahasiswa")

st.divider()

st.subheader(" Input Data")

# INPUT SESUAI FITUR

# 1. Suicidal Thoughts
suicidal = st.selectbox(
    "Pernah memiliki pikiran bunuh diri?",
    ["No", "Yes"]
)

# 2. Academic Pressure
academic_pressure = st.slider("Academic Pressure (0-10)", 0, 5, 2)

# 3. CGPA
cgpa = st.slider("CGPA", 0.0, 10.0, 5.0)

# 4. Financial Stress
financial_stress = st.slider("Financial Stress (0-10)", 0, 5, 2)

# 5. Age
age = st.slider("Age", 18, 60, 25)

# 6. City 
city = st.selectbox("City Type", ["Metro", "Non-Metro"])

# 7. Work/Study Hours
study_hours = st.slider("Work/Study Hours", 0, 12, 4)

# 8. Degree
degree = st.selectbox(
    "Degree",
    ["Undergraduate", "Postgraduate"]
)

# ENCODING (HARUS SESUAI TRAINING)

suicidal = 1 if suicidal == "Yes" else 0
city = 1 if city == "Metro" else 0
degree = 1 if degree == "Postgraduate" else 0

# SUSUN DATA (URUTAN HARUS SAMA)
input_data = np.array([[
    suicidal,
    academic_pressure,
    cgpa,
    financial_stress,
    age,
    city,
    study_hours,
    degree
]])

# SCALING
input_scaled = scaler.transform(input_data)

# PREDIKSI
st.divider()

if st.button(" Prediksi"):
    pred = model.predict(input_scaled)
    prob = model.predict_proba(input_scaled)

    st.subheader(" Hasil")

    if pred[0] == 1:
        st.error(" Terindikasi Depresi")
    else:
        st.success(" Tidak Depresi")

    st.write(f"Probabilitas: **{prob[0][1]*100:.2f}%**")

# INFO
st.divider()

