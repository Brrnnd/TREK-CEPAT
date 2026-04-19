import streamlit as st
import joblib
import numpy as np


# LOAD

model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

st.title(" Prediksi Depresi Mahasiswa ")

st.divider()

st.subheader(" Input Data")


# INPUT DASAR

suicidal = st.selectbox("Pernah memiliki pikiran bunuh diri?", ["No", "Yes"])

academic_pressure = st.slider("Tekanan Akademik", 0, 5, 2)
work_pressure = st.slider("Tekanan Kerja", 0, 5, 2)
financial_stress = st.slider("Tekanan Keuangan", 0, 5, 2)

study_satisfaction = st.slider("Kepuasan Belajar", 0, 5, 3)

cgpa = st.slider("Nilai", 0.0, 10.0, 5.0)
age = st.slider("Umur", 18, 60, 25)

study_hours = st.slider("Jam Kerja/Belajar", 0, 12, 4)

# City 
city = st.selectbox("Tempat Tinggal", ["Metro", "Non-Metro"])


# ENCODING

suicidal = 1 if suicidal == "Yes" else 0
city = 1 if city == "Metro" else 0


# FEATURE ENGINEERING

total_pressure = academic_pressure + work_pressure + financial_stress

pressure_ratio = academic_pressure / (study_satisfaction + 1)


# SUSUN SESUAI FINAL FEATURES

input_data = np.array([[
    suicidal,
    total_pressure,
    pressure_ratio,
    age,
    cgpa,
    academic_pressure,
    city,
    study_hours
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



st.divider()

