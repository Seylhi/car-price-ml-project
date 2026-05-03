import streamlit as st
import requests

st.title("Prediction du prix d'une voiture - 1er App ML")

km = st.number_input("Kilometrage", 0, 500000, 50000)
horsepower = st.number_input("Cheveaux", 50, 1000, 120)
year = st.number_input("Année", 1990, 2024, 2018)
cv_fisc = st.number_input("Puissance fiscale", 1, 50, 7)
co2 = st.number_input("CO2 g/km", 0, 500, 120)

if st.button("Predict price"):
    
    data = {
        "km": km,
        "horsepower": horsepower,
        "year": year,
        "CV_fisc": cv_fisc,
        "CO2_g_km": co2
    }

    response = requests.post("http://127.0.0.1:8000/predict", json=data)
    
    result = response.json()

    st.success(f"Estimation du prix : {result['predicted_price']:.2f} €")