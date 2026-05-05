"""
API FastAPI pour la prédiction du prix d'une voiture.
Route POST /predict : reçoit les caractéristiques du véhicule, retourne un prix estimé.
"""
from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Chargement du modèle au démarrage du serveur
model = joblib.load("model/car_model.pkl")

@app.post("/predict")
def predict(data: dict):

    input_df = pd.DataFrame([{
        #Caractéristiques d'entrée pour la prédiction.
        "km": data["km"],  # Kilométrage du véhicule
        "HorseP": data["horsepower"], # Puissance en chevaux (ch)
        "year": data["year"], # Année de mise en circulation
        "CV_fisc": data["CV_fisc"], # Puissance fiscale (chevaux fiscaux)
        "CO2_g_km": data["CO2_g_km"] # Émissions CO2 en g/km
    }])

    prediction = model.predict(input_df)

    return {
        "predicted_price": float(prediction[0])
    }