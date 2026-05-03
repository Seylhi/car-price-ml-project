from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("model/car_model.pkl")

@app.post("/predict")
def predict(data: dict):

    input_df = pd.DataFrame([{
        "km": data["km"],
        "HorseP": data["horsepower"],
        "year": data["year"],
        "CV_fisc": data["CV_fisc"],
        "CO2_g_km": data["CO2_g_km"]
    }])

    prediction = model.predict(input_df)

    return {
        "predicted_price": float(prediction[0])
    }