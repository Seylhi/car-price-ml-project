import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("data/data_set.csv", sep=";")

df = df.sample(30000, random_state=42)

df = df.drop(columns=["Unnamed: 0", "L_by_100km", "trunk_volume"])

df = df.fillna(df.median(numeric_only=True))

features = ["km", "HorseP", "year", "CV_fisc", "CO2_g_km"]

X = df[features]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=20,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

joblib.dump(model, "model/car_model.pkl")

print("Modele entrainé et sauvegardé!")