"""
Script d'entraînement du modèle de prédiction de prix de voiture.
Charge les données, entraîne un Random Forest et sauvegarde le modèle.
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Chargement du dataset (séparateur point-virgule)
df = pd.read_csv("data/data_set.csv", sep=";")

# Échantillon de 30 000 lignes pour réduire le temps d'entraînement
df = df.sample(30000, random_state=0)

# Suppression des colonnes non pertinentes pour la prédiction
df = df.drop(columns=["Unnamed: 0", "L_by_100km", "trunk_volume"])

# Remplacement des valeurs manquantes par la médiane de chaque colonne
df = df.fillna(df.median(numeric_only=True))

# Features sélectionnées pour entraîner le modèle
features = ["km", "HorseP", "year", "CV_fisc", "CO2_g_km"]

X = df[features]
y = df["price"]

# Split 80% entraînement / 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)
# Modèle Random Forest ,paramètres légers pour un premier entraînement rapide
model = RandomForestRegressor(
    n_estimators=20,  # Nombre d'arbres
    max_depth=10,     # Profondeur max pour éviter l'overfitting
    random_state=0,
    n_jobs=-1         # Utilise tous les cœurs CPU disponibles
)

model.fit(X_train, y_train)

# Sauvegarde du modèle entraîné pour utilisation dans l'API
joblib.dump(model, "model/car_model.pkl")

print("Modele entrainé et sauvegardé!")