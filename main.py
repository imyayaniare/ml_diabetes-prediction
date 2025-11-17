from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

# Charger le modèle et le scaler
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

# Définir la structure des données reçues
class Patient(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float
    Cluster: int  # la feature cluster issue du KMeans

# Créer l'application FastAPI
app = FastAPI()

# Route principale
@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API de prédiction du diabète"}

# Route pour prédire
@app.post("/predict")
def predict_diabetes(patient: Patient):
    # Transformer les données en array
    data = np.array([[patient.Pregnancies,
                      patient.Glucose,
                      patient.BloodPressure,
                      patient.SkinThickness,
                      patient.Insulin,
                      patient.BMI,
                      patient.DiabetesPedigreeFunction,
                      patient.Age,
                      patient.Cluster]])
    
    # Normaliser avec le scaler
    data_scaled = scaler.transform(data)
    
    # Prédiction
    prediction = model.predict(data_scaled)[0]
    probability = model.predict_proba(data_scaled)[0][1]
    
    return {
        "prediction": int(prediction),
        "probability": float(probability)
    }
