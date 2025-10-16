from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib

model = joblib.load("model") 

app = FastAPI()

class PatientData(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

@app.post("/predict")
def predict(data: PatientData):
    features = [
        data.age, data.sex, data.cp, data.trestbps, data.chol, data.fbs,
        data.restecg, data.thalach, data.exang, data.oldpeak, data.slope,
        data.ca, data.thal
    ]
    prediction = model.predict([features])
    #probabilities = model.predict_proba([features])[0]
    return {"prediction": "Insuffisance cardiaque détectée" if predict == 1 else "Aucune insuffisance cardiaque détectée"}