from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os

MODEL_PATH = os.path.join("models", "model.pkl")
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Simple Data Science API")

class InputData(BaseModel):
    feature1: float
    feature2: float

@app.get("/")
def root():
    return {"message": "Welcome to the Data Science API"}

@app.post("/predict")
def predict(data: InputData):
    features = [[data.feature1, data.feature2]]
    prediction = model.predict(features)[0]
    return {"prediction": prediction}
