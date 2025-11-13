import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os

MODEL_PATH = os.path.join("models", "model.pkl")

def train_and_save_model():
    data = pd.read_csv("data/sample_data.csv")
    X = data[['feature1', 'feature2']]
    y = data['target']

    model = LinearRegression()
    model.fit(X, y)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved at {MODEL_PATH}")

if __name__ == "__main__":
    train_and_save_model()
