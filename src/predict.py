import pickle
from pathlib import Path
import pandas as pd


MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "breast_cancer_pipeline.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


def predict(sample: pd.DataFrame):

    prediction = model.predict(sample)[0]

    if prediction == 1:
        return "Malignant"
    else:
        return "Benign"