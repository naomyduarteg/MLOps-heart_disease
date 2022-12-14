import pandas as pd
import joblib


# Load model
model = joblib.load('model_to_run/model.dat.gz')

def predict(X, model):
    prediction = model.predict(X)[0]
    return prediction


def get_model_response(input):
    X = pd.json_normalize(input.__dict__)
    prediction = predict(X, model)
    if prediction == 1:
        label = "Higher chances of heart disease"
    else:
        label = "Lower chances of heart disease"
    return {
        'label': label,
        'prediction': int(prediction)
    }