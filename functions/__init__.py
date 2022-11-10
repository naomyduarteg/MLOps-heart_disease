from fastapi import FastAPI
import joblib

# Initialize FastAPI app
app = FastAPI()

# Load model
model = joblib.load('model_to_run/model.dat.gz')