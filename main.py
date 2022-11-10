import datetime
from pydantic import BaseModel, Field
from functions import app
from functions.functions import get_model_response


model_name = "Heart Disease (Diagnostic)"
version = "v1.0.0"


# Input for data validation
class Input(BaseModel):
    Smoking: str
    AlcoholDrinking: str
    Stroke: str
    DiffWalking: str
    Sex: str
    AgeCategory: str
    Diabetic: str
    PhysicalActivity: str
    GenHealth: str 
    Asthma: str
    KidneyDisease: str
    SkinCancer: str
    BMI: float 
    PhysicalHealth: int
    MentalHealth: int
    SleepTime: int

    class Config:
        schema_extra = {
            "example": {
                "Smoking": "No",
                "AlcoholDrinking": "No",
                "Stroke": "No",
                "DiffWalking": "No",
                "Sex": "Female",
                "AgeCategory": "25-29",
                "Diabetic": "No",
                "PhysicalActivity": "Yes",
                "GenHealth": "Good",
                "Asthma": "No",
                "KidneyDisease": "No",
                "SkinCancer": "No",
                "BMI": 28.1,
                "PhysicalHealth": 3,
                "MentalHealth": 2,
                "SleepTime": 8
        }
        }


# Ouput for data validation
class Output(BaseModel):
    label: str
    prediction: int


@app.get('/')
async def model_info():
    """Return model information, version, how to call"""
    return {
        "name": model_name,
        "version": version
    }


@app.get('/heart_disease')
async def service_health():
    """Return service health"""
    return {
        "ok"
    }


@app.post('/predict', response_model=Output)
async def model_predict(input: Input):
    """Predict with input"""
    response = get_model_response(input)
    return response