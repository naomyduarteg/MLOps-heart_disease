from pydantic import BaseModel, Field
from functions.functions import get_model_response
from fastapi import FastAPI
import uvicorn


# Initialize FastAPI app
app = FastAPI()

model_name = "Heart Disease (Diagnostic)"
version = "v1.0.0"

#Input data validation for the model
class Input(BaseModel):
    Smoking: str = Field(..., description="Choose Yes/No")
    AlcoholDrinking: str = Field(..., description="Choose Yes/No")
    Stroke: str = Field(..., description="Choose Yes/No")
    DiffWalking: str = Field(..., description="Choose Yes/No")
    Sex: str = Field(..., description="Choose Female/Male")
    AgeCategory: str = Field(..., description="Choose 18-24/25-29/30-34/35-39/40-44/45-49/50-54/55-59/60-64/65-69/70-74/75-79/80 or older")
    Diabetic: str = Field(..., description="Choose Yes/No/No, borderline diabetes/Yes (during pregnancy)")
    PhysicalActivity: str = Field(..., description="Choose Yes/No")
    GenHealth: str = Field(..., description="Choose Excellent/Fair/Good/Poor/Very good")
    Asthma: str = Field(..., description="Choose Yes/No")
    KidneyDisease: str = Field(..., description="Choose Yes/No")
    SkinCancer: str = Field(..., description="Choose Yes/No")
    BMI: float = Field(..., gt=0)
    PhysicalHealth: int = Field(..., gt=0, le=30)
    MentalHealth: int = Field(..., gt=0, le=30)
    SleepTime: int = Field(..., gt=0, le=24)

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


# Ouput data validation
class Output(BaseModel):
    label: str
    prediction: int


@app.get('/')
async def model_info():
    """Return model information"""
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
    """Return prediction given the input"""
    response = get_model_response(input)
    return response


if __name__ == '__main__':
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)