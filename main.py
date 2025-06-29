from contextlib import asynccontextmanager
import os
import numpy as np
from pydantic import BaseModel
from fastapi import FastAPI
from keras.models import load_model

class FetalHealthData(BaseModel):
    accelerations: float
    fetal_movement: float
    uterine_contractions: float
    severe_decelerations: float
    
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = loading_model()
    yield 

app = FastAPI(
    title="MLOPS exercise",
    openapi_tags=[
        {
            "name": "Health",
            "description": "Get api health"
        },
        {
            "name": "Prediction",
            "description": "Model prediction"
        }
    ]
)


def loading_model():
    model_path = os.path.join("models", "mlops_unit4.keras")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model = load_model(model_path)
    return model

@app.on_event(event_type='startup')
def startup_event():
    global model 
    model = loading_model()



@app.get(path="/", tags=["Health"])
def api_health():
    return {"status": "healthy"}

@app.post("/predict", tags=["Prediction"])
def predict(request: FetalHealthData):
    received_data = np.array([
        request.accelerations, 
        request.fetal_movement, 
        request.uterine_contractions,
        request.severe_decelerations
    ]).reshape(1, -1)

    prediction = model.predict(received_data)
    return {"prediction" : str(prediction[0][0])}
