import logging
import os,sys
from pydantic import BaseModel,create_model
from fastapi import FastAPI,HTTPException,Body
import joblib
import pandas as pd

from pipelines.inference_pipeline import Inference_Pipeline

os.makedirs('logs',exist_ok=True)
logging.basicConfig(
    filename='artifacts/api.log',
    level = logging.INFO,
    format = "%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("Starting FastAPI Service")

app = FastAPI(title="Credit Risk api",version = '0.1')

schema = joblib.load("artifacts/input_schema.pkl")

def map_dtype(dtype_str: str):
    if "int" in dtype_str:
        return int
    if "float" in dtype_str:
        return float
    if "category" in dtype_str:
        return str
    return str

schema = joblib.load("artifacts/input_schema.pkl")

fields = {
    col: (map_dtype(dtype), ...)
    for col, dtype in schema.items()
}

InputModel = create_model("InputModel", **fields)
pipeline = Inference_Pipeline()

@app.get("/")
def get():
    return ("Credit Risk API Running")

@app.post("/predict")
def predict(data: InputModel):
    try:
        logging.info(f"Recieved request {data.dict()}")

        prediction = pipeline.predict(data.dict())

        logging.info(f"Prediction: {prediction}")

        return {"prediction":prediction}
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))



