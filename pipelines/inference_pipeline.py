import os,sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import yaml
import joblib
import logging
import pandas as pd
from utils.utility import feature_engineering


os.makedirs('logs',exist_ok=True)
logging.basicConfig(
    filename="logs/inference.log",
    level = logging.INFO,
    format = "%(asctime)s - %(levelname)s - %(message)s"
)

class Inference_Pipeline():
    def __init__(self,config = 'artifacts/config.yaml'):

        logging.info("Inference Pipeline Started")
        logging.info("Loading artifacts")
        with open(config) as f:
            config = yaml.safe_load(f)
        
        self.model_path = config['model_path']
        self.preprocessor_path = config['preprocessor_path']

        self.model = joblib.load(self.model_path)
        self.preprocessor = joblib.load(self.preprocessor_path)
        self.feature_names = joblib.load('artifacts/input_schema.pkl')

        logging.info(f"Loaded {self.model_path}")
        logging.info(f"loaded {self.preprocessor_path}")
        logging.info(f"Loaded raw feature set artifact/feature_names.pkl")
        logging.info(f"Features Loaded are{len(self.feature_names.keys())}: {self.feature_names.keys()}")
        logging.info(f"Respective dtype of the features are {self.feature_names.values()}")
        logging.info("Artifacts loaded Sucessfully")
    
    def _prepare_input(self,data):
        logging.info("Preparing input for prediction")
        if isinstance(data,dict):
            df = pd.DataFrame([data])
        elif isinstance(data,list):
            df = pd.DataFrame([data],columns=self.feature_names.keys())
        else:
            raise ValueError("The input must be a list or a dictionary")
        
        df = df[list(self.feature_names.keys())]

        df,features = feature_engineering(df)
        logging.info(f"total features:{len(df.columns)}")
        return df
    
    def predict(self,data):
        X = self._prepare_input(data)
        logging.info("Prediction precessing")
        X_preprocessed = self.preprocessor.transform(X)
        prediction = self.model.predict(X_preprocessed)
        logging.info(f"Prediction result {int(prediction[0])}")
        return int(prediction[0])
    
# Object creation and call of prediction
def predict(input_data):
    pipeline = Inference_Pipeline()
    pipeline.predict(input_data)
