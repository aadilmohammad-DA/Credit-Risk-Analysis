import os,sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score,precision_score,recall_score,classification_report
import logging


import joblib
import pickle
import yaml
from datetime import datetime
from utils.utility import feature_engineering

os.makedirs("logs",exist_ok=True)
logging.basicConfig(
    filename="logs/training.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("Training pipeline started")

# data ingestion
data = pd.read_csv("data/credit_risk_dataset.csv")

# Splitting the data
y = data['credit_risk']
X = data.drop(columns=['credit_risk','customer_id'],axis = 1)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)
schema = {col: str(X_train[col].dtype) for col in X_train.columns}
with open("artifacts/input_Schema.pkl",'wb') as f:
    joblib.dump(schema,f)
#feature_names = list(X_train.columns)
#joblib.dump(feature_names, "artifacts/feature_names.pkl")

num_cols = X_train.select_dtypes(include=['int64','float64']).columns
cat_cols = X_train.select_dtypes(include=['object','category']).columns

logging.info("Dataset features before adding new features")
logging.info(f"Numerical({len(num_cols)}):\n{num_cols}")
logging.info(f"Categorical({len(cat_cols)}):\n{cat_cols}")
# Calling feature engineering utility function 
# applying feature engineering on train and rest data
X_train,features = feature_engineering(X_train)
X_test,*args = feature_engineering(X_test)

num_cols = X_train.select_dtypes(include=['int64','float64']).columns
cat_cols = X_train.select_dtypes(include=['object','category']).columns
logging.info("new features added")
logging.info(f"{features}")

num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')
ohe = OneHotEncoder(handle_unknown='ignore')

logging.info("Pipeline being created")
num_pipe = Pipeline(steps=[
    ('imputer',num_imputer)
])

cat_pipe = Pipeline(steps=[
    ('imputer',cat_imputer),
    ('encoder',ohe)
])

preprocessor = ColumnTransformer(
    transformers=[
        ('numerical',num_pipe,num_cols),
        ('categorical',cat_pipe,cat_cols)
    ]
)

model = Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('classifier',LogisticRegression(max_iter=1000))
])

logging.info("Pipeline Creation completed")
logging.info("Training the model using train data and testing using test data")

model.fit(X_train,y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)
precision = precision_score(y_test,y_pred)
recall = recall_score(y_test,y_pred)

logging.info(f"{classification_report(y_test,y_pred)}")
logging.info("Prediction and Scoring Completed")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_version = f"model_{timestamp}.pkl"
preprocessor_version = f"preprocessor_{timestamp}.pkl"

os.makedirs('artifacts/models',exist_ok=True)
os.makedirs('artifacts/preprocessors',exist_ok=True)
joblib.dump(model.named_steps['classifier'],f"artifacts/models/{model_version}")
joblib.dump(model.named_steps['preprocessor'],f"artifacts/preprocessors/{preprocessor_version}")

logging.info(f"Saved artifact artifacts/models/{model_version}")
logging.info(f"Saved artifact artifacts/preprocessors/{preprocessor_version}")

#Saving the configuration for inference
config = {
    'accuracy' : accuracy,
    'precision' : float(precision),
    'recall': float(recall),
    'model_path': f"artifacts/models/{model_version}",
    'preprocessor_path': f"artifacts/prprocessors/{preprocessor_version}"
}

with open('artifacts/config.yaml','w') as f:
    yaml.dump(config,f)

logging.info(f"Saved config file")
logging.info("Training Process Completed")






