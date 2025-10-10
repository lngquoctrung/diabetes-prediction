#!/bin/bash
# Create storage directory
mkdir -p ~/services/public/diabetes-prediction-app

# Download model checkpoint, scaler and dataset
curl "https://github.com/lngquoctrung/diabetes-prediction/releases/download/v1.0.0/xgboost_model_checkpoint.pkl" -o "~/services/public/diabetes-prediction-app/xgboost_model_checkpoint.pkl"
curl "https://github.com/lngquoctrung/diabetes-prediction/releases/download/v1.0.0/xgb_min_max_scaler.pkl" -o "~/services/public/diabetes-prediction-app/xgb_min_max_scaler.pkl"
curl "https://github.com/lngquoctrung/diabetes-prediction/releases/download/v1.0.0/brfss_dataset.csv" -o "~/services/public/diabetes-prediction-app/brfss_dataset.csv"