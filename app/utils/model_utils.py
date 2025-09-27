import pickle
import pandas as pd
import numpy as np
import streamlit as st
import os
import sys

from pathlib import Path

root_dir = str(Path(__file__).parent.parent.parent.absolute())
dashboard_dir = os.path.join(root_dir, "app")
if not dashboard_dir in sys.path:
    sys.path.insert(0, dashboard_dir)
if not root_dir in sys.path:
    sys.path.insert(0, root_dir)

from src.config import BEST_F1_MODEL_FILE_PATH, BEST_F1_MODEL_SCALER_FILE_PATH
from src.models import DiabetesXGBoostClassifier
from src.utils import load_data

@st.cache_resource
def load_trained_model_and_scaler():
    """Load trained XGBoost model and scaler"""
    try:
        # Load pretrained model
        model = DiabetesXGBoostClassifier()
        model.load_model(model_path=BEST_F1_MODEL_FILE_PATH)

        # Load scaler
        scaler = load_data(path=BEST_F1_MODEL_SCALER_FILE_PATH)
        return model, scaler
    except FileNotFoundError as e:
        st.error(f"Model or scaler file not found: {e}")
        st.error("Please ensure model files are in the correct directory.")
        return None, None
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
        return None, None

def make_prediction(model, scaler, feature_df):
    """Make diabetes prediction using trained model"""
    try:
        # Apply scaling
        feature_df_scaled = scaler.transform(feature_df)
        if isinstance(feature_df_scaled, np.ndarray):
            feature_df_scaled = pd.DataFrame(feature_df_scaled, columns=feature_df.columns)
        
        # Make prediction
        probabilities = model.predict_proba(feature_df_scaled)
        prob_no_diabetes, prob_prediabetes, prob_diabetes = probabilities[0]
        
        predictions = {
            'No Diabetes': prob_no_diabetes,
            'Pre-diabetes': prob_prediabetes,
            'Diabetes': prob_diabetes
        }
        
        return predictions
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

def assess_risk_level(predictions):
    """Assess risk level based on prediction probabilities"""
    max_pred = max(predictions.values())
    max_class = max(predictions, key=predictions.get)
    
    if max_class == "No Diabetes":
        risk_level = "LOW"
        recommendation = "Excellent! Continue maintaining a healthy lifestyle."
    elif max_class == "Pre-diabetes":
        risk_level = "MODERATE"
        recommendation = "Attention needed! Lifestyle changes can reduce risk."
    else:
        risk_level = "HIGH"
        recommendation = "Consider consulting a doctor for advice and timely treatment."
    
    return risk_level, max_class, max_pred, recommendation

def get_model_info():
    """Return model information for display"""
    return {
        'model_name': 'XGBoost with Random Oversampling',
        'accuracy': '89.55%',
        'f1_score': '69.92%',
        'training_samples': '450,445',
        'data_source': 'BRFSS 2017-2023'
    }
