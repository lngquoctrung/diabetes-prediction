import pandas as pd
import numpy as np
import streamlit as st

@st.cache_data
def load_sample_data():
    """Generate sample data for demonstration"""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'BMI': np.random.normal(28, 6, n_samples),
        'Age': np.random.randint(18, 80, n_samples),
        'GenHlth': np.random.randint(1, 6, n_samples),
        'HighBP': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        'HighChol': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'PhysActivity': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
        'Diabetes': np.random.choice([0, 1, 2], n_samples, p=[0.82, 0.024, 0.156])
    }
    
    return pd.DataFrame(data)

def get_model_comparison_data():
    """Return model comparison results - Updated with actual results"""
    model_results = {
        'Model': ['LightGBM', 'XGBoost', 'LightGBM', 'XGBoost', 'Random Forest'],
        'Method': ['ADASYN Tomek Links', 'Random Oversampling', 'SMOTE Tomek Links', 'SMOTE', 'Random Undersampling'],
        'Accuracy': [0.9011, 0.8955, 0.9009, 0.8995, 0.6488],
        'Precision': [0.9167, 0.7770, 0.9113, 0.8970, 0.4465],
        'Recall': [0.6247, 0.6552, 0.6247, 0.6252, 0.5455],
        'F1-Score': [0.6902, 0.6992, 0.6902, 0.6875, 0.4416]
    }
    
    return pd.DataFrame(model_results)

def get_feature_importance_data():
    """Return feature importance data - Updated with actual importance"""
    return {
        'PhysHlth': 0.4400,
        'HlthScore': 0.3971,
        'CardioRisk': 0.3385,
        'RiskScore': 0.3053,
        'GenHlth': 0.3033,
        'CholesterolMeds': 0.2946,
        'HighBP': 0.2643,
        'BMI': 0.2485,
        'AlcoholDays': 0.2440,
        'DiffWalk': 0.2195
    }
