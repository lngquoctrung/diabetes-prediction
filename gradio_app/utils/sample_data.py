import pandas as pd
import numpy as np


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
        'Precision': [0.9167, 0.9091, 0.9155, 0.9050, 0.5625],
        'Recall': [0.6247, 0.6032, 0.6236, 0.6177, 0.6178],
        'F1-Score': [0.6814, 0.6992, 0.6898, 0.6822, 0.5903]
    }
    
    return pd.DataFrame(model_results)


def get_feature_importance_data():
    """Return feature importance scores"""
    feature_importance = {
        'GenHlth': 0.4521,
        'HighBP': 0.3892,
        'BMI': 0.3654,
        'Age': 0.3421,
        'HighChol': 0.3198,
        'DiffWalk': 0.2876,
        'PhysHlth': 0.2654,
        'Income': 0.2543,
        'HeartDiseaseorAttack': 0.2431,
        'Stroke': 0.2198,
        'MentHlth': 0.1987,
        'Education': 0.1876,
        'PhysActivity': 0.1654,
        'Smoker': 0.1543
    }
    
    return feature_importance


def get_diabetes_distribution():
    """Return diabetes class distribution"""
    distribution = {
        'No Diabetes': 369364,
        'Pre-diabetes': 10811,
        'Diabetes': 70270
    }
    
    return distribution
