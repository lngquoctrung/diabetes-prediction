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
    """Return model comparison results with simplified column names"""
    
    model_results = {
        'Model': [
            'LightGBM', 
            'XGBoost', 
            'LightGBM', 
            'XGBoost', 
            'Random Forest'
        ],
        'Method': [
            'ADASYN + Tomek Links', 
            'Random Oversampling', 
            'SMOTE + Tomek Links', 
            'SMOTE', 
            'Random Undersampling'
        ],
        'Accuracy': [0.9011, 0.8947, 0.9009, 0.8995, 0.8277],
        'Precision': [0.9167, 0.7715, 0.9113, 0.8970, 0.5920],
        'Recall': [0.6247, 0.6542, 0.6247, 0.6252, 0.6821],
        'F1 Score': [0.6902, 0.6969, 0.6902, 0.6874, 0.6235],
        'ROC AUC': [0.9233, 0.9235, 0.9231, 0.9209, 0.9075],
        
        # Precision per class
        'Precision Class 0': [0.9072, 0.9160, 0.9070, 0.9088, 0.9555],
        'Precision Class 1': [0.9829, 0.5890, 0.9666, 0.9391, 0.2343],
        'Precision Class 2': [0.8601, 0.8095, 0.8603, 0.8432, 0.5863],
        
        # Recall per class
        'Recall Class 0': [0.9817, 0.9632, 0.9817, 0.9781, 0.8400],
        'Recall Class 1': [0.2399, 0.3133, 0.2416, 0.2382, 0.3642],
        'Recall Class 2': [0.6524, 0.6861, 0.6508, 0.6591, 0.8420],
        
        # F1 Score per class
        'F1 Class 0': [0.9431, 0.9389, 0.9430, 0.9424, 0.8943],
        'F1 Class 1': [0.3847, 0.4127, 0.3863, 0.3816, 0.2852],
        'F1 Class 2': [0.7439, 0.7413, 0.7428, 0.7393, 0.6927]
    }
    
    df = pd.DataFrame(model_results)
    
    # Format numbers to percentages
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = (df[col] * 100).round(2).astype(str) + '%'
    
    return df


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
