import numpy as np
import sys
from pathlib import Path

root_dir = str(Path(__file__).parent.parent.parent.absolute())
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.config import BEST_F1_MODEL_FILE_PATH, BEST_F1_MODEL_SCALER_FILE_PATH
from src.models import DiabetesXGBoostClassifier
from src.utils import load_data


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
        print(f"Model or scaler file not found: {e}")
        print("Please ensure model files are in the correct directory.")
        return None, None
    
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None


def make_prediction(model, scaler, feature_df):
    """
    Make prediction using trained model
    
    Args:
        model: Trained model object
        scaler: Fitted scaler object
        feature_df: DataFrame with all required features
    
    Returns:
        prediction_class: Predicted class label
        probabilities: Array of class probabilities
    """
    try:
        # Scale features
        features_scaled = scaler.transform(feature_df)
        
        # Make prediction
        prediction = model.predict(features_scaled)
        probabilities = model.predict_proba(features_scaled)[0]
        
        # Map prediction to class label
        class_mapping = {0: "No Diabetes", 1: "Pre-diabetes", 2: "Diabetes"}
        prediction_class = class_mapping.get(prediction[0], "Unknown")
        
        return prediction_class, probabilities
    
    except Exception as e:
        print(f"Error making prediction: {e}")
        return "Error", np.array([0.0, 0.0, 0.0])


def assess_risk_level(prediction_class, probabilities):
    """
    Assess risk level based on prediction and probabilities
    
    Args:
        prediction_class: Predicted class label
        probabilities: Array of class probabilities
    
    Returns:
        risk_level: String describing risk level
        recommendation: String with health recommendation
    """
    diabetes_prob = probabilities[2]  # Probability of diabetes
    prediabetes_prob = probabilities[1]  # Probability of pre-diabetes
    
    # Determine risk level
    if prediction_class == "Diabetes":
        if diabetes_prob >= 0.8:
            risk_level = "🔴 HIGH RISK"
            recommendation = (
                "High probability of diabetes detected. Immediate medical consultation "
                "is strongly recommended. Consider comprehensive diabetes screening and "
                "discuss treatment options with healthcare provider."
            )
        else:
            risk_level = "🟠 MODERATE-HIGH RISK"
            recommendation = (
                "Moderate probability of diabetes. Schedule appointment with healthcare "
                "provider for thorough evaluation. Monitoring blood glucose levels and "
                "lifestyle modifications are advised."
            )
    
    elif prediction_class == "Pre-diabetes":
        risk_level = "🟡 MODERATE RISK"
        recommendation = (
            "Pre-diabetes indicators detected. This is a critical time for intervention. "
            "Lifestyle changes (diet, exercise) can prevent progression to diabetes. "
            "Regular monitoring and medical consultation recommended."
        )
    
    else:  # No Diabetes
        if prediabetes_prob >= 0.2 or diabetes_prob >= 0.15:
            risk_level = "🟢 LOW-MODERATE RISK"
            recommendation = (
                "Currently low risk, but some risk factors present. Maintain healthy "
                "lifestyle habits and regular health check-ups. Focus on preventive care."
            )
        else:
            risk_level = "🟢 LOW RISK"
            recommendation = (
                "Low diabetes risk detected. Continue maintaining healthy lifestyle habits. "
                "Regular health screenings are still recommended for ongoing wellness."
            )
    
    return risk_level, recommendation


def get_model_info():
    """Get information about the deployed model"""
    model_info = {
        "name": "XGBoost Classifier",
        "balancing_method": "Random Oversampling",
        "accuracy": "89.55%",
        "precision": "90.91%",
        "recall": "60.32%",
        "f1_score": "69.92%",
        "training_samples": "450,445",
        "features": "34"
    }
    return model_info
