import numpy as np
import sys
from pathlib import Path

root_dir = str(Path(__file__).parent.parent.parent.absolute())
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import os
import pickle
from xgboost import XGBClassifier

from src.config import RANDOM_STATE, LOG_FORMAT
from src.utils import load_data, make_dirs, sanitize_path, get_configured_logger

class DiabetesXGBoostClassifier:
    """
    Initialize an XGBoost Classifier model with essential parameters

    Args:
        n_estimators: Number of boosting rounds (default: 100)
        max_depth: Maximum tree depth for base learners (default: 6)
        learning_rate: Boosting learning rate (default: 0.1)
        subsample: Subsample ratio of the training instances (default: 1.0)
        colsample_bytree: Subsample ratio of columns when constructing each tree (default: 1.0)
        random_state: Random state for reproducibility
        log_file: Path to log file
        log_format: Log format string
    """
    def __init__(self,
                 n_estimators=100,
                 max_depth=6,
                 learning_rate=0.1,
                 subsample=1.0,
                 colsample_bytree=1.0,
                 random_state=RANDOM_STATE,
                 logger_name: str | None = __name__,
                 log_file: str = None,
                 log_format: str | None = LOG_FORMAT):

        self.xgb_classifier = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            eval_metric='mlogloss'
        )
        self.model_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree,
            'random_state': random_state
        }
        self.model_path = None

        # Log configuration
        self.logger = get_configured_logger(
            name=logger_name, 
            log_file=log_file, 
            log_format=log_format
        )

        # Log initial parameters
        self.logger.info("DiabetesXGBoostClassifier initialized with parameters:")
        for key, value in self.model_params.items():
            self.logger.info(f" {key}: {value}")

    def train(self, X, y):
        """
        Train the XGBoost model

        Args:
            X: Training features
            y: Training labels
        """
        try:
            self.logger.info("Starting XGBoost model training...")
            self.xgb_classifier.fit(X=X, y=y)
            self.logger.info(f"Model training completed. Training samples: {len(X)}")
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            raise

    def predict(self, X):
        """
        Make predictions using the trained model

        Args:
            X: Features to predict

        Returns:
            Predicted labels
        """
        try:
            predictions = self.xgb_classifier.predict(X=X)
            self.logger.info(f"Predictions made for {len(X)} samples")
            return predictions
        except Exception as e:
            self.logger.error(f"Error during prediction: {str(e)}")
            raise

    def predict_proba(self, X):
        """
        Predict class probabilities

        Args:
            X: Features to predict

        Returns:
            Predicted probabilities
        """
        try:
            probabilities = self.xgb_classifier.predict_proba(X=X)
            self.logger.info(f"Probabilities predicted for {len(X)} samples")
            return probabilities
        except Exception as e:
            self.logger.error(f"Error during probability prediction: {str(e)}")
            raise

    def evaluate(self, y_true, y_pred):
        """
        Evaluate model performance

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Dictionary containing evaluation metrics
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
        try:
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }

            self.logger.info("Model Evaluation Results:")
            self.logger.info(f"Accuracy: {accuracy:.4f}")
            self.logger.info(f"Precision: {precision:.4f}")
            self.logger.info(f"Recall: {recall:.4f}")
            self.logger.info(f"F1-Score: {f1:.4f}")

            report = classification_report(y_true, y_pred, zero_division=0)
            self.logger.info(f"Classification Report:\n{report}")

            cm = confusion_matrix(y_true, y_pred)
            self.logger.info(f"Confusion Matrix:\n{cm}")

            return metrics
        except Exception as e:
            self.logger.error(f"Error during evaluation: {str(e)}")
            raise

    def save_model(self, model_path: str):
        """
        Save the trained model to disk

        Args:
            model_path: Path to save the model
        """
        try:
            make_dirs(path=os.path.dirname(model_path))
            with open(model_path, 'wb') as f:
                pickle.dump(self.xgb_classifier, f)
            self.model_path = model_path
            self.logger.info(f"Model saved successfully to: {sanitize_path(model_path)}")
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self, model_path: str):
        """
        Load a trained model from the disk

        Args:
            model_path: Path to the saved model
        """
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {sanitize_path(model_path)}")
            with open(model_path, 'rb') as f:
                self.xgb_classifier = pickle.load(f)
            self.model_path = model_path
            self.logger.info(f"Model loaded successfully from: {sanitize_path(model_path)}")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

    def get_model_info(self):
        """
        Get information about the current model

        Returns:
            Dictionary with model information
        """
        info = self.model_params.copy()
        info['model_path'] = sanitize_path(self.model_path) if self.model_path else None
        return info

    def cross_validate(self, X, y, cv=5):
        """
        Perform cross-validation

        Args:
            X: Features
            y: Labels
            cv: Number of cross-validation folds

        Returns:
            Cross-validation scores
        """
        from sklearn.model_selection import cross_val_score
        try:
            scores = cross_val_score(estimator=self.xgb_classifier, X=X, y=y, cv=cv, scoring='accuracy')
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            self.logger.info("Cross-validation results:")
            self.logger.info(f"Mean accuracy: {mean_score:.4f} (+/- {std_score * 2:.4f})")
            return scores
        except Exception as e:
            self.logger.error(f"Error during cross-validation: {str(e)}")
            raise

def load_trained_model_and_scaler():
    """Load trained XGBoost model and scaler"""
    try:
        # Load pretrained model
        model = DiabetesXGBoostClassifier()
        model.load_model(model_path="https://media.githubusercontent.com/media/lngquoctrung/diabetes-prediction/refs/heads/main/artifacts/models/best_f1_model.pkl")
        
        # Load scaler
        scaler = load_data(path="https://media.githubusercontent.com/media/lngquoctrung/diabetes-prediction/refs/heads/main/artifacts/scalers/best_f1_model_scaler.pkl")
        
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
