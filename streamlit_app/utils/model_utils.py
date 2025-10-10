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

@st.cache_resource
def load_trained_model_and_scaler():
    """Load trained XGBoost model and scaler"""
    try:
        # Load pretrained model
        model = DiabetesXGBoostClassifier()
        model.load_model(model_path="~/service/public/diabetes-prediction-app/xgboost_model_checkpoint.pkl")

        # Load scaler
        scaler = load_data(path="~/services/public/diabetes-prediction-app/xgb_min_max_scaler.pkl")
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
