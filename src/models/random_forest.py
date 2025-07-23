import sys
from pathlib import Path

# Add the project path into the python path
root_dir = str(Path(__file__).parent.parent.parent.absolute())
if not root_dir in sys.path:
    sys.path.insert(0, root_dir)

import logging
import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix
from src.config import LOG_FORMAT, RANDOM_STATE
from src.utils import make_dirs


class DiabetesRandomForest:
    def __init__(self,
                 n_estimators=100,
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 max_features='sqrt',
                 random_state=RANDOM_STATE,
                 log_file: str = None,
                 log_format: str | None = LOG_FORMAT):
        """
        Initialize a Random Forest model with essential parameters

        Args:
            n_estimators: Number of trees in the forest (default: 100)
            max_depth: Maximum depth of the tree (None means unlimited)
            min_samples_split: Minimum samples required to split an internal node
            min_samples_leaf: Minimum samples required to be at a leaf node
            max_features: Number of features to consider when looking for the best split
            random_state: Random state for reproducibility
            log_file: Path to log file
            log_format: Log format string
        """

        # Initialize Random Forest with essential parameters
        self.random_forest = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            random_state=random_state
        )

        # Store parameters for reference
        self.model_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features,
            'random_state': random_state
        }

        self.model_path = None

        # Log configuration
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        log_formatter = logging.Formatter(log_format)

        # Clear existing handlers to avoid duplicate logs
        self.logger.handlers.clear()

        # Handler log to file
        if log_file:
            # Create a log directory
            make_dirs(os.path.dirname(log_file))
            log_file_handler = logging.FileHandler(log_file)
            log_file_handler.setLevel(logging.INFO)
            log_file_handler.setFormatter(log_formatter)
            self.logger.addHandler(log_file_handler)

        # Handler log to console
        log_console_handler = logging.StreamHandler()
        log_console_handler.setLevel(logging.INFO)
        log_console_handler.setFormatter(log_formatter)
        self.logger.addHandler(log_console_handler)

        # Log model initialization
        self.logger.info(f"DiabetesRandomForest initialized with parameters:")
        for key, value in self.model_params.items():
            self.logger.info(f"  {key}: {value}")

    def train(self, X, y):
        """
        Train the Random Forest model

        Args:
            X: Training features
            y: Training labels
        """
        try:
            self.logger.info("Starting Random Forest model training...")
            self.random_forest.fit(X, y)
            self.logger.info(f"Model training completed. Training samples: {len(X)}")
            self.logger.info(f"Number of trees: {self.random_forest.n_estimators}")
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
            predictions = self.random_forest.predict(X)
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
            probabilities = self.random_forest.predict_proba(X)
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
        try:
            # Calculate metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

            # Create a metrics dictionary
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }

            # Log metrics
            self.logger.info("Model Evaluation Results:")
            self.logger.info(f"Accuracy: {accuracy:.4f}")
            self.logger.info(f"Precision: {precision:.4f}")
            self.logger.info(f"Recall: {recall:.4f}")
            self.logger.info(f"F1-Score: {f1:.4f}")

            # Detailed classification report
            report = classification_report(y_true, y_pred)
            self.logger.info(f"Classification Report:\n{report}")

            # Confusion Matrix
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
            # Create a directory if it doesn't exist
            make_dirs(os.path.dirname(model_path))

            # Save the model
            with open(model_path, 'wb') as f:
                pickle.dump(self.random_forest, f)

            self.model_path = model_path
            self.logger.info(f"Model saved successfully to: {model_path}")

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
                raise FileNotFoundError(f"Model file not found: {model_path}")

            # Load the model
            with open(model_path, 'rb') as f:
                self.random_forest = pickle.load(f)

            self.model_path = model_path
            self.logger.info(f"Model loaded successfully from: {model_path}")

        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

    def get_model_info(self):
        """
        Get information about the current model

        Returns:
            Dictionary with model information
        """
        info = {
            'n_estimators': self.random_forest.n_estimators,
            'max_depth': self.random_forest.max_depth,
            'min_samples_split': self.random_forest.min_samples_split,
            'min_samples_leaf': self.random_forest.min_samples_leaf,
            'max_features': self.random_forest.max_features,
            'random_state': self.random_forest.random_state,
            'model_path': self.model_path
        }

        # Add feature info if the model is trained
        if hasattr(self.random_forest, 'feature_importances_'):
            info['n_features'] = self.random_forest.n_features_in_
            info['n_classes'] = self.random_forest.n_classes_

        return info

    def get_feature_importance(self):
        """
        Get feature importance from the random forest model

        Returns:
            Feature importance array
        """
        try:
            if hasattr(self.random_forest, 'feature_importances_'):
                feature_importance = self.random_forest.feature_importances_
                self.logger.info("Feature importance extracted successfully")
                return feature_importance
            else:
                self.logger.warning("Model has not been trained yet")
                return None
        except Exception as e:
            self.logger.error(f"Error getting feature importance: {str(e)}")
            raise

    def get_feature_importance_ranking(self, feature_names=None):
        """
        Get feature importance ranking

        Args:
            feature_names: List of feature names

        Returns:
            Sorted list of (feature_name/index, importance) tuples
        """
        try:
            if not hasattr(self.random_forest, 'feature_importances_'):
                self.logger.warning("Model has not been trained yet")
                return None

            importances = self.random_forest.feature_importances_

            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(len(importances))]

            # Create a list of (feature_name, importance) tuples and sort by importance
            feature_importance_pairs = list(zip(feature_names, importances))
            feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)

            self.logger.info("Feature importance ranking:")
            for i, (name, importance) in enumerate(feature_importance_pairs[:10]):  # Top 10
                self.logger.info(f"  {i + 1}. {name}: {importance:.4f}")

            return feature_importance_pairs

        except Exception as e:
            self.logger.error(f"Error getting feature importance ranking: {str(e)}")
            raise

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
        try:
            from sklearn.model_selection import cross_val_score

            scores = cross_val_score(self.random_forest, X, y, cv=cv, scoring='accuracy')
            mean_score = np.mean(scores)
            std_score = np.std(scores)

            self.logger.info(f"Cross-validation results:")
            self.logger.info(f"Mean accuracy: {mean_score:.4f} (+/- {std_score * 2:.4f})")

            return scores

        except Exception as e:
            self.logger.error(f"Error during cross-validation: {str(e)}")
            raise