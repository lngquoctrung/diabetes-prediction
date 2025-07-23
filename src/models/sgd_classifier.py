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
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix
from src.config import LOG_FORMAT, RANDOM_STATE
from src.utils import make_dirs


class DiabetesSGDClassifier:
    def __init__(self,
                 loss='hinge',
                 penalty='l2',
                 alpha=0.0001,
                 learning_rate='constant',
                 eta0=0.01,
                 max_iter=1000,
                 random_state=RANDOM_STATE,
                 log_file: str = None,
                 log_format: str | None = LOG_FORMAT):
        """
        Initialize SGD Classifier model with essential parameters

        Args:
            loss: Loss function ('hinge', 'log', 'squared_hinge', 'modified_huber', 'perceptron')
            penalty: Regularization penalty ('l2', 'l1', 'elasticnet', None)
            alpha: Regularization strength (default: 0.0001)
            learning_rate: Learning rate schedule ('constant', 'optimal', 'invscaling', 'adaptive')
            eta0: Initial learning rate (default: 0.01)
            max_iter: Maximum number of passes over training data
            random_state: Random state for reproducibility
            log_file: Path to log file
            log_format: Log format string
        """

        # Initialize SGD Classifier with essential parameters
        self.sgd_classifier = SGDClassifier(
            loss=loss,
            penalty=penalty,
            alpha=alpha,
            learning_rate=learning_rate,
            eta0=eta0,
            max_iter=max_iter,
            random_state=random_state
        )

        # Store parameters for reference
        self.model_params = {
            'loss': loss,
            'penalty': penalty,
            'alpha': alpha,
            'learning_rate': learning_rate,
            'eta0': eta0,
            'max_iter': max_iter,
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
        self.logger.info(f"DiabetesSGDClassifier initialized with parameters:")
        for key, value in self.model_params.items():
            self.logger.info(f"  {key}: {value}")

    def train(self, X, y):
        """
        Train the SGD Classifier model

        Args:
            X: Training features
            y: Training labels
        """
        try:
            self.logger.info("Starting SGD Classifier model training...")
            self.sgd_classifier.fit(X, y)
            self.logger.info(f"Model training completed. Training samples: {len(X)}")
            if hasattr(self.sgd_classifier, 'n_iter_'):
                self.logger.info(f"Number of iterations: {self.sgd_classifier.n_iter_}")
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
            predictions = self.sgd_classifier.predict(X)
            self.logger.info(f"Predictions made for {len(X)} samples")
            return predictions
        except Exception as e:
            self.logger.error(f"Error during prediction: {str(e)}")
            raise

    def predict_proba(self, X):
        """
        Predict class probabilities (only available for certain loss functions)

        Args:
            X: Features to predict

        Returns:
            Predicted probabilities
        """
        try:
            if self.sgd_classifier.loss not in ['log_loss', 'modified_huber']:
                self.logger.warning(
                    "Probability prediction not available for this loss function. Use 'log' or 'modified_huber'")
                return None

            probabilities = self.sgd_classifier.predict_proba(X)
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

            # Create metrics dictionary
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
            # Create directory if it doesn't exist
            make_dirs(os.path.dirname(model_path))

            # Save the model
            with open(model_path, 'wb') as f:
                pickle.dump(self.sgd_classifier, f)

            self.model_path = model_path
            self.logger.info(f"Model saved successfully to: {model_path}")

        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self, model_path: str):
        """
        Load a trained model from disk

        Args:
            model_path: Path to the saved model
        """
        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")

            # Load the model
            with open(model_path, 'rb') as f:
                self.sgd_classifier = pickle.load(f)

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
            'loss': self.sgd_classifier.loss,
            'penalty': self.sgd_classifier.penalty,
            'alpha': self.sgd_classifier.alpha,
            'learning_rate': self.sgd_classifier.learning_rate,
            'eta0': self.sgd_classifier.eta0,
            'max_iter': self.sgd_classifier.max_iter,
            'random_state': self.sgd_classifier.random_state,
            'model_path': self.model_path
        }

        # Add training info if model is trained
        if hasattr(self.sgd_classifier, 'coef_'):
            info['n_features'] = len(self.sgd_classifier.coef_[0])
            info['is_fitted'] = True
            if hasattr(self.sgd_classifier, 'n_iter_'):
                info['n_iterations'] = self.sgd_classifier.n_iter_
        else:
            info['is_fitted'] = False

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
        try:
            from sklearn.model_selection import cross_val_score

            scores = cross_val_score(self.sgd_classifier, X, y, cv=cv, scoring='accuracy')
            mean_score = np.mean(scores)
            std_score = np.std(scores)

            self.logger.info(f"Cross-validation results:")
            self.logger.info(f"Mean accuracy: {mean_score:.4f} (+/- {std_score * 2:.4f})")

            return scores

        except Exception as e:
            self.logger.error(f"Error during cross-validation: {str(e)}")
            raise