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
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix
from src.config import LOG_FORMAT
from src.utils import make_dirs


class DiabetesNaiveBayes:
    def __init__(self,
                 nb_type='gaussian',
                 var_smoothing=1e-09,
                 alpha=1.0,
                 binarize=0.0,
                 log_file: str = None,
                 log_format: str | None = LOG_FORMAT):
        """
        Initialize Naive Bayes model with essential parameters

        Args:
            nb_type: Type of Naive Bayes ('gaussian', 'multinomial', 'bernoulli')
            var_smoothing: Smoothing parameter for Gaussian NB (default: 1e-09)
            alpha: Smoothing parameter for Multinomial/Bernoulli NB (default: 1.0)
            binarize: Threshold for binarizing features in Bernoulli NB (default: 0.0)
            log_file: Path to log file
            log_format: Log format string
        """

        self.nb_type = nb_type

        # Initialize appropriate Naive Bayes classifier based on type
        if nb_type == 'gaussian':
            self.naive_bayes = GaussianNB(var_smoothing=var_smoothing)
            self.model_params = {
                'nb_type': nb_type,
                'var_smoothing': var_smoothing
            }
        elif nb_type == 'multinomial':
            self.naive_bayes = MultinomialNB(alpha=alpha)
            self.model_params = {
                'nb_type': nb_type,
                'alpha': alpha
            }
        elif nb_type == 'bernoulli':
            self.naive_bayes = BernoulliNB(alpha=alpha, binarize=binarize)
            self.model_params = {
                'nb_type': nb_type,
                'alpha': alpha,
                'binarize': binarize
            }
        else:
            raise ValueError("nb_type must be 'gaussian', 'multinomial', or 'bernoulli'")

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
        self.logger.info(f"DiabetesNaiveBayes ({nb_type.title()}) initialized with parameters:")
        for key, value in self.model_params.items():
            self.logger.info(f"  {key}: {value}")

    def train(self, X, y):
        """
        Train the Naive Bayes model

        Args:
            X: Training features
            y: Training labels
        """
        try:
            self.logger.info(f"Starting {self.nb_type.title()} Naive Bayes model training...")
            self.naive_bayes.fit(X, y)
            self.logger.info(f"Model training completed. Training samples: {len(X)}")

            # Log class information if available
            if hasattr(self.naive_bayes, 'classes_'):
                self.logger.info(f"Number of classes: {len(self.naive_bayes.classes_)}")
                self.logger.info(f"Classes: {self.naive_bayes.classes_}")

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
            predictions = self.naive_bayes.predict(X)
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
            probabilities = self.naive_bayes.predict_proba(X)
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
                pickle.dump(self.naive_bayes, f)

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
                self.naive_bayes = pickle.load(f)

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
            **self.model_params,
            'model_path': self.model_path
        }

        # Add training info if model is trained
        if hasattr(self.naive_bayes, 'classes_'):
            info['n_classes'] = len(self.naive_bayes.classes_)
            info['classes'] = self.naive_bayes.classes_.tolist()
            info['is_fitted'] = True

            if hasattr(self.naive_bayes, 'n_features_in_'):
                info['n_features'] = self.naive_bayes.n_features_in_
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

            scores = cross_val_score(self.naive_bayes, X, y, cv=cv, scoring='accuracy')
            mean_score = np.mean(scores)
            std_score = np.std(scores)

            self.logger.info(f"Cross-validation results:")
            self.logger.info(f"Mean accuracy: {mean_score:.4f} (+/- {std_score * 2:.4f})")

            return scores

        except Exception as e:
            self.logger.error(f"Error during cross-validation: {str(e)}")
            raise

    def grid_search(self, X, y, param_grid=None, cv=5):
        """
        Perform grid search for hyperparameter tuning

        Args:
            X: Features
            y: Labels
            param_grid: Dictionary of parameters to search
            cv: Number of cross-validation folds

        Returns:
            Best parameters and best score
        """
        try:
            from sklearn.model_selection import GridSearchCV

            if param_grid is None:
                if self.nb_type == 'gaussian':
                    param_grid = {
                        'var_smoothing': [1e-10, 1e-09, 1e-08, 1e-07, 1e-06]
                    }
                elif self.nb_type in ['multinomial', 'bernoulli']:
                    param_grid = {
                        'alpha': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
                    }
                    if self.nb_type == 'bernoulli':
                        param_grid['binarize'] = [0.0, 0.5, 1.0]

            self.logger.info("Starting grid search for hyperparameter tuning...")

            grid_search = GridSearchCV(
                self.naive_bayes,
                param_grid,
                cv=cv,
                scoring='accuracy',
                n_jobs=-1
            )

            grid_search.fit(X, y)

            # Update model with best parameters
            self.naive_bayes = grid_search.best_estimator_

            self.logger.info(f"Grid search completed")
            self.logger.info(f"Best parameters: {grid_search.best_params_}")
            self.logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")

            return grid_search.best_params_, grid_search.best_score_

        except Exception as e:
            self.logger.error(f"Error during grid search: {str(e)}")
            raise
