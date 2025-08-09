import sys
from pathlib import Path
# Add the project path into the python path
root_dir = str(Path(__file__).parent.parent.absolute())
if not root_dir in sys.path:
    sys.path.insert(0, root_dir)

import logging
import os
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from src.config import LOG_FORMAT, RANDOM_STATE
from src.utils import make_dirs

class DiabetesLogisticRegression:
    def __init__(self,
                 C=1.0,
                 penalty='l2',
                 solver='lbfgs',
                 class_weight=None,
                 max_iter=100,
                 verbose=0,
                 random_state=RANDOM_STATE,
                 n_jobs=None,
                 log_file: str = None,
                 log_format: str | None = LOG_FORMAT):
        """
        Initialize a Logistic Regression model with essential parameters

        Args:
            C: Inverse of regularization strength (default: 1.0)
            penalty: Regularization penalty ('l1', 'l2', 'elasticnet', 'none')
            solver: Algorithm to use ('lbfgs', 'liblinear', 'saga', 'newton-cg', 'sag')
            class_weight: Weights associated with classes in the form ``{class_label: weight}`` (default: None)
            max_iter: Maximum number of iterations (default: 100)
            verbose: Verbosity level (default: 0)
            random_state: Random state for reproducibility
            n_jobs: Number of parallel jobs to run (default: None)
            log_file: Path to log file
            log_format: Log format string
        """

        # Initialize Logistic Regression with essential parameters
        self.logistic_regression = LogisticRegression(
            C=C,
            penalty=penalty,
            class_weight=class_weight,
            solver=solver,
            max_iter=max_iter,
            verbose=verbose,
            random_state=random_state,
            n_jobs=n_jobs,
        )

        # Store parameters for reference
        self.model_params = {
            'C': C,
            'penalty': penalty,
            'class_weight': class_weight,
            'solver': solver,
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
        self.logger.info(f"DiabetesLogisticRegression initialized with parameters:")
        for key, value in self.model_params.items():
            self.logger.info(f"  {key}: {value}")

    def train(self, X, y):
        """
        Train the Logistic Regression model

        Args:
            X: Training features
            y: Training labels
        """
        try:
            self.logger.info("Starting Logistic Regression model training...")
            self.logistic_regression.fit(X, y)
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
            predictions = self.logistic_regression.predict(X)
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
            probabilities = self.logistic_regression.predict_proba(X)
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
            precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

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
            report = classification_report(y_true, y_pred, zero_division=0)
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
                pickle.dump(self.logistic_regression, f)

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
                self.logistic_regression = pickle.load(f)

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
            'C': self.logistic_regression.C,
            'penalty': self.logistic_regression.penalty,
            'solver': self.logistic_regression.solver,
            'max_iter': self.logistic_regression.max_iter,
            'random_state': self.logistic_regression.random_state,
            'model_path': self.model_path
        }
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

            scores = cross_val_score(self.logistic_regression, X, y, cv=cv, scoring='accuracy')
            mean_score = np.mean(scores)
            std_score = np.std(scores)

            self.logger.info(f"Cross-validation results:")
            self.logger.info(f"Mean accuracy: {mean_score:.4f} (+/- {std_score * 2:.4f})")

            return scores

        except Exception as e:
            self.logger.error(f"Error during cross-validation: {str(e)}")
            raise

class DiabetesNaiveBayes:
    def __init__(self,
                 nb_type='gaussian',
                 var_smoothing=1e-09,
                 alpha=1.0,
                 binarize=0.0,
                 log_file: str = None,
                 log_format: str | None = LOG_FORMAT):
        """
        Initialize a Naive Bayes model with essential parameters

        Args:
            nb_type: Type of Naive Bayes ('gaussian', 'multinomial', 'bernoulli')
            var_smoothing: Smoothing parameter for Gaussian NB (default: 1e-09)
            alpha: Smoothing parameter for Multinomial/Bernoulli NB (default: 1.0)
            binarize: Threshold for binarizing features in Bernoulli NB (default: 0.0)
            log_file: Path to log file
            log_format: Log format string
        """

        self.nb_type = nb_type

        # Initialize the appropriate Naive Bayes classifier based on type
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
            precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

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
            report = classification_report(y_true, y_pred, zero_division=0)
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
                pickle.dump(self.naive_bayes, f)

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

        # Add training info if the model is trained
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
            Best parameters and the best score
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
            precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

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
            report = classification_report(y_true, y_pred, zero_division=0)
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

class DiabetesSGDClassifier:
    def __init__(self,
                 loss='hinge',
                 penalty='l2',
                 alpha=0.0001,
                 learning_rate='constant',
                 eta0=0.01,
                 max_iter=1000,
                 random_state=RANDOM_STATE,
                 n_jobs=None,
                 log_file: str = None,
                 log_format: str | None = LOG_FORMAT):
        """
        Initialize an SGD Classifier model with essential parameters

        Args:
            loss: Loss function ('hinge', 'log', 'squared_hinge', 'modified_huber', 'perceptron')
            penalty: Regularization penalty ('l2', 'l1', 'elasticnet', None)
            alpha: Regularization strength (default: 0.0001)
            learning_rate: Learning rate schedule ('constant', 'optimal', 'invscaling', 'adaptive')
            eta0: Initial learning rate (default: 0.01)
            max_iter: Maximum number of passes over training data
            random_state: Random state for reproducibility
            n_jobs: Number of CPU cores to use (default: None)
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
            random_state=random_state,
            n_jobs=n_jobs,
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
            precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

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
            report = classification_report(y_true, y_pred, zero_division=0)
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
                pickle.dump(self.sgd_classifier, f)

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

        # Add training info if the model is trained
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