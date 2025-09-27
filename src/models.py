import sys
from pathlib import Path
# Add the project path into the python path
root_dir = str(Path(__file__).parent.parent.absolute())
if not root_dir in sys.path:
    sys.path.insert(0, root_dir)

import os
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from src.config import LOG_FORMAT, RANDOM_STATE
from src.utils import make_dirs, sanitize_path, get_configured_logger

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
                 logger_name: str | None = __name__,
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
        self.logger = get_configured_logger(
            name=logger_name, 
            log_file=log_file, 
            log_format=log_format
        )

        # Log model initialization
        self.logger.info(msg=f"DiabetesLogisticRegression initialized with parameters:")
        for key, value in self.model_params.items():
            self.logger.info(msg=f"  {key}: {value}")

    def train(self, X, y):
        """
        Train the Logistic Regression model

        Args:
            X: Training features
            y: Training labels
        """
        try:
            self.logger.info(msg="Starting Logistic Regression model training...")
            self.logistic_regression.fit(X=X, y=y)
            self.logger.info(msg=f"Model training completed. Training samples: {len(X)}")
        except Exception as e:
            self.logger.error(msg=f"Error during training: {str(e)}")
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
            predictions = self.logistic_regression.predict(X=X)
            self.logger.info(msg=f"Predictions made for {len(X)} samples")
            return predictions
        except Exception as e:
            self.logger.error(msg=f"Error during prediction: {str(e)}")
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
            probabilities = self.logistic_regression.predict_proba(X=X)
            self.logger.info(msg=f"Probabilities predicted for {len(X)} samples")
            return probabilities
        except Exception as e:
            self.logger.error(msg=f"Error during probability prediction: {str(e)}")
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
            accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
            precision = precision_score(y_true=y_true, y_pred=y_pred, average='macro', zero_division=0)
            recall = recall_score(y_true=y_true, y_pred=y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_true=y_true, y_pred=y_pred, average='macro', zero_division=0)

            # Create a metrics dictionary
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }

            # Log metrics
            self.logger.info(msg="Model Evaluation Results:")
            self.logger.info(msg=f"Accuracy: {accuracy:.4f}")
            self.logger.info(msg=f"Precision: {precision:.4f}")
            self.logger.info(msg=f"Recall: {recall:.4f}")
            self.logger.info(msg=f"F1-Score: {f1:.4f}")

            # Detailed classification report
            report = classification_report(y_true=y_true,y_pred= y_pred, zero_division=0)
            self.logger.info(msg=f"Classification Report:\n{report}")

            # Confusion Matrix
            cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
            self.logger.info(msg=f"Confusion Matrix:\n{cm}")

            return metrics

        except Exception as e:
            self.logger.error(msg=f"Error during evaluation: {str(e)}")
            raise

    def save_model(self, model_path: str):
        """
        Save the trained model to disk

        Args:
            model_path: Path to save the model
        """
        try:
            # Create a directory if it doesn't exist
            make_dirs(path=os.path.dirname(model_path))

            # Save the model
            with open(file=model_path, mode='wb') as f:
                pickle.dump(self.logistic_regression, f)

            self.model_path = model_path
            self.logger.info(msg=f"Model saved successfully to: {sanitize_path(model_path)}")

        except Exception as e:
            self.logger.error(msg=f"Error saving model: {str(e)}")
            raise

    def load_model(self, model_path: str):
        """
        Load a trained model from the disk

        Args:
            model_path: Path to the saved model
        """
        try:
            if not os.path.exists(path=model_path):
                raise FileNotFoundError(f"Model file not found: {sanitize_path(model_path)}")

            # Load the model
            with open(file=model_path, mode='rb') as f:
                self.logistic_regression = pickle.load(f)

            self.model_path = model_path
            self.logger.info(msg=f"Model loaded successfully from: {sanitize_path(model_path)}")

        except Exception as e:
            self.logger.error(msg=f"Error loading model: {str(e)}")
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
            'model_path': sanitize_path(self.model_path) if self.model_path else None
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

            scores = cross_val_score(estimator=self.logistic_regression, X=X, y=y, cv=cv, scoring='accuracy')
            mean_score = np.mean(scores)
            std_score = np.std(scores)

            self.logger.info(msg=f"Cross-validation results:")
            self.logger.info(msg=f"Mean accuracy: {mean_score:.4f} (+/- {std_score * 2:.4f})")

            return scores

        except Exception as e:
            self.logger.error(msg=f"Error during cross-validation: {str(e)}")
            raise

class DiabetesNaiveBayes:
    def __init__(self,
                 nb_type='gaussian',
                 var_smoothing=1e-09,
                 alpha=1.0,
                 binarize=0.0,
                 logger_name: str | None = __name__,
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
        self.logger = get_configured_logger(
            name=logger_name, 
            log_file=log_file, 
            log_format=log_format
        )

        # Log model initialization
        self.logger.info(msg=f"DiabetesNaiveBayes ({nb_type.title()}) initialized with parameters:")
        for key, value in self.model_params.items():
            self.logger.info(msg=f"  {key}: {value}")

    def train(self, X, y):
        """
        Train the Naive Bayes model

        Args:
            X: Training features
            y: Training labels
        """
        try:
            self.logger.info(msg=f"Starting {self.nb_type.title()} Naive Bayes model training...")
            self.naive_bayes.fit(X=X, y=y)
            self.logger.info(msg=f"Model training completed. Training samples: {len(X)}")

            # Log class information if available
            if hasattr(self.naive_bayes, 'classes_'):
                self.logger.info(msg=f"Number of classes: {len(self.naive_bayes.classes_)}")
                self.logger.info(msg=f"Classes: {self.naive_bayes.classes_}")

        except Exception as e:
            self.logger.error(msg=f"Error during training: {str(e)}")
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
            predictions = self.naive_bayes.predict(X=X)
            self.logger.info(msg=f"Predictions made for {len(X)} samples")
            return predictions
        except Exception as e:
            self.logger.error(msg=f"Error during prediction: {str(e)}")
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
            probabilities = self.naive_bayes.predict_proba(X=X)
            self.logger.info(msg=f"Probabilities predicted for {len(X)} samples")
            return probabilities
        except Exception as e:
            self.logger.error(msg=f"Error during probability prediction: {str(e)}")
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
            accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
            precision = precision_score(y_true=y_true, y_pred=y_pred, average='macro', zero_division=0)
            recall = recall_score(y_true=y_true, y_pred=y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_true=y_true, y_pred=y_pred, average='macro', zero_division=0)

            # Create a metrics dictionary
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }

            # Log metrics
            self.logger.info(msg="Model Evaluation Results:")
            self.logger.info(msg=f"Accuracy: {accuracy:.4f}")
            self.logger.info(msg=f"Precision: {precision:.4f}")
            self.logger.info(msg=f"Recall: {recall:.4f}")
            self.logger.info(msg=f"F1-Score: {f1:.4f}")

            # Detailed classification report
            report = classification_report(y_true=y_true, y_pred=y_pred, zero_division=0)
            self.logger.info(msg=f"Classification Report:\n{report}")

            # Confusion Matrix
            cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
            self.logger.info(msg=f"Confusion Matrix:\n{cm}")

            return metrics

        except Exception as e:
            self.logger.error(msg=f"Error during evaluation: {str(e)}")
            raise

    def save_model(self, model_path: str):
        """
        Save the trained model to disk

        Args:
            model_path: Path to save the model
        """
        try:
            # Create a directory if it doesn't exist
            make_dirs(path=os.path.dirname(model_path))

            # Save the model
            with open(model_path, 'wb') as f:
                pickle.dump(self.naive_bayes, f)

            self.model_path = model_path
            self.logger.info(msg=f"Model saved successfully to: {sanitize_path(model_path)}")

        except Exception as e:
            self.logger.error(msg=f"Error saving model: {str(e)}")
            raise

    def load_model(self, model_path: str):
        """
        Load a trained model from the disk

        Args:
            model_path: Path to the saved model
        """
        try:
            if not os.path.exists(path=model_path):
                raise FileNotFoundError(f"Model file not found: {sanitize_path(model_path)}")

            # Load the model
            with open(file=model_path, mode='rb') as f:
                self.naive_bayes = pickle.load(f)

            self.model_path = model_path
            self.logger.info(msg=f"Model loaded successfully from: {sanitize_path(model_path)}")

        except Exception as e:
            self.logger.error(msg=f"Error loading model: {str(e)}")
            raise

    def get_model_info(self):
        """
        Get information about the current model

        Returns:
            Dictionary with model information
        """
        info = {
            **self.model_params,
            'model_path': sanitize_path(self.model_path) if self.model_path else None
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

            scores = cross_val_score(estimator=self.naive_bayes, X=X, y=y, cv=cv, scoring='accuracy')
            mean_score = np.mean(scores)
            std_score = np.std(scores)

            self.logger.info(msg=f"Cross-validation results:")
            self.logger.info(msg=f"Mean accuracy: {mean_score:.4f} (+/- {std_score * 2:.4f})")

            return scores

        except Exception as e:
            self.logger.error(msg=f"Error during cross-validation: {str(e)}")
            raise

class DiabetesRandomForest:
    def __init__(self,
                 n_estimators=100,
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 max_features='sqrt',
                 class_weight='balanced',
                 random_state=RANDOM_STATE,
                 logger_name: str | None = __name__,
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
            class_weight: The weight of samples in the training set for balancing the model (default: 'balanced')
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
            class_weight=class_weight,
            random_state=random_state,
        )

        # Store parameters for reference
        self.model_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features,
            'class_weight': class_weight,
            'random_state': random_state
        }

        self.model_path = None

        # Log configuration
        self.logger = get_configured_logger(
            name=logger_name, 
            log_file=log_file, 
            log_format=log_format
        )

        # Log model initialization
        self.logger.info(msg=f"DiabetesRandomForest initialized with parameters:")
        for key, value in self.model_params.items():
            self.logger.info(msg=f"  {key}: {value}")

    def train(self, X, y):
        """
        Train the Random Forest model

        Args:
            X: Training features
            y: Training labels
        """
        try:
            self.logger.info(msg="Starting Random Forest model training...")
            self.random_forest.fit(X=X, y=y)
            self.logger.info(msg=f"Model training completed. Training samples: {len(X)}")
            self.logger.info(msg=f"Number of trees: {self.random_forest.n_estimators}")
        except Exception as e:
            self.logger.error(msg=f"Error during training: {str(e)}")
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
            predictions = self.random_forest.predict(X=X)
            self.logger.info(msg=f"Predictions made for {len(X)} samples")
            return predictions
        except Exception as e:
            self.logger.error(msg=f"Error during prediction: {str(e)}")
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
            probabilities = self.random_forest.predict_proba(X=X)
            self.logger.info(msg=f"Probabilities predicted for {len(X)} samples")
            return probabilities
        except Exception as e:
            self.logger.error(msg=f"Error during probability prediction: {str(e)}")
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
            accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
            precision = precision_score(y_true=y_true, y_pred=y_pred, average='macro', zero_division=0)
            recall = recall_score(y_true=y_true, y_pred=y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_true=y_true, y_pred=y_pred, average='macro', zero_division=0)

            # Create a metrics dictionary
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }

            # Log metrics
            self.logger.info(msg="Model Evaluation Results:")
            self.logger.info(msg=f"Accuracy: {accuracy:.4f}")
            self.logger.info(msg=f"Precision: {precision:.4f}")
            self.logger.info(msg=f"Recall: {recall:.4f}")
            self.logger.info(msg=f"F1-Score: {f1:.4f}")

            # Detailed classification report
            report = classification_report(y_true=y_true, y_pred=y_pred, zero_division=0)
            self.logger.info(msg=f"Classification Report:\n{report}")

            # Confusion Matrix
            cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
            self.logger.info(msg=f"Confusion Matrix:\n{cm}")

            return metrics

        except Exception as e:
            self.logger.error(msg=f"Error during evaluation: {str(e)}")
            raise

    def save_model(self, model_path: str):
        """
        Save the trained model to disk

        Args:
            model_path: Path to save the model
        """
        try:
            # Create a directory if it doesn't exist
            make_dirs(path=os.path.dirname(model_path))

            # Save the model
            with open(file=model_path, mode='wb') as f:
                pickle.dump(self.random_forest, f)

            self.model_path = model_path
            self.logger.info(msg=f"Model saved successfully to: {sanitize_path(model_path)}")

        except Exception as e:
            self.logger.error(msg=f"Error saving model: {str(e)}")
            raise

    def load_model(self, model_path: str):
        """
        Load a trained model from the disk

        Args:
            model_path: Path to the saved model
        """
        try:
            if not os.path.exists(path=model_path):
                raise FileNotFoundError(f"Model file not found: {sanitize_path(model_path)}")

            # Load the model
            with open(file=model_path, mode='rb') as f:
                self.random_forest = pickle.load(f)

            self.model_path = model_path
            self.logger.info(msg=f"Model loaded successfully from: {sanitize_path(model_path)}")

        except Exception as e:
            self.logger.error(msg=f"Error loading model: {str(e)}")
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
            'model_path': sanitize_path(self.model_path) if self.model_path else None
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
                self.logger.info(msg="Feature importance extracted successfully")
                return feature_importance
            else:
                self.logger.warning(msg="Model has not been trained yet")
                return None
        except Exception as e:
            self.logger.error(msg=f"Error getting feature importance: {str(e)}")
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
                self.logger.warning(msg="Model has not been trained yet")
                return None

            importances = self.random_forest.feature_importances_

            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(len(importances))]

            # Create a list of (feature_name, importance) tuples and sort by importance
            feature_importance_pairs = list(zip(feature_names, importances))
            feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)

            self.logger.info("Feature importance ranking:")
            for i, (name, importance) in enumerate(feature_importance_pairs[:10]):  # Top 10
                self.logger.info(msg=f"  {i + 1}. {name}: {importance:.4f}")

            return feature_importance_pairs

        except Exception as e:
            self.logger.error(msg=f"Error getting feature importance ranking: {str(e)}")
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

            scores = cross_val_score(estimator=self.random_forest, X=X, y=y, cv=cv, scoring='accuracy')
            mean_score = np.mean(scores)
            std_score = np.std(scores)

            self.logger.info(msg=f"Cross-validation results:")
            self.logger.info(msg=f"Mean accuracy: {mean_score:.4f} (+/- {std_score * 2:.4f})")

            return scores

        except Exception as e:
            self.logger.error(msg=f"Error during cross-validation: {str(e)}")
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
                 logger_name: str | None = __name__,
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
        self.logger = get_configured_logger(
            name=logger_name, 
            log_file=log_file, 
            log_format=log_format
        )

        # Log model initialization
        self.logger.info(msg=f"DiabetesSGDClassifier initialized with parameters:")
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
            self.logger.info(msg="Starting SGD Classifier model training...")
            self.sgd_classifier.fit(X=X, y=y)
            self.logger.info(msg=f"Model training completed. Training samples: {len(X)}")
            if hasattr(self.sgd_classifier, 'n_iter_'):
                self.logger.info(msg=f"Number of iterations: {self.sgd_classifier.n_iter_}")
        except Exception as e:
            self.logger.error(msg=f"Error during training: {str(e)}")
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
            predictions = self.sgd_classifier.predict(X=X)
            self.logger.info(msg=f"Predictions made for {len(X)} samples")
            return predictions
        except Exception as e:
            self.logger.error(msg=f"Error during prediction: {str(e)}")
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
                    msg="Probability prediction not available for this loss function. Use 'log' or 'modified_huber'"
                )
                return None

            probabilities = self.sgd_classifier.predict_proba(X=X)
            self.logger.info(msg=f"Probabilities predicted for {len(X)} samples")
            return probabilities
        except Exception as e:
            self.logger.error(msg=f"Error during probability prediction: {str(e)}")
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
            accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
            precision = precision_score(y_true=y_true, y_pred=y_pred, average='macro', zero_division=0)
            recall = recall_score(y_true=y_true, y_pred=y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_true=y_true, y_pred=y_pred, average='macro', zero_division=0)

            # Create a metrics dictionary
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }

            # Log metrics
            self.logger.info(msg="Model Evaluation Results:")
            self.logger.info(msg=f"Accuracy: {accuracy:.4f}")
            self.logger.info(msg=f"Precision: {precision:.4f}")
            self.logger.info(msg=f"Recall: {recall:.4f}")
            self.logger.info(msg=f"F1-Score: {f1:.4f}")

            # Detailed classification report
            report = classification_report(y_true=y_true, y_pred=y_pred, zero_division=0)
            self.logger.info(msg=f"Classification Report:\n{report}")

            # Confusion Matrix
            cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
            self.logger.info(msg=f"Confusion Matrix:\n{cm}")

            return metrics

        except Exception as e:
            self.logger.error(msg=f"Error during evaluation: {str(e)}")
            raise

    def save_model(self, model_path: str):
        """
        Save the trained model to disk

        Args:
            model_path: Path to save the model
        """
        try:
            # Create a directory if it doesn't exist
            make_dirs(path=os.path.dirname(model_path))

            # Save the model
            with open(file=model_path, mode='wb') as f:
                pickle.dump(self.sgd_classifier, f)

            self.model_path = model_path
            self.logger.info(msg=f"Model saved successfully to: {sanitize_path(model_path)}")

        except Exception as e:
            self.logger.error(msg=f"Error saving model: {str(e)}")
            raise

    def load_model(self, model_path: str):
        """
        Load a trained model from the disk

        Args:
            model_path: Path to the saved model
        """
        try:
            if not os.path.exists(path=model_path):
                raise FileNotFoundError(f"Model file not found: {sanitize_path(model_path)}")

            # Load the model
            with open(file=model_path, mode='rb') as f:
                self.sgd_classifier = pickle.load(f)

            self.model_path = model_path
            self.logger.info(msg=f"Model loaded successfully from: {sanitize_path(model_path)}")

        except Exception as e:
            self.logger.error(msg=f"Error loading model: {str(e)}")
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
            'model_path': sanitize_path(self.model_path) if self.model_path else None
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

            scores = cross_val_score(estimator=self.sgd_classifier, X=X, y=y, cv=cv, scoring='accuracy')
            mean_score = np.mean(scores)
            std_score = np.std(scores)

            self.logger.info(msg=f"Cross-validation results:")
            self.logger.info(msg=f"Mean accuracy: {mean_score:.4f} (+/- {std_score * 2:.4f})")

            return scores

        except Exception as e:
            self.logger.error(msg=f"Error during cross-validation: {str(e)}")
            raise

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

class DiabetesLightGBMClassifier:
    """
    Initialize a LightGBM Classifier model with essential parameters

    Args:
        n_estimators: Number of boosting rounds (default: 100)
        max_depth: Maximum tree depth for base learners (default: -1 means no limit)
        learning_rate: Boosting learning rate (default: 0.1)
        num_leaves: Number of leaves in one tree (default: 31)
        subsample: Subsample ratio of the training instances (default: 1.0)
        colsample_bytree: Subsample ratio of columns when constructing each tree (default: 1.0)
        random_state: Random state for reproducibility
        log_file: Path to log file
        log_format: Log format string
    """
    def __init__(self,
                 n_estimators=100,
                 max_depth=-1,
                 learning_rate=0.1,
                 num_leaves=31,
                 subsample=1.0,
                 colsample_bytree=1.0,
                 random_state=RANDOM_STATE,
                 logger_name: str | None = __name__,
                 log_file: str = None,
                 log_format: str | None = LOG_FORMAT):

        self.lgbm_classifier = LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            force_row_wise=True, 
            random_state=random_state,
        )
        self.model_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'num_leaves': num_leaves,
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
        self.logger.info("DiabetesLightGBMClassifier initialized with parameters:")
        for key, value in self.model_params.items():
            self.logger.info(f" {key}: {value}")

    def train(self, X, y):
        """
        Train the LightGBM model

        Args:
            X: Training features
            y: Training labels
        """
        try:
            self.logger.info("Starting LightGBM model training...")
            self.lgbm_classifier.fit(X=X, y=y)
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
            predictions = self.lgbm_classifier.predict(X=X)
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
            probabilities = self.lgbm_classifier.predict_proba(X=X)
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
                pickle.dump(self.lgbm_classifier, f)
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
                self.lgbm_classifier = pickle.load(f)
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
            scores = cross_val_score(estimator=self.lgbm_classifier, X=X, y=y, cv=cv, scoring='accuracy')
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            self.logger.info("Cross-validation results:")
            self.logger.info(f"Mean accuracy: {mean_score:.4f} (+/- {std_score * 2:.4f})")
            return scores
        except Exception as e:
            self.logger.error(f"Error during cross-validation: {str(e)}")
            raise