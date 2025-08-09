import sys
from pathlib import Path
# Add the project path into the python path
root_dir = str(Path(__file__).parent.parent.parent.absolute())
if not root_dir in sys.path:
    sys.path.insert(0, root_dir)

import pandas as pd
import logging
import os

from imblearn.over_sampling import SMOTE, RandomOverSampler
from src.config import LOG_FORMAT
from src.utils import make_dirs


class OverSamplingBalancer:
    """Class for over-sampling techniques to balance imbalanced datasets"""

    def __init__(self, log_file: str | None = None,
                 log_format: str | None = LOG_FORMAT):
        """
        Constructor of OverSamplingBalancer class

        Parameters
        ----------
            log_file: str or None
                The log filename
            log_format: str
                The log format
        """
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

        self.logger.info("OverSamplingBalancer initialized successfully")

    def apply_random_oversampling(self, X, y, sampling_strategy=None):
        """
        Apply Random Over Sampling to balance the dataset

        Parameters
        ----------
            X: array-like
                Training features
            y: array-like
                Training labels
            sampling_strategy: dict, str, or None
                Sampling strategy for each class
                - dict: {class_label: desired_count}
                - 'auto': balance all classes to majority class
                - 'minority': balance minority classes to majority class
                - None: equivalent to 'auto'

        Returns
        -------
            tuple: (X_resampled, y_resampled)
                Resampled features and labels
        """
        self.logger.info("Starting Random Over Sampling process...")

        # Log original distribution
        original_distribution = pd.Series(y).value_counts().sort_index()
        self.logger.info("Original class distribution:")
        for class_label, count in original_distribution.items():
            self.logger.info(f"  - Class {class_label}: {count} samples")

        try:
            # Initialize Random Over Sampler
            ros = RandomOverSampler(
                sampling_strategy=sampling_strategy,
                random_state=42
            )

            # Apply resampling
            X_resampled, y_resampled = ros.fit_resample(X, y)

            # Log new distribution
            new_distribution = pd.Series(y_resampled).value_counts().sort_index()
            self.logger.info("New class distribution after Random Over Sampling:")
            for class_label, count in new_distribution.items():
                original_count = original_distribution.get(class_label, 0)
                added_samples = count - original_count
                self.logger.info(f"  - Class {class_label}: {count} samples (+{added_samples})")

            total_original = len(y)
            total_new = len(y_resampled)
            self.logger.info(f"Dataset size: {total_original} -> {total_new} samples (+{total_new - total_original})")
            self.logger.info("Random Over Sampling completed successfully")

            return X_resampled, y_resampled

        except Exception as e:
            self.logger.error(f"Error during Random Over Sampling: {str(e)}")
            raise

    def apply_smote(self, X, y, sampling_strategy="auto", k_neighbors=5):
        """
        Apply SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset

        Parameters
        ----------
            X: array-like
                Training features
            y: array-like
                Training labels
            sampling_strategy: dict, str, or None
                Sampling strategy for each class
                - dict: {class_label: desired_count}
                - 'auto': balance all classes to majority class
                - 'minority': balance minority classes to majority class
                - None: equivalent to 'auto'
            k_neighbors: int
                Number of nearest neighbors for SMOTE algorithm

        Returns
        -------
            tuple: (X_resampled, y_resampled)
                Resampled features and labels
        """
        self.logger.info("Starting SMOTE process...")

        # Log original distribution
        original_distribution = pd.Series(y).value_counts().sort_index()
        self.logger.info("Original class distribution:")
        for class_label, count in original_distribution.items():
            self.logger.info(f"  - Class {class_label}: {count} samples")

        try:
            # Initialize SMOTE
            smote = SMOTE(
                sampling_strategy=sampling_strategy,
                k_neighbors=k_neighbors,
                random_state=42,
            )

            # Apply resampling
            X_resampled, y_resampled = smote.fit_resample(X, y)

            # Log new distribution
            new_distribution = pd.Series(y_resampled).value_counts().sort_index()
            self.logger.info("New class distribution after SMOTE:")
            for class_label, count in new_distribution.items():
                original_count = original_distribution.get(class_label, 0)
                added_samples = count - original_count
                self.logger.info(f"  - Class {class_label}: {count} samples (+{added_samples} synthetic)")

            total_original = len(y)
            total_new = len(y_resampled)
            self.logger.info(
                f"Dataset size: {total_original} -> {total_new} samples (+{total_new - total_original} synthetic)")
            self.logger.info("SMOTE completed successfully")

            return X_resampled, y_resampled

        except Exception as e:
            self.logger.error(f"Error during SMOTE: {str(e)}")
            raise