import pandas as pd
import numpy as np
import logging
import os
import sys
from pathlib import Path
from imblearn.combine import SMOTETomek, SMOTEENN

# Add the project path into the python path
root_dir = str(Path(__file__).parent.parent.parent.absolute())
if not root_dir in sys.path:
    sys.path.insert(0, root_dir)

from src.config import LOG_FORMAT
from src.utils import make_dirs


class HybridSamplingBalancer:
    """Class for hybrid sampling techniques (SMOTE + Under-sampling) to balance imbalanced datasets"""

    def __init__(self, log_file: str | None = None, log_format: str | None = LOG_FORMAT):
        """
        Constructor of HybridSamplingBalancer class

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

        self.logger.info("HybridSamplingBalancer initialized successfully")

    def apply_smote_tomek(self, X, y, sampling_strategy="auto", n_jobs=1):
        """
        Apply SMOTETomek hybrid sampling to balance the dataset

        SMOTETomek combines SMOTE over-sampling with Tomek links under-sampling:
        1. First applies SMOTE to over-sample minority classes
        2. Then applies Tomek links to remove noisy and borderline samples

        Parameters
        ----------
            X: array-like
                Training features
            y: array-like
                Training labels
            sampling_strategy: dict, str, or None
                Sampling strategy for SMOTE over-sampling
                - dict: {class_label: desired_count}
                - 'auto': balance all classes to majority class
                - None: equivalent to 'auto'
            n_jobs: int
                Number of CPU cores to use for parallel processing

        Returns
        -------
            tuple: (X_resampled, y_resampled)
                Resampled features and labels
        """
        self.logger.info("Starting SMOTETomek hybrid sampling process...")

        # Log original distribution
        original_distribution = pd.Series(y).value_counts().sort_index()
        self.logger.info("Original class distribution:")
        for class_label, count in original_distribution.items():
            self.logger.info(f"  - Class {class_label}: {count} samples")

        try:
            # Initialize SMOTETomek
            smote_tomek = SMOTETomek(
                sampling_strategy=sampling_strategy,
                smote=None,  # Use default SMOTE
                tomek=None,  # Use default TomekLinks
                n_jobs=n_jobs,
                random_state=42
            )

            # Apply hybrid resampling
            X_resampled, y_resampled = smote_tomek.fit_resample(X, y)

            # Log new distribution
            new_distribution = pd.Series(y_resampled).value_counts().sort_index()
            self.logger.info("New class distribution after SMOTETomek:")
            for class_label, count in new_distribution.items():
                original_count = original_distribution.get(class_label, 0)
                net_change = count - original_count
                if net_change > 0:
                    self.logger.info(f"  - Class {class_label}: {count} samples (+{net_change} net)")
                elif net_change < 0:
                    self.logger.info(f"  - Class {class_label}: {count} samples ({net_change} net)")
                else:
                    self.logger.info(f"  - Class {class_label}: {count} samples (unchanged)")

            total_original = len(y)
            total_new = len(y_resampled)
            net_change = total_new - total_original
            if net_change > 0:
                self.logger.info(f"Dataset size: {total_original} -> {total_new} samples (+{net_change})")
            elif net_change < 0:
                self.logger.info(f"Dataset size: {total_original} -> {total_new} samples ({net_change})")
            else:
                self.logger.info(f"Dataset size: {total_original} -> {total_new} samples (unchanged)")

            self.logger.info("SMOTETomek processing completed successfully")

            return X_resampled, y_resampled

        except Exception as e:
            self.logger.error(f"Error during SMOTETomek processing: {str(e)}")
            raise

    def apply_smote_enn(self, X, y, sampling_strategy="auto", n_jobs=1):
        """
        Apply SMOTEENN hybrid sampling to balance the dataset

        SMOTEENN combines SMOTE over-sampling with Edited Nearest Neighbors under-sampling:
        1. First applies SMOTE to over-sample minority classes
        2. Then applies ENN to remove samples that differ from their neighbors

        Parameters
        ----------
            X: array-like
                Training features
            y: array-like
                Training labels
            sampling_strategy: dict, str, or None
                Sampling strategy for SMOTE over-sampling
                - dict: {class_label: desired_count}
                - 'auto': balance all classes to majority class
                - None: equivalent to 'auto'
            n_jobs: int
                Number of CPU cores to use for parallel processing

        Returns
        -------
            tuple: (X_resampled, y_resampled)
                Resampled features and labels
        """
        self.logger.info("Starting SMOTEENN hybrid sampling process...")

        # Log original distribution
        original_distribution = pd.Series(y).value_counts().sort_index()
        self.logger.info("Original class distribution:")
        for class_label, count in original_distribution.items():
            self.logger.info(f"  - Class {class_label}: {count} samples")

        try:
            # Initialize SMOTEENN
            smote_enn = SMOTEENN(
                sampling_strategy=sampling_strategy,
                smote=None,  # Use default SMOTE
                enn=None,  # Use default EditedNearestNeighbours
                n_jobs=n_jobs,
                random_state=42
            )

            # Apply hybrid resampling
            X_resampled, y_resampled = smote_enn.fit_resample(X, y)

            # Log new distribution
            new_distribution = pd.Series(y_resampled).value_counts().sort_index()
            self.logger.info("New class distribution after SMOTEENN:")
            for class_label, count in new_distribution.items():
                original_count = original_distribution.get(class_label, 0)
                net_change = count - original_count
                if net_change > 0:
                    self.logger.info(f"  - Class {class_label}: {count} samples (+{net_change} net)")
                elif net_change < 0:
                    self.logger.info(f"  - Class {class_label}: {count} samples ({net_change} net)")
                else:
                    self.logger.info(f"  - Class {class_label}: {count} samples (unchanged)")

            total_original = len(y)
            total_new = len(y_resampled)
            net_change = total_new - total_original
            if net_change > 0:
                self.logger.info(f"Dataset size: {total_original} -> {total_new} samples (+{net_change})")
            elif net_change < 0:
                self.logger.info(f"Dataset size: {total_original} -> {total_new} samples ({net_change})")
            else:
                self.logger.info(f"Dataset size: {total_original} -> {total_new} samples (unchanged)")

            self.logger.info("SMOTEENN processing completed successfully")

            return X_resampled, y_resampled

        except Exception as e:
            self.logger.error(f"Error during SMOTEENN processing: {str(e)}")
            raise