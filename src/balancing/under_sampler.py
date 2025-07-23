import pandas as pd
import numpy as np
import logging
import os
import sys
from pathlib import Path
from imblearn.under_sampling import RandomUnderSampler, TomekLinks

# Add the project path into the python path
root_dir = str(Path(__file__).parent.parent.parent.absolute())
if not root_dir in sys.path:
    sys.path.insert(0, root_dir)

from src.config import LOG_FORMAT
from src.utils import make_dirs


class UnderSamplingBalancer:
    """Class for under-sampling techniques to balance imbalanced datasets"""

    def __init__(self, log_file: str | None = None, log_format: str | None = LOG_FORMAT):
        """
        Constructor of UnderSamplingBalancer class

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

        self.logger.info("UnderSamplingBalancer initialized successfully")

    def apply_random_undersampling(self, X, y, sampling_strategy="auto"):
        """
        Apply Random Under Sampling to balance the dataset by removing majority class samples

        Parameters
        ----------
            X: array-like
                Training features
            y: array-like
                Training labels
            sampling_strategy: dict, str, or None
                Sampling strategy for each class
                - dict: {class_label: desired_count}
                - 'auto': balance all classes to minority class
                - 'majority': undersample only majority class
                - 'not minority': undersample all classes except minority
                - None: equivalent to 'auto'

        Returns
        -------
            tuple: (X_resampled, y_resampled)
                Resampled features and labels
        """
        self.logger.info("Starting Random Under Sampling process...")

        # Log original distribution
        original_distribution = pd.Series(y).value_counts().sort_index()
        self.logger.info("Original class distribution:")
        for class_label, count in original_distribution.items():
            self.logger.info(f"  - Class {class_label}: {count} samples")

        try:
            # Initialize Random Under Sampler
            rus = RandomUnderSampler(
                sampling_strategy=sampling_strategy,
                random_state=42,
                replacement=False  # Không lặp lại mẫu

            )

            # Apply resampling
            X_resampled, y_resampled = rus.fit_resample(X, y)

            # Log new distribution
            new_distribution = pd.Series(y_resampled).value_counts().sort_index()
            self.logger.info("New class distribution after Random Under Sampling:")
            for class_label, count in new_distribution.items():
                original_count = original_distribution.get(class_label, 0)
                removed_samples = original_count - count
                self.logger.info(f"  - Class {class_label}: {count} samples (-{removed_samples})")

            total_original = len(y)
            total_new = len(y_resampled)
            self.logger.info(f"Dataset size: {total_original} -> {total_new} samples (-{total_original - total_new})")
            self.logger.info("Random Under Sampling completed successfully")

            return X_resampled, y_resampled

        except Exception as e:
            self.logger.error(f"Error during Random Under Sampling: {str(e)}")
            raise

    def apply_tomek_links(self, X, y, sampling_strategy="auto", n_jobs=1):
        """
        Apply Tomek Links to balance the dataset by removing noisy and borderline samples

        Parameters
        ----------
            X: array-like
                Training features
            y: array-like
                Training labels
            sampling_strategy: str or list
                Strategy for removing Tomek links:
                - 'auto' or 'not minority': remove majority class from Tomek links (default)
                - 'majority': remove only majority class samples
                - 'all': remove both samples in each Tomek link
                - list: specify classes to be affected
            n_jobs: int
                Number of CPU cores to use for parallel processing

        Returns
        -------
            tuple: (X_resampled, y_resampled)
                Resampled features and labels
        """
        self.logger.info("Starting Tomek Links process...")

        # Log original distribution
        original_distribution = pd.Series(y).value_counts().sort_index()
        self.logger.info("Original class distribution:")
        for class_label, count in original_distribution.items():
            self.logger.info(f"  - Class {class_label}: {count} samples")

        try:
            # Initialize Tomek Links
            tomek = TomekLinks(
                sampling_strategy=sampling_strategy if sampling_strategy else 'auto',
                n_jobs=n_jobs
            )

            # Apply resampling
            X_resampled, y_resampled = tomek.fit_resample(X, y)

            # Log new distribution
            new_distribution = pd.Series(y_resampled).value_counts().sort_index()
            self.logger.info("New class distribution after Tomek Links:")
            for class_label, count in new_distribution.items():
                original_count = original_distribution.get(class_label, 0)
                removed_samples = original_count - count
                if removed_samples > 0:
                    self.logger.info(f"  - Class {class_label}: {count} samples (-{removed_samples} Tomek links)")
                else:
                    self.logger.info(f"  - Class {class_label}: {count} samples (unchanged)")

            total_original = len(y)
            total_new = len(y_resampled)
            total_removed = total_original - total_new
            self.logger.info(f"Dataset size: {total_original} -> {total_new} samples (-{total_removed} Tomek links)")
            self.logger.info("Tomek Links processing completed successfully")

            return X_resampled, y_resampled

        except Exception as e:
            self.logger.error(f"Error during Tomek Links processing: {str(e)}")
            raise