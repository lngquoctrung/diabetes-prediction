import sys
from pathlib import Path
# Add the project path into the python path
root_dir = str(Path(__file__).parent.parent.absolute())
if not root_dir in sys.path:
    sys.path.insert(0, root_dir)

import pandas as pd
import os

from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, EditedNearestNeighbours
from imblearn.combine import SMOTETomek, SMOTEENN
from src.config import LOG_FORMAT, RANDOM_STATE
from src.utils import get_configured_logger


class OverSamplingBalancer:
    """Class for over-sampling techniques to balance imbalanced datasets"""

    def __init__(self, logger_name: str | None = __name__, log_file: str | None = None,
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
        self.logger = get_configured_logger(
            name=logger_name, 
            log_file=log_file, 
            log_format=log_format
        )

        self.logger.info(msg="OverSamplingBalancer initialized successfully")

    def apply_random_oversampling(self, X, y, sampling_strategy=None, random_state=RANDOM_STATE):
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
            random_state: int
                Random state

        Returns
        -------
            tuple: (X_resampled, y_resampled)
                Resampled features and labels
        """
        self.logger.info(msg="Starting Random Over Sampling process...")

        # Log original distribution
        original_distribution = pd.Series(data=y).value_counts().sort_index()
        self.logger.info(msg="Original class distribution:")
        for class_label, count in original_distribution.items():
            self.logger.info(msg=f"  - Class {class_label}: {count} samples")

        try:
            # Initialize Random Over Sampler
            ros = RandomOverSampler(
                sampling_strategy=sampling_strategy,
                random_state=random_state
            )

            # Apply resampling
            X_resampled, y_resampled = ros.fit_resample(X=X, y=y)

            # Log new distribution
            new_distribution = pd.Series(data=y_resampled).value_counts().sort_index()
            self.logger.info(msg="New class distribution after Random Over Sampling:")
            for class_label, count in new_distribution.items():
                original_count = original_distribution.get(class_label, 0)
                added_samples = count - original_count
                self.logger.info(msg=f"  - Class {class_label}: {count} samples (+{added_samples})")

            total_original = len(y)
            total_new = len(y_resampled)
            self.logger.info(msg=f"Dataset size: {total_original} -> {total_new} samples (+{total_new - total_original})")
            self.logger.info(msg="Random Over Sampling completed successfully")

            return X_resampled, y_resampled

        except Exception as e:
            self.logger.error(msg=f"Error during Random Over Sampling: {str(e)}")
            raise

    def apply_smote(self, X, y, sampling_strategy="auto", k_neighbors=5, random_state=RANDOM_STATE):
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
            random_state: int
                Random state

        Returns
        -------
            tuple: (X_resampled, y_resampled)
                Resampled features and labels
        """
        self.logger.info(msg="Starting SMOTE process...")

        # Log original distribution
        original_distribution = pd.Series(data=y).value_counts().sort_index()
        self.logger.info("Original class distribution:")
        for class_label, count in original_distribution.items():
            self.logger.info(f"  - Class {class_label}: {count} samples")

        try:
            # Initialize SMOTE
            smote = SMOTE(
                sampling_strategy=sampling_strategy,
                k_neighbors=k_neighbors,
                random_state=random_state,
            )

            # Apply resampling
            X_resampled, y_resampled = smote.fit_resample(X=X, y=y)

            # Log new distribution
            new_distribution = pd.Series(data=y_resampled).value_counts().sort_index()
            self.logger.info(msg="New class distribution after SMOTE:")
            for class_label, count in new_distribution.items():
                original_count = original_distribution.get(class_label, 0)
                added_samples = count - original_count
                self.logger.info(msg=f"  - Class {class_label}: {count} samples (+{added_samples} synthetic)")

            total_original = len(y)
            total_new = len(y_resampled)
            self.logger.info(
                msg=f"Dataset size: {total_original} -> {total_new} samples (+{total_new - total_original} synthetic)"
            )
            self.logger.info(msg="SMOTE completed successfully")

            return X_resampled, y_resampled

        except Exception as e:
            self.logger.error(msg=f"Error during SMOTE: {str(e)}")
            raise

    def apply_adasyn(self, X, y, sampling_strategy="auto", n_neighbors=5, random_state=RANDOM_STATE):
        """
        Apply ADASYN over-sampling to balance the dataset

        Parameters
        ----------
        X: array-like (training features)
        y: array-like (training labels)
        sampling_strategy: dict, str, or None (sampling strategy)
        n_neighbors: number of nearest neighbors
        random_state: int

        Returns
        -------
        tuple: (X_resampled, y_resampled)
        """
        self.logger.info("Starting ADASYN over-sampling process...")
        original_distribution = pd.Series(y).value_counts().sort_index()
        self.logger.info("Original class distribution:")
        for cls, count in original_distribution.items():
            self.logger.info(f" - Class {cls}: {count} samples")
        try:
            adasyn = ADASYN(sampling_strategy=sampling_strategy, n_neighbors=n_neighbors, random_state=random_state)
            X_res, y_res = adasyn.fit_resample(X, y)
            new_distribution = pd.Series(y_res).value_counts().sort_index()
            self.logger.info("New class distribution after ADASYN:")
            for cls, count in new_distribution.items():
                original_count = original_distribution.get(cls, 0)
                added = count - original_count
                self.logger.info(f" - Class {cls}: {count} samples (+{added} synthetic)")
            self.logger.info(f"Dataset size: {len(y)} -> {len(y_res)} samples (+{len(y_res)-len(y)} synthetic)")
            self.logger.info("ADASYN over-sampling completed successfully")
            return X_res, y_res
        except Exception as e:
            self.logger.error(f"Error during ADASYN over-sampling: {str(e)}")
            raise

class UnderSamplingBalancer:
    """Class for under-sampling techniques to balance imbalanced datasets"""

    def __init__(self, logger_name: str | None = __name__, log_file: str | None = None,
                 log_format: str | None = LOG_FORMAT):
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
        self.logger = get_configured_logger(
            name=logger_name, 
            log_file=log_file, 
            log_format=log_format
        )

        self.logger.info(msg="UnderSamplingBalancer initialized successfully")

    def apply_random_undersampling(self, X, y, sampling_strategy="auto", random_state=RANDOM_STATE):
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
            random_state: int
                Random state

        Returns
        -------
            tuple: (X_resampled, y_resampled)
                Resampled features and labels
        """
        self.logger.info(msg="Starting Random Under Sampling process...")

        # Log original distribution
        original_distribution = pd.Series(data=y).value_counts().sort_index()
        self.logger.info(msg="Original class distribution:")
        for class_label, count in original_distribution.items():
            self.logger.info(msg=f"  - Class {class_label}: {count} samples")

        try:
            # Initialize Random Under Sampler
            rus = RandomUnderSampler(
                sampling_strategy=sampling_strategy,
                random_state=random_state,
                replacement=False  # Do not duplicate sample
            )

            # Apply resampling
            X_resampled, y_resampled = rus.fit_resample(X=X, y=y)

            # Log new distribution
            new_distribution = pd.Series(data=y_resampled).value_counts().sort_index()
            self.logger.info(msg="New class distribution after Random Under Sampling:")
            for class_label, count in new_distribution.items():
                original_count = original_distribution.get(class_label, 0)
                removed_samples = original_count - count
                self.logger.info(msg=f"  - Class {class_label}: {count} samples (-{removed_samples})")

            total_original = len(y)
            total_new = len(y_resampled)
            self.logger.info(msg=f"Dataset size: {total_original} -> {total_new} samples (-{total_original - total_new})")
            self.logger.info(msg="Random Under Sampling completed successfully")

            return X_resampled, y_resampled

        except Exception as e:
            self.logger.error(msg=f"Error during Random Under Sampling: {str(e)}")
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
        self.logger.info(msg="Starting Tomek Links process...")

        # Log original distribution
        original_distribution = pd.Series(data=y).value_counts().sort_index()
        self.logger.info(msg="Original class distribution:")
        for class_label, count in original_distribution.items():
            self.logger.info(msg=f"  - Class {class_label}: {count} samples")

        try:
            # Initialize Tomek Links
            tomek = TomekLinks(
                sampling_strategy=sampling_strategy if sampling_strategy else 'auto',
                n_jobs=n_jobs
            )

            # Apply resampling
            X_resampled, y_resampled = tomek.fit_resample(X=X, y=y)

            # Log new distribution
            new_distribution = pd.Series(data=y_resampled).value_counts().sort_index()
            self.logger.info(msg="New class distribution after Tomek Links:")
            for class_label, count in new_distribution.items():
                original_count = original_distribution.get(class_label, 0)
                removed_samples = original_count - count
                if removed_samples > 0:
                    self.logger.info(msg=f"  - Class {class_label}: {count} samples (-{removed_samples} Tomek links)")
                else:
                    self.logger.info(msg=f"  - Class {class_label}: {count} samples (unchanged)")

            total_original = len(y)
            total_new = len(y_resampled)
            total_removed = total_original - total_new
            self.logger.info(msg=f"Dataset size: {total_original} -> {total_new} samples (-{total_removed} Tomek links)")
            self.logger.info(msg="Tomek Links processing completed successfully")

            return X_resampled, y_resampled

        except Exception as e:
            self.logger.error(msg=f"Error during Tomek Links processing: {str(e)}")
            raise

    def apply_edited_nearest_neighbours(self, X, y, sampling_strategy="auto", n_neighbors=3, n_jobs=1):
        """
        Apply Edited Nearest Neighbours (ENN) under-sampling process

        Parameters
        ----------
        X: array-like (training features)
        y: array-like (training labels)
        sampling_strategy: str or list (sampling strategy)
        n_neighbors: int (number of neighbors)
        n_jobs: int (parallel jobs for ENN)

        Returns
        -------
        tuple: (X_resampled, y_resampled)
        """
        self.logger.info("Starting Edited Nearest Neighbours (ENN) under-sampling process...")
        original_distribution = pd.Series(y).value_counts().sort_index()
        self.logger.info("Original class distribution:")
        for cls, count in original_distribution.items():
            self.logger.info(f" - Class {cls}: {count} samples")
        try:
            enn = EditedNearestNeighbours(sampling_strategy=sampling_strategy, n_neighbors=n_neighbors, n_jobs=n_jobs)
            X_res, y_res = enn.fit_resample(X, y)
            new_distribution = pd.Series(y_res).value_counts().sort_index()
            self.logger.info("New class distribution after ENN:")
            for cls, count in new_distribution.items():
                removed = original_distribution.get(cls, 0) - count
                self.logger.info(f" - Class {cls}: {count} samples (-{removed} removed)")
            self.logger.info(f"Dataset size: {len(y)} -> {len(y_res)} samples (-{len(y) - len(y_res)})")
            self.logger.info("ENN under-sampling completed successfully")
            return X_res, y_res
        except Exception as e:
            self.logger.error(f"Error during ENN under-sampling: {str(e)}")
            raise

class HybridSamplingBalancer:
    """Class for hybrid sampling techniques (SMOTE and Under-sampling) to balance imbalanced datasets"""

    def __init__(self, logger_name: str | None = __name__, log_file: str | None = None,
                 log_format: str | None = LOG_FORMAT):
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
        self.logger = get_configured_logger(
            name=logger_name, 
            log_file=log_file, 
            log_format=log_format
        )

        self.logger.info(msg="HybridSamplingBalancer initialized successfully")

    def apply_smote_tomek(self, X, y, sampling_strategy="auto", random_state=RANDOM_STATE, n_jobs=1):
        """
        Apply SMOTETomek hybrid sampling to balance the dataset

        SMOTETomek combines SMOTE over-sampling with Tomek links under-sampling:
        1. First, applies SMOTE to over-sample minority classes
        2. Then apply Tomek links to remove noisy and borderline samples

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
            random_state: int
                Random state
            n_jobs: int
                Number of CPU cores to use for parallel processing

        Returns
        -------
            tuple: (X_resampled, y_resampled)
                Resampled features and labels
        """
        self.logger.info(msg="Starting SMOTETomek hybrid sampling process...")

        # Log original distribution
        original_distribution = pd.Series(data=y).value_counts().sort_index()
        self.logger.info(msg="Original class distribution:")
        for class_label, count in original_distribution.items():
            self.logger.info(msg=f"  - Class {class_label}: {count} samples")

        try:
            # Initialize SMOTETomek
            smote_tomek = SMOTETomek(
                sampling_strategy=sampling_strategy,
                smote=None,  # Use default SMOTE
                tomek=None,  # Use default TomekLinks
                n_jobs=n_jobs,
                random_state=random_state
            )

            # Apply hybrid resampling
            X_resampled, y_resampled = smote_tomek.fit_resample(X=X, y=y)

            # Log new distribution
            new_distribution = pd.Series(data=y_resampled).value_counts().sort_index()
            self.logger.info(msg="New class distribution after SMOTETomek:")
            for class_label, count in new_distribution.items():
                original_count = original_distribution.get(class_label, 0)
                net_change = count - original_count
                if net_change > 0:
                    self.logger.info(msg=f"  - Class {class_label}: {count} samples (+{net_change} net)")
                elif net_change < 0:
                    self.logger.info(msg=f"  - Class {class_label}: {count} samples ({net_change} net)")
                else:
                    self.logger.info(msg=f"  - Class {class_label}: {count} samples (unchanged)")

            total_original = len(y)
            total_new = len(y_resampled)
            net_change = total_new - total_original
            if net_change > 0:
                self.logger.info(msg=f"Dataset size: {total_original} -> {total_new} samples (+{net_change})")
            elif net_change < 0:
                self.logger.info(msg=f"Dataset size: {total_original} -> {total_new} samples ({net_change})")
            else:
                self.logger.info(msg=f"Dataset size: {total_original} -> {total_new} samples (unchanged)")

            self.logger.info(msg="SMOTETomek processing completed successfully")

            return X_resampled, y_resampled

        except Exception as e:
            self.logger.error(msg=f"Error during SMOTETomek processing: {str(e)}")
            raise

    def apply_smote_enn(self, X, y, sampling_strategy="auto", random_state=RANDOM_STATE, n_jobs=1):
        """
        Apply SMOTEENN hybrid sampling to balance the dataset

        SMOTEENN combines SMOTE over-sampling with Edited Nearest Neighbors under-sampling:
        1. First, applies SMOTE to over-sample minority classes
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
            random_state: int
                Random state
            n_jobs: int
                Number of CPU cores to use for parallel processing

        Returns
        -------
            tuple: (X_resampled, y_resampled)
                Resampled features and labels
        """
        self.logger.info(msg="Starting SMOTEENN hybrid sampling process...")

        # Log original distribution
        original_distribution = pd.Series(data=y).value_counts().sort_index()
        self.logger.info(msg="Original class distribution:")
        for class_label, count in original_distribution.items():
            self.logger.info(msg=f"  - Class {class_label}: {count} samples")

        try:
            # Initialize SMOTEENN
            smote_enn = SMOTEENN(
                sampling_strategy=sampling_strategy,
                smote=None,  # Use default SMOTE
                enn=None,  # Use default EditedNearestNeighbours
                n_jobs=n_jobs,
                random_state=random_state
            )

            # Apply hybrid resampling
            X_resampled, y_resampled = smote_enn.fit_resample(X=X, y=y)

            # Log new distribution
            new_distribution = pd.Series(data=y_resampled).value_counts().sort_index()
            self.logger.info(msg="New class distribution after SMOTEENN:")
            for class_label, count in new_distribution.items():
                original_count = original_distribution.get(class_label, 0)
                net_change = count - original_count
                if net_change > 0:
                    self.logger.info(msg=f"  - Class {class_label}: {count} samples (+{net_change} net)")
                elif net_change < 0:
                    self.logger.info(msg=f"  - Class {class_label}: {count} samples ({net_change} net)")
                else:
                    self.logger.info(msg=f"  - Class {class_label}: {count} samples (unchanged)")

            total_original = len(y)
            total_new = len(y_resampled)
            net_change = total_new - total_original
            if net_change > 0:
                self.logger.info(msg=f"Dataset size: {total_original} -> {total_new} samples (+{net_change})")
            elif net_change < 0:
                self.logger.info(msg=f"Dataset size: {total_original} -> {total_new} samples ({net_change})")
            else:
                self.logger.info(msg=f"Dataset size: {total_original} -> {total_new} samples (unchanged)")

            self.logger.info(msg="SMOTEENN processing completed successfully")

            return X_resampled, y_resampled

        except Exception as e:
            self.logger.error(msg=f"Error during SMOTEENN processing: {str(e)}")
            raise

    def apply_adasyn_tomek(self, X, y, sampling_strategy="auto", random_state=RANDOM_STATE, n_jobs=1):
        """
        Apply ADASYN + TomekLinks hybrid sampling to balance the dataset

        This method performs:
        1. ADASYN to over-sample minority classes adaptively.
        2. TomekLinks to remove overlapping borderline samples.

        Parameters
        ----------
            X: array-like
                Training features
            y: array-like
                Training labels
            sampling_strategy: dict, str, or None
                Sampling strategy for ADASYN over-sampling
                - dict: {class_label: desired_count}
                - 'auto': balance all classes to majority class
                - None: equivalent to 'auto'
            random_state: int
                Random state
            n_jobs: int
                Number of CPU cores to use for parallel processing

        Returns
        -------
            tuple: (X_resampled, y_resampled)
                Resampled features and labels
        """
        self.logger.info(msg="Starting ADASYN + TomekLinks hybrid sampling process...")

        # Log original distribution
        original_distribution = pd.Series(data=y).value_counts().sort_index()
        self.logger.info(msg="Original class distribution:")
        for class_label, count in original_distribution.items():
            self.logger.info(msg=f"  - Class {class_label}: {count} samples")

        try:
            # Step 1: Apply ADASYN
            adasyn = ADASYN(sampling_strategy=sampling_strategy, random_state=random_state)
            X_adasyn, y_adasyn = adasyn.fit_resample(X, y)

            # Step 2: Apply TomekLinks (default removes from majority class)
            tomek = TomekLinks(n_jobs=n_jobs)
            X_resampled, y_resampled = tomek.fit_resample(X_adasyn, y_adasyn)

            # Log new distribution
            new_distribution = pd.Series(data=y_resampled).value_counts().sort_index()
            self.logger.info(msg="New class distribution after ADASYN + TomekLinks:")
            for class_label, count in new_distribution.items():
                original_count = original_distribution.get(class_label, 0)
                net_change = count - original_count
                if net_change > 0:
                    self.logger.info(msg=f"  - Class {class_label}: {count} samples (+{net_change} net)")
                elif net_change < 0:
                    self.logger.info(msg=f"  - Class {class_label}: {count} samples ({net_change} net)")
                else:
                    self.logger.info(msg=f"  - Class {class_label}: {count} samples (unchanged)")

            total_original = len(y)
            total_new = len(y_resampled)
            net_change = total_new - total_original
            if net_change > 0:
                self.logger.info(msg=f"Dataset size: {total_original} -> {total_new} samples (+{net_change})")
            elif net_change < 0:
                self.logger.info(msg=f"Dataset size: {total_original} -> {total_new} samples ({net_change})")
            else:
                self.logger.info(msg=f"Dataset size: {total_original} -> {total_new} samples (unchanged)")

            self.logger.info(msg="ADASYN + TomekLinks processing completed successfully")

            return X_resampled, y_resampled

        except Exception as e:
            self.logger.error(msg=f"Error during ADASYN + TomekLinks processing: {str(e)}")
            raise
