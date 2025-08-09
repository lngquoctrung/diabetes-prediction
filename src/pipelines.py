# Add the project path into the python path
import sys
from pathlib import Path
root_dir = str(Path(__file__).parent.parent.absolute())
if not root_dir in sys.path:
    sys.path.insert(0, root_dir)

import pandas as pd
import logging
import os

from src.config import (
    LOG_FORMAT, BRFSS_17_DATA_PATH, BRFSS_19_DATA_PATH, BRFSS_21_DATA_PATH,
    BRLSS_17_URL, BRLSS_19_URL, BRLSS_21_URL,
    BRFSS_17_FILENAME, BRFSS_19_FILENAME, BRFSS_21_FILENAME,
    BRFSS_FILTERING_FILE_PATH, PROCESSED_DATA_DIR
)
from src.data import BrfssDataLoader, BrfssDataFiltering
from src.utils import make_dirs

class DataPipeline:
    """Complete data pipeline for BRFSS data processing"""

    def __init__(self, log_file: str = None, log_format: str | None = LOG_FORMAT):
        """Initialize DataPipeline with logging configuration"""
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

        self.logger.info("DataPipeline initialized successfully")

        # Initialize components
        self.brfss_data_filtering = BrfssDataFiltering(log_file=log_file)

    def run_pipeline(self):
        """Run the complete data pipeline"""
        self.logger.info("Starting data pipeline processing")

        # Create processed data directory
        make_dirs(PROCESSED_DATA_DIR)

        # Load data
        missing_data = {}
        for filename, items in {
            BRFSS_17_FILENAME: [BRFSS_17_DATA_PATH, BRLSS_17_URL],
            BRFSS_19_FILENAME: [BRFSS_19_DATA_PATH, BRLSS_19_URL],
            BRFSS_21_FILENAME: [BRFSS_21_DATA_PATH, BRLSS_21_URL]
        }.items():
            if not os.path.exists(items[0]):
                missing_data[filename] = items[1]
        brfss_data_loader = BrfssDataLoader()
        brfss_data_loader.load_data(
            urls=list(missing_data.values()),
            filenames=list(missing_data.keys())
        )

        # Step 1: Load raw data from all years
        datasets = {}
        file_paths = {
            2017: BRFSS_17_DATA_PATH,
            2019: BRFSS_19_DATA_PATH,
            2021: BRFSS_21_DATA_PATH
        }

        for year, file_path in file_paths.items():
            self.logger.info(f"Loading raw data for {year}")

            # Load and filter data
            filtered_df = self.brfss_data_filtering.select_features(file_path, dropna=True)

            if filtered_df is None:
                self.logger.error(f"Failed to filter data from {file_path}")
                continue

            filtered_df['Year'] = year
            datasets[year] = filtered_df

            self.logger.info(f"Loaded {year} raw data. Shape: {filtered_df.shape}")

        # Combine all datasets
        self.logger.info("Combining all datasets")
        valid_dfs = [df for df in datasets.values() if not df.empty and not df.isna().all().all()]
        final_processed_df = pd.concat(valid_dfs, ignore_index=True)

        # Save processed data
        self.logger.info(f"Saving processed data to {BRFSS_FILTERING_FILE_PATH}")
        final_processed_df.to_csv(BRFSS_FILTERING_FILE_PATH, index=False)

        self.logger.info(f"Data pipeline completed successfully. Final shape: {final_processed_df.shape}")

        # Log missing values summary
        missing_summary = final_processed_df.isnull().sum()
        self.logger.info("Missing values summary:")
        for col, count in missing_summary.items():
            if count > 0:
                self.logger.info(f"  {col}: {count} ({count / len(final_processed_df) * 100:.2f}%)")


if __name__ == "__main__":
    # Example usage
    pipeline = DataPipeline(log_file="./logs/data_pipeline.log")
    pipeline.run_pipeline()

    df = pd.read_csv(BRFSS_FILTERING_FILE_PATH)
    if df is not None:
        print(f"Pipeline completed successfully!")
        print(f"Final dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")

        # Show cholesterol columns summary
        print("\nCholesterol columns summary:")
        for col in ['HighChol', 'CholCheck']:
            if col in df.columns:
                print(f"{col}: {df[col].isnull().sum()} missing values")
                print(f"  Value counts: {df[col].value_counts().to_dict()}")
    else:
        print("Pipeline failed!")
