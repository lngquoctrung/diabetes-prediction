import sys
from pathlib import Path

root_path = str(Path(__file__).parent.parent.absolute())
if not root_path in sys.path:
    sys.path.insert(0, root_path)

import os
import shutil
import pandas as pd

from src.data.data_loader import BrfssDataLoader
from src.config import (
    BRFSS_17_FILENAME,
    BRFSS_19_FILENAME,
    BRFSS_21_FILENAME,
    BRLSS_17_URL,
    BRLSS_19_URL,
    BRLSS_21_URL,
    BRFSS_17_DATA_PATH,
    BRFSS_19_DATA_PATH,
    BRFSS_21_DATA_PATH,
    PROCESSED_DATA_DIR,
    BRFSS_FILTERING_FILE_PATH,
)
from src.data.data_filtering import DataFiltering
from src.utils import make_dirs

def test_data_filtering():
    log_file = "./logs/test_data_filtering.log"
    data_filtering = DataFiltering(log_file=log_file)

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

        # Load and filter data
        filtered_df = data_filtering.select_features(file_path, dropna=True)

        if filtered_df is None:
            continue

        filtered_df['Year'] = year
        datasets[year] = filtered_df

    # Combine all datasets
    valid_dfs = [df for df in datasets.values() if not df.empty and not df.isna().all().all()]
    final_processed_df = pd.concat(valid_dfs, ignore_index=True)

    # Save processed data
    final_processed_df.to_csv(BRFSS_FILTERING_FILE_PATH, index=False)

    assert os.path.exists(BRFSS_FILTERING_FILE_PATH) == True

    # Remove the data directory after testing
    shutil.rmtree(os.path.dirname(BRFSS_17_DATA_PATH))
    shutil.rmtree(os.path.dirname(BRFSS_FILTERING_FILE_PATH))