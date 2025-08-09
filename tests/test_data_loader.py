import sys
from pathlib import Path
root_path = str(Path(__file__).parent.parent.absolute())
if not root_path in sys.path:
    sys.path.insert(0, root_path)

import os
import shutil

from src.data import BrfssDataLoader
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
)

def test_data_loader():
    # Downlaod dataset
    log_file = "./logs/test_data_loader.log"
    data_loader = BrfssDataLoader(
        log_file=log_file
    )
    data_loader.load_data(
        urls=[
            BRLSS_17_URL,
            BRLSS_19_URL,
            BRLSS_21_URL,
        ],
        filenames=[
            BRFSS_17_FILENAME,
            BRFSS_19_FILENAME,
            BRFSS_21_FILENAME
        ])

    # Check the files exist
    assert os.path.exists(BRFSS_17_DATA_PATH) == True
    assert os.path.exists(BRFSS_19_DATA_PATH) == True
    assert os.path.exists(BRFSS_21_DATA_PATH) == True

    # Remove the data directory after testing
    shutil.rmtree(os.path.dirname(BRFSS_17_DATA_PATH))