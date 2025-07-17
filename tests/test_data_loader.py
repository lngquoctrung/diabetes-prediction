import sys
from pathlib import Path

root_path = str(Path(__file__).parent.parent.absolute())
if not root_path in sys.path:
    sys.path.insert(0, root_path)

from src.data.data_loader import BrfssDataLoader
from src.config import BRFSS_20_FILENAME, BRFSS_21_FILENAME, BRFSS_22_FILENAME

if __name__ == "__main__":
    data_loader = BrfssDataLoader(
        log_file="./logs/data_loader.log"
    )
    data_loader.load_data(
        urls=[
            "https://www.cdc.gov/brfss/annual_data/2020/files/LLCP2020XPT.zip",
            "https://www.cdc.gov/brfss/annual_data/2021/files/LLCP2021XPT.zip",
            "https://www.cdc.gov/brfss/annual_data/2022/files/LLCP2022XPT.zip",
        ],
        filenames=[
            BRFSS_20_FILENAME,
            BRFSS_21_FILENAME,
            BRFSS_22_FILENAME
        ])