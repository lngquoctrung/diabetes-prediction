from pathlib import Path

# ===== Data =====
DATA_DIR = Path(__file__).parent.parent.absolute() / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

BRFSS_20_FILENAME = "brfss_2020.XPT"
BRFSS_21_FILENAME = "brfss_2021.XPT"
BRFSS_22_FILENAME = "brfss_2022.XPT"

BRFSS_20_DATA_PATH = RAW_DATA_DIR / BRFSS_20_FILENAME
BRFSS_21_DATA_PATH = RAW_DATA_DIR / BRFSS_21_FILENAME
BRFSS_22_DATA_PATH = RAW_DATA_DIR / BRFSS_22_FILENAME

BRFSS_DATA_FILENAME = RAW_DATA_DIR / "BRFSS_20_21_22.csv"

# ===== Logging =====
LOG_FORMAT = "%(asctime)s - [%(name)s] - %(levelname)s - %(message)s"