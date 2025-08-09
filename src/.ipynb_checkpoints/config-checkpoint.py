from pathlib import Path

# ===== Data =====
DATA_DIR = Path(__file__).parent.parent.absolute() / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

BRLSS_17_URL = "https://www.cdc.gov/brfss/annual_data/2017/files/LLCP2017XPT.zip"
BRLSS_19_URL = "https://www.cdc.gov/brfss/annual_data/2019/files/LLCP2019XPT.zip"
BRLSS_21_URL = "https://www.cdc.gov/brfss/annual_data/2021/files/LLCP2021XPT.zip"

BRFSS_17_FILENAME = "brfss_2017.XPT"
BRFSS_19_FILENAME = "brfss_2019.XPT"
BRFSS_21_FILENAME = "brfss_2021.XPT"

BRFSS_17_DATA_PATH = RAW_DATA_DIR / BRFSS_17_FILENAME
BRFSS_19_DATA_PATH = RAW_DATA_DIR / BRFSS_19_FILENAME
BRFSS_21_DATA_PATH = RAW_DATA_DIR / BRFSS_21_FILENAME

BRFSS_FILTERING_FILE_PATH = PROCESSED_DATA_DIR / "brfss_filtering.csv"

# ===== Logging =====
LOG_FORMAT = "%(asctime)s - [%(name)s] - %(levelname)s - %(message)s"

# ===== Train and test data
TRAIN_SIZE = 0.8

# ===== Balancing =====
BALANCING_STRATEGY = {
    0: 574477,
    1: 150000,
    2: 100000,
}

OVER_SAMPLING_STRATEGY = {
    1: 50000,
    2: 100000,
}

UNDER_SAMPLING_STRATEGY = {
    0: 300000,
    1: 13570,
    2: 50000
}

# ===== Model =====
RANDOM_STATE = 42
N_JOBS = -1
# SGD Classifier
SGD_LOSS = "log_loss"
SGD_PENALTY = "l2"
SGD_ALPHA = 0.0001
SGD_MAX_ITER = 1000
SGD_LEARNING_RATE = "constant"

# Logistic regression
LR_C = 1.0
LR_PENALTY = "l2"
LR_CLASS_WEIGHT = None
LR_SOLVER = "saga"
LR_MAX_ITER = 100
LR_VERBOSE = 0

# Random Forest
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 25

# Naive Bayes
NB_TYPE = "gaussian"
NB_VAR_SMOOTHING = 1e-9
NB_ALPHA = 1.0
NB_BINARIZE = 0.0