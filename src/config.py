from pathlib import Path

# ===== Data =====
DATA_DIR = Path(__file__).parent.parent.absolute() / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

BRFSS_17_URL = "https://www.cdc.gov/brfss/annual_data/2017/files/LLCP2017XPT.zip"
BRFSS_19_URL = "https://www.cdc.gov/brfss/annual_data/2019/files/LLCP2019XPT.zip"
BRFSS_21_URL = "https://www.cdc.gov/brfss/annual_data/2021/files/LLCP2021XPT.zip"

BRFSS_17_FILENAME = "brfss_2017.XPT"
BRFSS_19_FILENAME = "brfss_2019.XPT"
BRFSS_21_FILENAME = "brfss_2021.XPT"

BRFSS_17_DATA_PATH = str(RAW_DATA_DIR / BRFSS_17_FILENAME)
BRFSS_19_DATA_PATH = str(RAW_DATA_DIR / BRFSS_19_FILENAME)
BRFSS_21_DATA_PATH = str(RAW_DATA_DIR / BRFSS_21_FILENAME)

BRFSS_CLEANED_FILE_PATH = str(PROCESSED_DATA_DIR / "brfss_dataset.csv")

# ===== Logging =====
LOG_DIR = str(Path(__file__).parent.parent.absolute() / "logs")
LOG_FORMAT = "%(asctime)s - [%(name)s] - %(levelname)s - %(message)s"

# ===== Train and test data
TRAIN_SIZE = 0.8

# ===== Balancing =====
OVER_SAMPLING_STRATEGY = {
    1.0: 50000,
    2.0: 100000,
}

UNDER_SAMPLING_STRATEGY = {
    0.0: 150000, 
    1.0: 9586,
    2.0: 66362,
}

TESTING_DATA_FILE_PATH = str(PROCESSED_DATA_DIR / "testing_data.pkl")
RANDOM_OVER_SAMPLING_DATA_FILE_PATH = str(PROCESSED_DATA_DIR / "random_over_sampling_data.pkl")
SMOTE_DATA_FILE_PATH = str(PROCESSED_DATA_DIR / "smote_data.pkl")
ADASYN_DATA_FILE_PATH = str(PROCESSED_DATA_DIR / "adasyn_data.pkl")
RANDOM_UNDER_SAMPLING_DATA_FILE_PATH = str(PROCESSED_DATA_DIR / "random_under_sampling_data.pkl")
TOMEK_LINKS_DATA_FILE_PATH = str(PROCESSED_DATA_DIR / "tomek_links_data.pkl")
EDITED_NEIGHBORS_DATA_FILE_PATH = str(PROCESSED_DATA_DIR / "edited_neighbors_data.pkl")
SMOTE_TOMEK_LINKS_DATA_FILE_PATH = str(PROCESSED_DATA_DIR / "smote_tomek_links_data.pkl")
SMOTE_ENN_DATA_FILE_PATH = str(PROCESSED_DATA_DIR / "smote_enn_data.pkl")
ADASYN_TOMEK_DATA_FILE_PATH = str(PROCESSED_DATA_DIR / "adasyn_tomek_data.pkl")


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
LR_MAX_ITER = 10000
LR_VERBOSE = 0

# Random Forest
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 25
RF_MIN_SAMPLES_SPLIT = 5
RF_MIN_SAMPLES_LEAF = 5
RF_CLASS_WEIGHT = "balanced"

# Naive Bayes
NB_TYPE = "gaussian"
NB_VAR_SMOOTHING = 1e-9
NB_ALPHA = 1.0
NB_BINARIZE = 0.0

# XGBoost
XGB_N_ESTIMATORS = 100
XGB_MAX_DEPTH = 6
XGB_LEARNING_RATE = 0.1
XGB_SUBSAMPLE = 1.0
XGB_COLSAMPLE_BY_TREE = 1.0

# LightGBM
LGBM_N_ESTIMATORS = 100
LGBM_MAX_DEPTH = -1
LGBM_LEARNING_RATE = 0.1
LGBM_NUM_LEAVES = 31
LGBM_SUBSAMPLE = 1.0
LGBM_COLSAMPLE_BY_TREE = 1.0

# ===== ARTIFACTS =====
ARTIFACT_DIR = Path(__file__).parent.parent.absolute() / "artifacts"
SCALERS_DIR = ARTIFACT_DIR / "scalers"
MODEL_DIR = ARTIFACT_DIR / "models"
TRAINING_DIR = ARTIFACT_DIR / "training_2017_2019_2021"

BEST_ACCURACY_MODEL_FILE_PATH = str(MODEL_DIR / "best_accuracy_model.pkl")
BEST_ACCURACY_MODEL_SCALER_FILE_PATH = str(SCALERS_DIR / "best_model_scaler.pkl")

BEST_F1_MODEL_FILE_PATH = str(MODEL_DIR / "best_f1_model.pkl")
BEST_F1_MODEL_SCALER_FILE_PATH = str(SCALERS_DIR / "best_f1_model_scaler.pkl")

BEST_PRECISION_MODEL_FILE_PATH = str(MODEL_DIR / "best_precision_model.pkl")
BEST_PRECISION_MODEL_SCALER_FILE_PATH = str(SCALERS_DIR / "best_precision_model_scaler.pkl")

BEST_RECALL_MODEL_FILE_PATH = str(MODEL_DIR / "best_recall_model.pkl")
BEST_RECALL_MODEL_SCALER_FILE_PATH = str(SCALERS_DIR / "best_recall_model_scaler.pkl")