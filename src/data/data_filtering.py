import sys
from pathlib import Path

# Add the project path into the python path
root_dir = str(Path(__file__).parent.parent.parent.absolute())
if not root_dir in sys.path:
    sys.path.insert(0, root_dir)

import pandas as pd
import logging
import os

from src.config import LOG_FORMAT
from src.utils import make_dirs

class DataFiltering:
    def __init__(self, log_file: str = None, log_format: str | None = LOG_FORMAT):
        """
        Filter BRFSS dataset based on selected features.

        Parameters:
            log_file (str): Path to the log file
            log_format (str): Log format
        """
        self.feature_column_mapping = {
            "Diabetes": ("DIABETE3", "DIABETE4"),
            "HighBP": ("_RFHYPE5", "_RFHYPE6"),
            "HighChol": ("TOLDHI2", "TOLDHI3"),
            "CholCheck": ("_CHOLCH1", "_CHOLCH2", "_CHOLCH3"),
            "BMI": "_BMI5",
            "Smoker": "SMOKE100",
            "Stroke": "CVDSTRK3",
            "HeartDiseaseorAttack": "_MICHD",
            "PhysActivity": "_TOTINDA",
            "HvyAlcoholConsump": ("_RFDRHV5", "_RFDRHV7", "_RFDRHV8"),
            "AnyHealthcare": ("HLTHPLN1", "PRIMINSR"),
            "NoDocbcCost": ("MEDCOST", "MEDCOST1"),
            "GenHlth": "GENHLTH",
            "MentHlth": "MENTHLTH",
            "PhysHlth": "PHYSHLTH",
            "DiffWalk": "DIFFWALK",
            "Sex": ("SEX", "_SEX"),
            "Age": "_AGEG5YR",
            "Education": "EDUCA",
            "Income": ("INCOME2", "INCOME3"),
        }

        # Log configuration
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        log_formatter = logging.Formatter(log_format)
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

    def _process_features(self, df):
        """Process feature values for training data - chỉ process các cột khác, giữ nguyên cholcheck và highchol"""
        self.logger.info("Processing feature values for training")
        copy_df = df.copy()

        # Process Diabetes values
        copy_df['Diabetes'] = copy_df['Diabetes'].replace({2: 0, 3: 0, 1: 2, 4: 1})
        copy_df = copy_df[~copy_df['Diabetes'].isin([7, 9])]

        # Process High Blood Pressure values
        copy_df['HighBP'] = copy_df['HighBP'].replace({1: 0, 2: 1})
        copy_df = copy_df[~copy_df['HighBP'].isin([7, 9])]

        # Process High Cholesteron values
        copy_df['HighChol'] = copy_df['HighChol'].replace({2: 0})
        copy_df = copy_df[~copy_df['HighChol'].isin([7, 9])]

        # Process Cholesteron Check values
        copy_df['CholCheck'] = copy_df['CholCheck'].replace({3: 0, 2: 0})
        copy_df = copy_df[~copy_df['CholCheck'].isin([7, 9])]

        # Process BMI values
        copy_df['BMI'] = copy_df['BMI'].div(100).round(0)

        # Process Stroke values
        copy_df['Stroke'] = copy_df['Stroke'].replace({2: 0})
        copy_df = copy_df[~copy_df['Stroke'].isin([7, 9])]

        # Process Heart Disease or Attack values
        copy_df['HeartDiseaseorAttack'] = copy_df['HeartDiseaseorAttack'].replace({2: 0})

        # Process Physical Activity values
        copy_df['PhysActivity'] = copy_df['PhysActivity'].replace({2: 0})
        copy_df = copy_df[~copy_df['PhysActivity'].isin([9])]

        # Process General Health values
        copy_df = copy_df[~copy_df['GenHlth'].isin([7, 9])]

        # Process Mental Health values
        copy_df['MentHlth'] = copy_df['MentHlth'].replace({88: 0})
        copy_df = copy_df[~copy_df['MentHlth'].isin([77, 99])]

        # Process Physical Health values
        copy_df['PhysHlth'] = copy_df['PhysHlth'].replace({88: 0})
        copy_df = copy_df[~copy_df['PhysHlth'].isin([77, 99])]

        # Process Difficulty Walking values
        copy_df['DiffWalk'] = copy_df['DiffWalk'].replace({2: 0})
        copy_df = copy_df[~copy_df['DiffWalk'].isin([7, 9])]

        # Process Age values
        copy_df = copy_df[copy_df['Age'] != 14]

        # Process Income values
        copy_df = copy_df[~copy_df['Income'].isin([77, 99])]

        # Process Smoker values
        copy_df['Smoker'] = copy_df['Smoker'].replace({2: 0})
        copy_df = copy_df[~copy_df['Smoker'].isin([7, 9])]

        # Process Heavy Alcohol Consumption values
        copy_df['HvyAlcoholConsump'] = copy_df['HvyAlcoholConsump'].replace({1: 0 ,2: 1})
        copy_df = copy_df[~copy_df['HvyAlcoholConsump'].isin([9])]

        # Process Any Healthcare values
        copy_df['AnyHealthcare'] = copy_df['AnyHealthcare'].replace({2: 0})
        copy_df = copy_df[~copy_df['AnyHealthcare'].isin([7, 9])]

        # Process NoDocbcCost values
        copy_df['NoDocbcCost'] = copy_df['NoDocbcCost'].replace({2: 0})
        copy_df = copy_df[~copy_df['NoDocbcCost'].isin([7, 9])]

        # Process Sex values
        copy_df['Sex'] = copy_df['Sex'].replace({2: 0})  # 1: Male, 0: Female

        # Process Education values
        copy_df = copy_df[~copy_df['Education'].isin([9])]

        self.logger.info(f"Feature processing completed. Shape: {copy_df.shape}")
        return copy_df

    def select_features(self, file_path: str, dropna=False) -> pd.DataFrame | None:
        """
        Read data from .XPT file and select columns corresponding to defined features.

        Parameters:
            file_path (str): Path to .XPT data file
            dropna (bool): Whether to drop rows with missing values

        Returns:
            pd.DataFrame | None: DataFrame contains only the required features, or None if there is an error
        """
        self.logger.info(f"Selecting features from {file_path}")

        try:
            brfss_dataframe = pd.read_sas(file_path, encoding="utf-8")
        except Exception as e:
            self.logger.error(f"Error reading SAS file: {e}")
            return None

        selected_features = {}
        for feature_name, columns in self.feature_column_mapping.items():
            if isinstance(columns, str):
                columns = (columns,)  # convert to tuple for consistency

            selected_col = None
            for col in columns:
                if col in brfss_dataframe.columns:
                    selected_col = col
                    break

            if selected_col:
                selected_features[feature_name] = brfss_dataframe[selected_col]
                self.logger.debug(f"Feature '{feature_name}' mapped to column '{selected_col}'")
            else:
                self.logger.warning(f"No available columns found for feature '{feature_name}'")
                selected_features[feature_name] = None

        if not selected_features:
            self.logger.error("No features could be selected. Returning None.")
            return None

        dataframe = pd.DataFrame(selected_features)
        if dropna:
            dataframe.dropna(inplace=True)

        dataframe = self._process_features(dataframe)
        return dataframe