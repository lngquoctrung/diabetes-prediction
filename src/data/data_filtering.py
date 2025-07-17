import pandas as pd
import logging

from src.config import LOG_FORMAT

class DataFiltering:
    def __init__(self, log_file: str = None, log_format: str | None = LOG_FORMAT):
        """"""
        self.feature_column_mapping = {
            "Diabetes": "DIABETE4",
            "HighBP": ("PDIABTST", "_RFHYPE5", "_RFHYPE6"),
            "HighChol": ("TOLDHI2", "TOLDHI3"),
            "CholCheck": ("_CHOLCH2", "_CHOLCH3"),
            "BMI": "_BMI5",
            "Smoker": "SMOKE100",
            "Stroke": "CVDSTRK3",
            "HeartDiseaseorAttack": "_MICHD",
            "PhysActivity": "_TOTINDA",
            "HvyAlcoholConsump": ("_RFDRHV7", "_RFDRHV8"),
            "AnyHealthcare": ("HLTHPLN1", "PRIMINSR"),
            "NoDocbcCost": ("MEDCOST", "MEDCOST1"),
            "GenHlth": "GENHLTH",
            "MentHlth": "MENTHLTH",
            "PhysHlth": "PHYSHLTH",
            "DiffWalk": "DIFFWALK",
            "Sex": "_SEX",
            "Age": "_AGEG5YR",
            "Education": "EDUCA",
            "Income": ("INCOME2", "INCOME3"),
        }

        # Log configuration
        log_formatter = logging.Formatter(log_format)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(log_formatter)
            self.logger.addHandler(file_handler)
        consolse_handler = logging.StreamHandler()
        consolse_handler.setFormatter(log_formatter)
        self.logger.addHandler(consolse_handler)

    def _process_features(self, df: pd.DataFrame):
        copy_df = df.copy()
        # Process Diabetes value values
        copy_df['Diabetes'] = copy_df['Diabetes'].replace({2: 0, 3: 0, 1: 2, 4: 1})
        copy_df = copy_df[copy_df['Diabetes'] != 7]
        copy_df = copy_df[copy_df['Diabetes'] != 9]

        # Process High-Blood Pressure values
        copy_df['HighBP'] = copy_df['HighBP'].replace({1: 0, 2: 1})
        copy_df = copy_df[copy_df['HighBP'] != 9]

        # Process High Cholesteron values
        copy_df['HighChol'] = copy_df['HighChol'].replace({2: 0})
        copy_df = copy_df[copy_df['HighChol'] != 7]
        copy_df = copy_df[copy_df['HighChol'] != 9]

        # Process BMI values
        copy_df['BMI'] = copy_df['BMI'].div(100).round(0)

        # Process Stroke values
        copy_df['Stroke'] = copy_df['Stroke'].replace({2: 0})
        copy_df = copy_df[copy_df['Stroke'] != 7]
        copy_df = copy_df[copy_df['Stroke'] != 9]

        copy_df['HeartDiseaseorAttack'] = copy_df['HeartDiseaseorAttack'].replace({2: 0})

        copy_df['PhysActivity'] = copy_df['PhysActivity'].replace({2: 0})

        copy_df = copy_df[copy_df['GenHlth'] != 7]
        copy_df = copy_df[copy_df['GenHlth'] != 9]

        copy_df['MentHlth'] = copy_df['MentHlth'].replace({88: 0})
        copy_df = copy_df[copy_df['MentHlth'] != 77]
        copy_df = copy_df[copy_df['MentHlth'] != 99]

        copy_df['PhysHlth'] = copy_df['PhysHlth'].replace({88: 0})
        copy_df = copy_df[copy_df['PhysHlth'] != 77]
        copy_df = copy_df[copy_df['PhysHlth'] != 99]

        copy_df['DiffWalk'] = copy_df['DiffWalk'].replace({2: 0})
        copy_df = copy_df[copy_df['DiffWalk'] != 7]
        copy_df = copy_df[copy_df['DiffWalk'] != 9]

        copy_df = copy_df[copy_df['Age'] != 14]

        copy_df = copy_df[copy_df['Income'] != 77]
        copy_df = copy_df[copy_df['Income'] != 99]
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
            brfss_df = pd.read_sas(file_path, encoding="utf-8")
        except Exception as e:
            self.logger.error(f"Error reading SAS file: {e}")
            return None

        selected_features = {}

        for feature_name, columns in self.feature_column_mapping.items():
            if isinstance(columns, str):
                columns = (columns,)  # convert to tuple for consistency

            selected_col = None
            for col in columns:
                if col in brfss_df.columns:
                    selected_col = col
                    break

            if selected_col:
                selected_features[feature_name] = brfss_df[selected_col]
                self.logger.debug(f"Feature '{feature_name}' mapped to column '{selected_col}'")
            else:
                self.logger.warning(f"No available columns found for feature '{feature_name}'")

        if not selected_features:
            self.logger.error("No features could be selected. Returning None.")
            return None

        df = pd.DataFrame(selected_features)

        selected_df = self._process_features(df)
        if dropna:
            selected_df.dropna(inplace=True)

        return selected_df