import sys
from pathlib import Path
# Add the project path into the python path
root_dir = str(Path(__file__).parent.parent.absolute())
if not root_dir in sys.path:
    sys.path.insert(0, root_dir)

import pandas as pd
import requests
import logging
import os
import zipfile
import shutil

from tqdm import tqdm
from typing import Optional
from src.config import RAW_DATA_DIR, LOG_FORMAT
from src.utils import make_dirs

class BrfssDataLoader:
    """Class for download, extract and load BRFSS dataset"""

    def __init__(self, chunk_size=1024, des_dir=RAW_DATA_DIR,
                 log_file: str | None = None, log_format: str | None = LOG_FORMAT):
        """
        Constructor of BrfssDataLoader class

        Parameters
        ----------
            chunk_size: int
                The size of each data block that program will handle at a time
            des_dir: str
                The destination folder path
            log_file: str or None
                The log filename
            log_format: str
                The log format
        """
        self.chunk_size = chunk_size
        self.des_dir = des_dir
        # Create a folder to store data
        make_dirs(path=os.path.dirname(des_dir))

        # Log configuration
        self.logger = logging.getLogger(name=__name__)
        self.logger.setLevel(level=logging.DEBUG)
        log_formatter = logging.Formatter(fmt=log_format)
        # Clear existing handlers to avoid duplicate logs
        self.logger.handlers.clear()
        # Handler log to file
        if log_file:
            # Create a log directory
            make_dirs(path=os.path.dirname(log_file))
            log_file_handler = logging.FileHandler(filename=log_file)
            log_file_handler.setLevel(level=logging.INFO)
            log_file_handler.setFormatter(fmt=log_formatter)
            self.logger.addHandler(hdlr=log_file_handler)
        # Handler log to console
        log_console_handler = logging.StreamHandler()
        log_console_handler.setLevel(level=logging.INFO)
        log_console_handler.setFormatter(fmt=log_formatter)
        self.logger.addHandler(hdlr=log_console_handler)

        self.logger.info("BrfssDataLoader initialized successfully")

    def _download_data(self, url, file_name, file_path):
        self.logger.info(f"Navigating to URL: {url}")
        try:
            response = requests.get(url=url, stream=True)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            self.logger.error(msg=f"Failed to download {file_name} from {url}. Error: {e}")
            raise

        # Get the size of a file
        total_size = int(response.headers.get("Content-Length", 0))
        self.logger.info(msg=f"Starting download: {file_name} ({total_size / 1024:.2f} KB)")

        # Write a file to local and show download progress
        try:
            with open(file_path, "wb") as file, tqdm(
                    desc=file_name,
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
            ) as progress_bar:
                for data in response.iter_content(chunk_size=self.chunk_size):
                    size = file.write(data)
                    progress_bar.update(size)
            self.logger.info(msg=f"Downloaded: {file_name} -> {file_path}")
        except Exception as e:
            self.logger.error(msg=f"Error writing file {file_name} to {file_path}: {e}")
            raise

    def _clean_filename(self, filename):
        # Remove leading/trailing spaces
        cleaned = filename.strip()

        # Remove any spaces in the filename
        cleaned = cleaned.replace(' ', '')

        # Ensure it has .XPT extension
        if not cleaned.upper().endswith('.XPT'):
            # Remove any existing extension and add .XPT
            name_without_ext = os.path.splitext(cleaned)[0]
            cleaned = name_without_ext + '.XPT'

        return cleaned

    def _extract_zip_file(self, zip_file_path, custom_filename=None):
        self.logger.info(f"Attempting to extract zip file: {zip_file_path}")
        try:
            with zipfile.ZipFile(file=zip_file_path, mode="r") as zip_ref:
                file_list = zip_ref.namelist()
                if not file_list:
                    self.logger.warning(msg=f"No files found in archive: {zip_file_path}")
                    return

                original_file_name = file_list[0]

                # Determine the final filename
                if custom_filename:
                    # Ensure custom filename has .XPT extension
                    if not custom_filename.endswith('.XPT'):
                        final_filename = custom_filename + '.XPT'
                    else:
                        final_filename = custom_filename
                else:
                    # Use cleaned original filename
                    final_filename = self._clean_filename(filename=original_file_name)

                final_file_path = f"{self.des_dir}/{final_filename}"

                if not os.path.exists(path=final_file_path):
                    self.logger.info(msg=f"Extracting '{original_file_name}' to {self.des_dir}")
                    # Extract the original file first
                    zip_ref.extract(member=original_file_name, path=self.des_dir)

                    # If the filename needs to be cleaned up, rename the extracted file
                    if original_file_name != final_filename:
                        original_file_path = f"{self.des_dir}/{original_file_name}"
                        os.rename(src=original_file_path, dst=final_file_path)
                        self.logger.info(msg=f"Renamed '{original_file_name}' to '{final_filename}'")

                    self.logger.info(msg=f"Extracted {final_filename} successfully")
                else:
                    self.logger.info(msg=f"File {final_filename} already exists at {final_file_path}")
        except zipfile.BadZipFile as e:
            self.logger.error(msg=f"Invalid zip file: {zip_file_path}. Error: {e}")
            raise
        except Exception as e:
            self.logger.error(msg=f"Unexpected error during zip extraction: {e}")
            raise

    def load_data(self, urls: list, filenames: list = None):
        """
        Download and load BRFSS dataset by urls

        Parameters
        ----------
            urls: list
                The list of BRFSS data urls in each year
            filenames: list or None
                The list of custom filenames for extracted files. If None, use original names from zip files.
                The .XPT extension will be automatically added if not present.
        """
        self.logger.info(msg="Starting data loading process...")
        for idx, url in enumerate(urls):
            # Get zip filename from URL
            zip_file_name = url.split('/')[-1]
            zip_file_path = f"{self.des_dir}/zip/{zip_file_name.strip()}"

            # Get a custom filename for an extracted file
            custom_filename = None
            if filenames and idx < len(filenames):
                custom_filename = filenames[idx]

            make_dirs(path=os.path.dirname(zip_file_path))

            self.logger.info(msg=f"Processing: {zip_file_name}")
            if not os.path.exists(path=zip_file_path):
                self.logger.info(msg=f"{zip_file_name} not found locally. Downloading...")
                try:
                    self._download_data(
                        url=url,
                        file_name=zip_file_name,
                        file_path=zip_file_path
                    )
                except Exception:
                    self.logger.error(msg=f"Skipping file due to download error: {zip_file_name}")
                    continue
            else:
                self.logger.info(msg=f"{zip_file_name} already exists at {zip_file_path}")

            try:
                self._extract_zip_file(
                    zip_file_path=zip_file_path,
                    custom_filename=custom_filename
                )
            except Exception:
                self.logger.error(msg=f"Skipping file due to extraction error: {zip_file_name}")
                continue

        # Remove zip folder
        zip_folder = f"{self.des_dir}/zip"
        if os.path.exists(path=zip_folder):
            self.logger.info(msg=f"Removing zip folder: {zip_folder}")
            shutil.rmtree(zip_folder)
        else:
            self.logger.warning(msg=f"Zip folder not found: {zip_folder}")

        self.logger.info(msg="Data loading completed successfully.")

class BrfssDataCleaner:
    def __init__(self, log_file: str = None, log_format: str | None = LOG_FORMAT):
        """
        Clean BRFSS dataset based on selected features.

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
            "AnyHealthcare": ("HLTHPLN1", "PRIMINSR", "PRIMINS1"),
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
        self.logger = logging.getLogger(name=__name__)
        self.logger.setLevel(level=logging.DEBUG)
        log_formatter = logging.Formatter(fmt=log_format)
        # Clear existing handlers to avoid duplicate logs
        self.logger.handlers.clear()
        # Handler log to file
        if log_file:
            # Create a log directory
            make_dirs(path=os.path.dirname(log_file))
            log_file_handler = logging.FileHandler(filename=log_file)
            log_file_handler.setLevel(level=logging.INFO)
            log_file_handler.setFormatter(fmt=log_formatter)
            self.logger.addHandler(hdlr=log_file_handler)
        # Handler log to console
        log_console_handler = logging.StreamHandler()
        log_console_handler.setLevel(level=logging.INFO)
        log_console_handler.setFormatter(fmt=log_formatter)
        self.logger.addHandler(hdlr=log_console_handler)

        self.logger.info(msg="BrfssDataCleaner initialized successfully")

    def _process_features(self, df):
        """Process feature values for training data"""
        self.logger.info(msg="Processing feature values for training")
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
        copy_df = copy_df[~copy_df['Sex'].isin([9])]

        # Process Education values
        copy_df = copy_df[~copy_df['Education'].isin([9])]

        self.logger.info(msg=f"Feature processing completed. Shape: {copy_df.shape}")
        return copy_df

    def clean(self, file_path: str, dropna=False) -> Optional[pd.DataFrame]:
        """
        Read data from .XPT file and select columns corresponding to defined features.

        Parameters:
            file_path (str): Path to .XPT data file
            dropna (bool): Whether to drop rows with missing values

        Returns:
            pd.DataFrame | None: DataFrame contains only the required features, or None if there is an error
        """
        self.logger.info(msg=f"Selecting features from {file_path}")

        try:
            brfss_dataframe = pd.read_sas(filepath_or_buffer=file_path, encoding="utf-8")
        except Exception as e:
            self.logger.error(msg=f"Error reading SAS file: {e}")
            return None

        selected_features = {}
        for feature_name, columns in self.feature_column_mapping.items():
            if isinstance(columns, str):
                columns = (columns, )  # convert to tuple for consistency

            selected_col = None
            for col in columns:
                if col in brfss_dataframe.columns:
                    selected_col = col
                    break

            if selected_col:
                selected_features[feature_name] = brfss_dataframe[selected_col]
                self.logger.debug(msg=f"Feature '{feature_name}' mapped to column '{selected_col}'")
            else:
                self.logger.warning(msg=f"No available columns found for feature '{feature_name}'")
                selected_features[feature_name] = None

        if not selected_features:
            self.logger.error(msg="No features could be selected. Returning None.")
            return None

        dataframe = pd.DataFrame(data=selected_features)
        if dropna:
            dataframe.dropna(inplace=True)

        dataframe = self._process_features(dataframe)
        return dataframe