import sys
from pathlib import Path
# Add the project path into the python path
root_dir = str(Path(__file__).parent.parent.absolute())
if not root_dir in sys.path:
    sys.path.insert(0, root_dir)

import pandas as pd
import requests
import os
import zipfile
import shutil

from tqdm import tqdm
from typing import Optional
from src.config import RAW_DATA_DIR, LOG_FORMAT
from src.utils import make_dirs, sanitize_path, get_configured_logger

class BrfssDataLoader:
    """Class for download, extract and load BRFSS dataset"""

    def __init__(self, chunk_size=1024, des_dir=RAW_DATA_DIR, logger_name: str | None = __name__, 
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
        self.logger = get_configured_logger(
            name=logger_name, 
            log_file=log_file, 
            log_format=log_format
        )

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
            self.logger.info(msg=f"Downloaded: {file_name} -> {sanitize_path(file_path)}")
        except Exception as e:
            self.logger.error(msg=f"Error writing file {file_name} to {sanitize_path(file_path)}: {e}")
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
        self.logger.info(f"Attempting to extract zip file: {sanitize_path(zip_file_path)}")
        try:
            with zipfile.ZipFile(file=zip_file_path, mode="r") as zip_ref:
                file_list = zip_ref.namelist()
                if not file_list:
                    self.logger.warning(msg=f"No files found in archive: {sanitize_path(zip_file_path)}")
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
                    self.logger.info(msg=f"Extracting '{original_file_name}' to {sanitize_path(self.des_dir)}")
                    # Extract the original file first
                    zip_ref.extract(member=original_file_name, path=self.des_dir)

                    # If the filename needs to be cleaned up, rename the extracted file
                    if original_file_name != final_filename:
                        original_file_path = f"{self.des_dir}/{original_file_name}"
                        os.rename(src=original_file_path, dst=final_file_path)
                        self.logger.info(msg=f"Renamed '{original_file_name}' to '{final_filename}'")

                    self.logger.info(msg=f"Extracted {final_filename} successfully")
                else:
                    self.logger.info(msg=f"File {final_filename} already exists at {sanitize_path(final_file_path)}")
        except zipfile.BadZipFile as e:
            self.logger.error(msg=f"Invalid zip file: {sanitize_path(zip_file_path)}. Error: {e}")
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
                self.logger.info(msg=f"{zip_file_name} already exists at {sanitize_path(zip_file_path)}")

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
            self.logger.info(msg=f"Removing zip folder: {sanitize_path(zip_folder)}")
            shutil.rmtree(zip_folder)
        else:
            self.logger.warning(msg=f"Zip folder not found: {sanitize_path(zip_folder)}")

        self.logger.info(msg="Data loading completed successfully.")

class BrfssDataCleaner:
    def __init__(self, logger_name: str | None = __name__, log_file: str = None, log_format: str | None = LOG_FORMAT):
        """
        Clean BRFSS dataset based on feature metadata.

        Parameters:
            log_file (str): Path to the log file
            log_format (str): Log format
        """
        self.feature_metadata = {
            "Diabetes": {
                "mapping_variables": ["DIABETE3", "DIABETE4"],
                "question": "(Ever told) you have diabetes?",
                "section": "Chronic Health Conditions",
                "mapping_type": "Category",
                "mapping_values": {
                    "admit_values": {
                        1.0: {"label": "Yes", "mapped_value": 2.0},
                        2.0: {"label": "Yes, but female told only during pregnancy", "mapped_value": 0.0},
                        3.0: {"label": "No", "mapped_value": 0.0},
                        4.0: {"label": "No, pre-diabetes or borderline diabetes", "mapped_value": 1.0},
                    },
                    "exclude_values": {
                        7.0: {"label": "Don't know", "mapped_value": None},
                        9.0: {"label": "Refused", "mapped_value": None},
                    }
                }
            },
            "HighBP": {
                "mapping_variables": ["_RFHYPE5", "_RFHYPE6"],
                "question": "Adults who have been told they have high blood pressure by a doctor, nurse, or other health professional",
                "section": "Calculated Variables",
                "mapping_type": "Category",
                "mapping_values": {
                    "admit_values": {
                        1.0: {"label": "No", "mapped_value": 0.0},
                        2.0: {"label": "Yes", "mapped_value": 1.0},
                    },
                    "exclude_values": {
                        9.0: {"label": "Don’t know/Refused/Missing", "mapped_value": None},
                    }
                }
            },
            "HighChol": {
                "mapping_variables": ["TOLDHI2", "TOLDHI3"],
                "question": "Ever told cholesterol is high?",
                "section": "Cholesterol Awareness",
                "mapping_type": "Category",
                "mapping_values": {
                    "admit_values": {
                        1.0: {"label": "Yes", "mapped_value": 1.0},
                        2.0: {"label": "No", "mapped_value": 0.0},
                    },
                    "exclude_values": {
                        7.0: {"label": "Don’t know", "mapped_value": None},
                        9.0: {"label": "Refused", "mapped_value": None},
                    }
                }
            },
            "CholCheck": {
                "mapping_variables": ["_CHOLCH1", "_CHOLCH2", "_CHOLCH3"],
                "question": "Cholesterol check within past five years",
                "section": "Calculated Variables",
                "mapping_type": "Category",
                "mapping_values": {
                    "admit_values": {
                        1.0: {"label": "Had cholesterol checked in past 5 years", "mapped_value": 1.0},
                        2.0: {"label": "Did not have cholesterol checked in past 5 years", "mapped_value": 0.0},
                        3.0: {"label": "Have never had cholesterol checked", "mapped_value": 0.0},
                    },
                    "exclude_values": {
                        9.0: {"label": "Don’t know/Refused/Missing", "mapped_value": None},
                    }
                }
            },
            "BMI": {
                "mapping_variables": ["_BMI5"],
                "question": "Body Mass Index (BMI)",
                "section": "Calculated Variables",
                "mapping_type": "Numerical", 
                "mapping_values": {
                    "admit_values": {},
                    "exclude_values": {}
                }
            },
            "Smoker": {
                "mapping_variables": ["SMOKE100"],
                "question": "Have you smoked at least 100 cigarettes in your entire life?",
                "section": "Tobacco Use",
                "mapping_type": "Category",
                "mapping_values": {
                    "admit_values": {
                        1.0: {"label": "Yes", "mapped_value": 1.0},
                        2.0: {"label": "No", "mapped_value": 0.0},
                    },
                    "exclude_values": {
                        7.0: {"label": "Don’t know", "mapped_value": None},
                        9.0: {"label": "Refused", "mapped_value": None},
                    }
                }
            },
            "Stroke": {
                "mapping_variables": ["CVDSTRK3"],
                "question": "(Ever told) (you had) a stroke?",
                "section": "Chronic Health Conditions",
                "mapping_type": "Category",
                "mapping_values": {
                    "admit_values": {
                        1.0: {"label": "Yes", "mapped_value": 1.0},
                        2.0: {"label": "No", "mapped_value": 0.0},
                    },
                    "exclude_values": {
                        7.0: {"label": "Don’t know/Not Sure", "mapped_value": None},
                        9.0: {"label": "Refused", "mapped_value": None},
                    }
                }
            },
            "HeartDiseaseorAttack": {
                "mapping_variables": ["_MICHD"],
                "question": "Ever had Congenital Heart Defect (CHD) or Myocardial Infarction (MI)?",
                "section": "Calculated Variables",
                "mapping_type": "Category",
                "mapping_values": {
                    "admit_values": {
                        1.0: {"label": "Reported having MI or CHD", "mapped_value": 1.0},
                        2.0: {"label": "Did not report having MI or CHD", "mapped_value": 0.0},
                    },
                    "exclude_values": {}
                }
            },
            "PhysActivity": {
                "mapping_variables": ["_TOTINDA"],
                "question": "Adults who reported doing physical activity or exercise during the past 30 days other than their regular job",
                "section": "Calculated Variables",
                "mapping_type": "Category",
                "mapping_values": {
                    "admit_values": {
                        1.0: {"label": "Had physical activity or exercise", "mapped_value": 1.0},
                        2.0: {"label": "No physical activity or exercise in last 30 days", "mapped_value": 0.0},
                    },
                    "exclude_values": {
                        9.0: {"label": "Don’t know/Refused/Missing", "mapped_value": None},
                    }
                }
            },
            "HvyAlcoholConsump": {
                "mapping_variables": ["_RFDRHV5", "_RFDRHV7", "_RFDRHV8"],
                "question": "Heavy drinkers (adult men having more than 14 drinks per week and adult women having more than 7 drinks per week)",
                "section": "Calculated Variables",
                "mapping_type": "Category",
                "mapping_values": {
                    "admit_values": {
                        1.0: {"label": "No", "mapped_value": 0.0},
                        2.0: {"label": "Yes", "mapped_value": 1.0},
                    },
                    "exclude_values": {
                        9.0: {"label": "Don’t know/Refused/Missing", "mapped_value": None},
                    }
                }
            },
            "AnyHealthcare": {
                "mapping_variables": ["HLTHPLN1", "PRIMINSR", "PRIMINS1"],
                "question": "Have any health care coverage?",
                "section": "Health Care Access",
                "mapping_type": "Category",
                "mapping_values": {
                    "2017-2019": {
                        "admit_values": {
                            1.0: {"label": "Yes", "mapped_value": 1.0},
                            2.0: {"label": "No", "mapped_value": 0.0},
                        },
                        "exclude_values": {
                            7.0: {"label": "Don’t know/Not Sure", "mapped_value": None},
                            9.0: {"label": "Refused", "mapped_value": None},
                        }
                    }, 
                    "2021-2023": {
                        "admit_values": {
                            1.0: {"label": "A plan purchased through an employer or union", "mapped_value": 1.0},
                            2.0: {"label": "A private nongovernmental plan that you or another family member buys on your own", "mapped_value": 1.0},
                            3.0: {"label": "Medicare", "mapped_value": 1.0},
                            4.0: {"label": "Medigap", "mapped_value": 1.0},
                            5.0: {"label": "Medicaid", "mapped_value": 1.0},
                            6.0: {"label": "Children´s Health Insurance Program (CHIP)", "mapped_value": 1.0},
                            7.0: {"label": "Military related health care: TRICARE (CHAMPUS) / VA health care / CHAMP- VA", "mapped_value": 1.0},
                            8.0: {"label": "Indian Health Service", "mapped_value": 1.0},
                            9.0: {"label": "State sponsored health plan", "mapped_value": 1.0},
                            10.0: {"label": "Other government program", "mapped_value": 1.0},
                            88.0: {"label": "No coverage of any type", "mapped_value": 0.0},
                        },
                        "exclude_values": {
                            77.0: {"label": "Don’t know/Not Sure", "mapped_value": None},
                            99.0: {"label": "Refused", "mapped_value": None},
                        }
                    }, 
                },
            },
            "NoDocbcCost": {
                "mapping_variables": ["MEDCOST", "MEDCOST1"],
                "question": "Was there a time in the past 12 months when you needed to see a doctor but could not because you could not afford it?",
                "section": "Health Care Access",
                "mapping_type": "Category",
                "mapping_values": {
                    "admit_values": {
                        1.0: {"label": "Yes", "mapped_value": 1.0},
                        2.0: {"label": "No", "mapped_value": 0.0},
                    },
                    "exclude_values": {
                        7.0: {"label": "Don’t know/Not sure", "mapped_value": None},
                        9.0: {"label": "Refused", "mapped_value": None},
                    }
                }
            },
            "GenHlth": {
                "mapping_variables": ["GENHLTH"],
                "question": "Would you say that in general your health is",
                "section": "Health Status",
                "mapping_type": "Category",
                "mapping_values": {
                    "admit_values": {
                        1.0: {"label": "Excellent", "mapped_value": 1.0},
                        2.0: {"label": "Very good", "mapped_value": 2.0},
                        3.0: {"label": "Good", "mapped_value": 3.0},
                        4.0: {"label": "Fair", "mapped_value": 4.0},
                        5.0: {"label": "Poor", "mapped_value": 5.0},
                    },
                    "exclude_values": {
                        7.0: {"label": "Don’t know/Not sure", "mapped_value": None},
                        9.0: {"label": "Refused", "mapped_value": None},
                    }
                }
            },
            "MentHlth": {
                "mapping_variables": ["MENTHLTH"],
                "question": "Number of days mental health not good",
                "section": "Healthy Days",
                "mapping_type": "Numerical",
                "mapping_values": {
                    "admit_values": {
                        88.0: {"label": "None", "mapped_value": 0.0},
                    },
                    "exclude_values": {
                        77.0: {"label": "Don’t know/Not sure", "mapped_value": None},
                        99.0: {"label": "Refused", "mapped_value": None},
                    }
                }
            },
            "PhysHlth": {
                "mapping_variables": ["PHYSHLTH"],
                "question": "Number of days physical health not good",
                "section": "Healthy Days",
                "mapping_type": "Numerical",
                "mapping_values": {
                    "admit_values": {
                        88.0: {"label": "None", "mapped_value": 0.0},
                    },
                    "exclude_values": {
                        77.0: {"label": "Don’t know/Not sure", "mapped_value": None},
                        99.0: {"label": "Refused", "mapped_value": None},
                    }
                }
            },
            "DiffWalk": {
                "mapping_variables": ["DIFFWALK"],
                "question": "Do you have serious difficulty walking or climbing stairs?",
                "section": "Disability",
                "mapping_type": "Category",
                "mapping_values": {
                    "admit_values": {
                        1.0: {"label": "Yes", "mapped_value": 1.0},
                        2.0: {"label": "No", "mapped_value": 0.0},
                    },
                    "exclude_values": {
                        7.0: {"label": "Don’t know/Not Sure", "mapped_value": None},
                        9.0: {"label": "Refused", "mapped_value": None},
                    }
                }
            },
            "Sex": {
                "mapping_variables": ["SEX", "_SEX"],
                "question": "What was your sex at birth? Was it male or female?",
                "section": "Demographics",
                "mapping_type": "Category",
                "mapping_values": {
                    "admit_values": {
                        1.0: {"label": "Male", "mapped_value": 0.0},
                        2.0: {"label": "Female", "mapped_value": 1.0},
                    },
                    "exclude_values": {}
                }
            },
            "Age": {
                "mapping_variables": ["_AGEG5YR"],
                "question": "Reported age in five-year age categories calculated variable",
                "section": "Calculated Variables",
                "mapping_type": "Category",
                "mapping_values": {
                    "admit_values": {
                        1.0: {"label": "Age 18 to 24", "mapped_value": 1.0},
                        2.0: {"label": "Age 25 to 29", "mapped_value": 2.0},
                        3.0: {"label": "Age 30 to 34", "mapped_value": 3.0},
                        4.0: {"label": "Age 35 to 39", "mapped_value": 4.0},
                        5.0: {"label": "Age 40 to 44", "mapped_value": 5.0},
                        6.0: {"label": "Age 45 to 49", "mapped_value": 6.0},
                        7.0: {"label": "Age 50 to 54", "mapped_value": 7.0},
                        8.0: {"label": "Age 55 to 59", "mapped_value": 8.0},
                        9.0: {"label": "Age 60 to 64", "mapped_value": 9.0},
                        10.0: {"label": "Age 65 to 69", "mapped_value": 10.0},
                        11.0: {"label": "Age 70 to 74", "mapped_value": 11.0},
                        12.0: {"label": "Age 75 to 79", "mapped_value": 12.0},
                        13.0: {"label": "Age 80 or older", "mapped_value": 13.0},
                    },
                    "exclude_values": {
                        14.0: {"label": "Don’t know/Refused/Missing", "mapped_value": None},
                    }
                }
            },
            "Education": {
                "mapping_variables": ["EDUCA"],
                "question": "What is the highest grade or year of school you completed?",
                "section": "Demographics",
                "mapping_type": "Category",
                "mapping_values": {
                    "admit_values": {
                        1.0: {"label": "Never attended school or only kindergarten", "mapped_value": 1.0},
                        2.0: {"label": "Grades 1 through 8 (Elementary)", "mapped_value": 2.0},
                        3.0: {"label": "Grades 9 through 11 (Some high school)", "mapped_value": 3.0},
                        4.0: {"label": "Grade 12 or GED (High school graduate)", "mapped_value": 4.0},
                        5.0: {"label": "College 1 year to 3 years (Some college or technical school)", "mapped_value": 5.0},
                        6.0: {"label": "College 4 years or more (College graduate)", "mapped_value": 6.0},
                    },
                    "exclude_values": {
                        9.0: {"label": "Refused", "mapped_value": None},
                    }
                }
            },
            "Income": {
                "mapping_variables": ["INCOME2", "INCOME3"],
                "question": "",
                "section": "",
                "mapping_type": "Category",
                "mapping_values": {
                    "admit_values": {
                        1.0: {"label": "Less than $10,000", "mapped_value": 1.0},
                        2.0: {"label": "Less than $15,000 ($10,000 to < $15,000)", "mapped_value": 2.0},
                        3.0: {"label": "Less than $20,000 ($15,000 to < $20,000)", "mapped_value": 3.0},
                        4.0: {"label": "Less than $25,000 ($20,000 to < $25,000)", "mapped_value": 4.0},
                        5.0: {"label": "Less than $35,000 ($25,000 to < $35,000)", "mapped_value": 5.0},
                        6.0: {"label": "Less than $50,000 ($35,000 to < $50,000)", "mapped_value": 6.0},
                        7.0: {"label": "Less than $75,000 ($50,000 to < $75,000)", "mapped_value": 7.0},
                        8.0: {"label": "Less than $100,000? ($75,000 to < $100,000)", "mapped_value": 8.0},
                        9.0: {"label": "Less than $150,000? ($100,000 to < $150,000)", "mapped_value": 9.0},
                        10.0: {"label": "Less than $200,000? ($150,000 to < $200,000)", "mapped_value": 10.0},
                        11.0: {"label": "$200,000 or more", "mapped_value": 11.0},
                    },
                    "exclude_values": {
                        77.0: {"label": "Don’t know/Not Sure", "mapped_value": None},
                        99.0: {"label": "Refused", "mapped_value": None},
                    }
                }
            },
            "Depression": {
                "mapping_variables": ["ADDEPEV2", "ADDEPEV3"],
                "question": "(Ever told) you had a depressive disorder?",
                "section": "Chronic Health Conditions",
                "mapping_type": "Category",
                "mapping_values": {
                    "admit_values": {
                        1.0: {"label": "Yes", "mapped_value": 1.0},
                        2.0: {"label": "No", "mapped_value": 0.0},
                    },
                    "exclude_values": {
                        7.0: {"label": "Don’t know/Not Sure", "mapped_value": None},
                        9.0: {"label": "Refused", "mapped_value": None},
                    }
                }
            },
            "CognitiveIssues": {
                "mapping_variables": ["DECIDE"],
                "question": "Difficulty Concentrating or Remembering",
                "section": "Demographics",
                "mapping_type": "Category",
                "mapping_values": {
                    "admit_values": {
                        1.0: {"label": "Yes", "mapped_value": 1.0},
                        2.0: {"label": "No", "mapped_value": 0.0},
                    },
                    "exclude_values": {
                        7.0: {"label": "Don’t know/Not Sure", "mapped_value": None},
                        9.0: {"label": "Refused", "mapped_value": None},
                    }
                }
            },
            "KidneyDisease": {
                "mapping_variables": ["CHCKIDNY", "CHCKDNY2"],
                "question": "Ever told you have kidney disease?",
                "section": "Chronic Health Conditions",
                "mapping_type": "Category",
                "mapping_values": {
                    "admit_values": {
                        1.0: {"label": "Yes", "mapped_value": 1.0},
                        2.0: {"label": "No", "mapped_value": 0.0},
                    },
                    "exclude_values": {
                        7.0: {"label": "Don’t know/Not Sure", "mapped_value": None},
                        9.0: {"label": "Refused", "mapped_value": None},
                    }
                }
            },
            "DiagnosedHeartAttack": {
                "mapping_variables": ["CVDINFR4"],
                "question": "Ever Diagnosed with Heart Attack?",
                "section": "Chronic Health Conditions",
                "mapping_type": "Category",
                "mapping_values": {
                    "admit_values": {
                        1.0: {"label": "Yes", "mapped_value": 1.0},
                        2.0: {"label": "No", "mapped_value": 0.0},
                    },
                    "exclude_values": {
                        7.0: {"label": "Don’t know/Not Sure", "mapped_value": None},
                        9.0: {"label": "Refused", "mapped_value": None},
                    }
                }
            },
            "CoronaryHeartDisease": {
                "mapping_variables": ["CVDCRHD4"],
                "question": "Ever Diagnosed with Angina or Coronary Heart Disease",
                "section": "Chronic Health Conditions",
                "mapping_type": "Category",
                "mapping_values": {
                    "admit_values": {
                        1.0: {"label": "Yes", "mapped_value": 1.0},
                        2.0: {"label": "No", "mapped_value": 0.0},
                    },
                    "exclude_values": {
                        7.0: {"label": "Don’t know/Not Sure", "mapped_value": None},
                        9.0: {"label": "Refused", "mapped_value": None},
                    }
                }
            },
            "COPD": {
                "mapping_variables": ["CHCCOPD1", "CHCCOPD2", "CHCCOPD3"],
                "question": "Ever told you had Chronic Obstructive Pulmonary Disease (COPD) emphysema or chronic bronchitis?",
                "section": "Chronic Health Conditions",
                "mapping_type": "Category",
                "mapping_values": {
                    "admit_values": {
                        1.0: {"label": "Yes", "mapped_value": 1.0},
                        2.0: {"label": "No", "mapped_value": 0.0},
                    },
                    "exclude_values": {
                        7.0: {"label": "Don’t know/Not Sure", "mapped_value": None},
                        9.0: {"label": "Refused", "mapped_value": None},
                    }
                }
            },
            "AlcoholDays": {
                "mapping_variables": ["ALCDAY4", "ALCDAY5"],
                "question": "Days in past 30 had alcoholic beverage",
                "section": "Alcohol Consumption",
                "mapping_type": "Numerical",
                "mapping_values": {
                    "admit_values": {
                        888.0: {"label": "None", "mapped_value": 0.0},
                    },
                    "exclude_values": {
                        777.0: {"label": "Don’t know/Not Sure", "mapped_value": None},
                        999.0: {"label": "Refused", "mapped_value": None},
                    }
                }
            },
            "LastCheckup": {
                "mapping_variables": ["CHECKUP1"],
                "question": "About how long has it been since you last visited a doctor for a routine checkup?",
                "section": "Health Care Access",
                "mapping_type": "Category",
                "mapping_values": {
                    "admit_values": {
                        1.0: {"label": "Within past year (anytime < 12 months ago)", "mapped_value": 1.0},
                        2.0: {"label": "Within past 2 years (1 year but < 2 years ago)", "mapped_value": 2.0},
                        3.0: {"label": "Within past 5 years (2 years but < 5 years ago)", "mapped_value": 3.0},
                        4.0: {"label": "5 or more years ago", "mapped_value": 0.0},
                        8.0: {"label": "Never", "mapped_value": 0.0},
                    },
                    "exclude_values": {
                        7.0: {"label": "Don’t know/Not Sure", "mapped_value": None},
                        9.0: {"label": "Refused", "mapped_value": None},
                    }
                }
            },
            "HasPersonalDoctor": {
                "mapping_variables": ["PERSDOC2", "PERSDOC3"],
                "question": "",
                "section": "",
                "mapping_type": "Category",
                "mapping_values": {
                    "admit_values": {
                        1.0: {"label": "Yes, only one", "mapped_value": 1.0},
                        2.0: {"label": "More than one", "mapped_value": 1.0},
                        3.0: {"label": "No", "mapped_value": 0.0},
                    },
                    "exclude_values": {
                        7.0: {"label": "Don’t know/Not Sure", "mapped_value": None},
                        9.0: {"label": "Refused", "mapped_value": None},
                    }
                }
            },
            "CholesterolMeds": {
                "mapping_variables": ["CHOLMED1", "CHOLMED2", "CHOLMED3"],
                "question": "Are you currently taking medicine prescribed by your doctor or other health professional for your cholesterol?",
                "section": "Cholesterol Awareness",
                "mapping_type": "Category",
                "mapping_values": {
                    "admit_values": {
                        1.0: {"label": "Yes", "mapped_value": 1.0},
                        2.0: {"label": "No", "mapped_value": 0.0},
                    },
                    "exclude_values": {
                        7.0: {"label": "Don’t know/Not Sure", "mapped_value": None},
                        9.0: {"label": "Refused", "mapped_value": None},
                    }
                }
            },
            "MaritalStatus": {
                "mapping_variables": ["MARITAL"],
                "question": "Are you: (marital status)?",
                "section": "",
                "mapping_type": "Category",
                "mapping_values": {
                    "admit_values": {
                        1.0: {"label": "Married", "mapped_value": 1.0},
                        2.0: {"label": "Divorced", "mapped_value": 2.0},
                        3.0: {"label": "Widowed", "mapped_value": 3.0},
                        4.0: {"label": "Separated", "mapped_value": 4.0},
                        5.0: {"label": "Never married", "mapped_value": 5.0},
                        6.0: {"label": "A member of an unmarried couple", "mapped_value": 6.0},
                    },
                    "exclude_values": {
                        9.0: {"label": "Refused", "mapped_value": None},
                    }
                }
            },
            "EmploymentStatus": {
                "mapping_variables": ["EMPLOY1"],
                "question": "Are you currently employed?",
                "section": "",
                "mapping_type": "Category",
                "mapping_values": {
                    "admit_values": {
                        1.0: {"label": "Employed for wages", "mapped_value": 1.0},
                        2.0: {"label": "Self-employed", "mapped_value": 2.0},
                        3.0: {"label": "Out of work for 1 year or more", "mapped_value": 3.0},
                        4.0: {"label": "Out of work for < 1 year", "mapped_value": 4.0},
                        5.0: {"label": "A homemaker", "mapped_value": 5.0},
                        6.0: {"label": "A student", "mapped_value": 6.0},
                        7.0: {"label": "Retired", "mapped_value": 7.0},
                        8.0: {"label": "Unable to work", "mapped_value": 8.0},
                    },
                    "exclude_values": {
                        9.0: {"label": "Refused", "mapped_value": None},
                    }
                }
            },
            "CannotAffordDoctor": {
                "mapping_variables": ["MEDCOST", "MEDCOST1"],
                "question": "Was there a time in the past 12 months when you needed to see a doctor but could not because you could not afford it?",
                "section": "Health Care Access",
                "mapping_type": "Category",
                "mapping_values": {
                    "admit_values": {
                        1.0: {"label": "Yes", "mapped_value": 1.0},
                        2.0: {"label": "No", "mapped_value": 0.0},
                    },
                    "exclude_values": {
                        7.0: {"label": "Don’t know/Not Sure", "mapped_value": None},
                        9.0: {"label": "Refused", "mapped_value": None},
                    }
                }
            },
        }
        
        # Log configuration
        self.logger = get_configured_logger(
            name=logger_name, 
            log_file=log_file, 
            log_format=log_format
        )

        self.logger.info(msg="BrfssDataCleaner initialized successfully")

    def _determine_year_period(self, df):
        """Determine year period for AnyHealthcare feature mapping"""
        # Try to determine year from common year columns
        if "HLTHPLN1" in df.columns:
            return "2017-2019"
        else:
            return "2021-2023"

    def _process_feature(self, df, original_df, feature_name, metadata, selected_column):
        """Process individual feature based on metadata - returns tuple (processed_series, mask)"""
        self.logger.info(f"Processing feature: {feature_name}")
        self.logger.info(f"Source Column: {selected_column}")
        self.logger.info(f"Question: {metadata['question']}")
        self.logger.info(f"Section: {metadata['section']}")
        
        # Work on a copy to avoid modifying the original
        working_series = df[feature_name].copy()
        
        # Handle special cases
        if feature_name == "AnyHealthcare":
            year_period = self._determine_year_period(original_df)
            self.logger.info(f"Using year period: {year_period} for AnyHealthcare mapping")
            mapping_values = metadata['mapping_values'][year_period]
        elif feature_name == "BMI":
            # Special handling for BMI - divide by 100 and round
            self.logger.info("Raw value mapping:")
            self.logger.info("Dividing BMI values by 100 and rounding to nearest integer")
            working_series = working_series.div(100).round(0)
            return working_series, pd.Series(True, index=df.index)  # Keep all rows for BMI
        elif feature_name == "AlcoholDays":
            # Special handling for AlcoholDays - modulo 100 for last 2 digits
            self.logger.info("Raw value mapping:")
            
            # Handle special values first
            mapping_values = metadata['mapping_values']
            for val, info in mapping_values['admit_values'].items():
                if val in working_series.values:
                    self.logger.info(f"{int(val)} → {info['mapped_value']} ({info['label']})")
                    working_series = working_series.replace(val, info['mapped_value'])
            
            # Apply modulo 100 for other values (excluding special values and exclude values)
            all_special_values = list(mapping_values['admit_values'].keys()) + list(mapping_values['exclude_values'].keys())
            mask = ~working_series.isin(all_special_values)
            working_series.loc[mask] = working_series.loc[mask] % 100
            self.logger.info("Other values: taking modulo 100 to get last 2 digits")
            
            # Create mask for exclude values
            exclude_values = list(mapping_values['exclude_values'].keys())
            if exclude_values:
                exclude_labels = [mapping_values['exclude_values'][val]['label'] for val in exclude_values]
                self.logger.info(f"Dropping values: {', '.join([f'{int(val)} ({label})' for val, label in zip(exclude_values, exclude_labels)])}")
                return working_series, ~working_series.isin(exclude_values)
            
            return working_series, pd.Series(True, index=df.index)
        else:
            mapping_values = metadata['mapping_values']
        
        # Log raw value mapping for admit values
        if mapping_values['admit_values']:
            self.logger.info("Raw value mapping:")
            for original_val, mapping_info in mapping_values['admit_values'].items():
                if original_val in working_series.values:
                    mapped_val = mapping_info['mapped_value']
                    label = mapping_info['label']
                    self.logger.info(f"{int(original_val)} → {mapped_val} ({label})")
        
        # Apply value mapping for admit values - use map to avoid conflicts
        if mapping_values['admit_values']:
            mapping_dict = {}
            for original_val, mapping_info in mapping_values['admit_values'].items():
                mapping_dict[original_val] = mapping_info['mapped_value']
            
            # Apply mapping all at once to avoid sequential replace conflicts
            working_series = working_series.map(mapping_dict).fillna(working_series)
        
        # Create mask for exclude values
        exclude_values = list(mapping_values['exclude_values'].keys())
        if exclude_values:
            exclude_labels = [mapping_values['exclude_values'][val]['label'] for val in exclude_values]
            self.logger.info(f"Dropping values: {', '.join([f'{int(val)} ({label})' for val, label in zip(exclude_values, exclude_labels)])}")
            mask = ~working_series.isin(exclude_values)
            return working_series, mask
        
        return working_series, pd.Series(True, index=df.index)

    def _process_features(self, df, original_df):
        """Process all features based on metadata"""
        self.logger.info(msg="Processing feature values for training")
        processed_df = df.copy()
        
        # Create a master mask to track which rows to keep
        master_mask = pd.Series(True, index=processed_df.index)
        
        for feature_name, metadata in self.feature_metadata.items():
            if feature_name in processed_df.columns:
                # Find the selected column for this feature
                selected_column = None
                for col in metadata['mapping_variables']:
                    if col in original_df.columns:
                        selected_column = col
                        break
                
                # Process feature and get both processed series and mask
                processed_series, feature_mask = self._process_feature(processed_df, original_df, feature_name, metadata, selected_column)
                
                # Update the processed dataframe with the processed series
                processed_df[feature_name] = processed_series
                
                # Update master mask
                master_mask = master_mask & feature_mask
                
                self.logger.info(f"Feature '{feature_name}' processing completed")
                self.logger.info("=" * 50)
        
        # Apply the master mask to filter the dataframe
        processed_df = processed_df[master_mask]
        
        self.logger.info(msg=f"Feature processing completed. Final shape: {processed_df.shape}")
        return processed_df

    def clean(self, file_path: str, dropna=False, year: int = None) -> Optional[pd.DataFrame]:
        """
        Read data from .XPT file and select columns corresponding to defined features.

        Parameters:
            file_path (str): Path to .XPT data file
            dropna (bool): Whether to drop rows with missing values
            year (int): Year of the dataset for special handling

        Returns:
            pd.DataFrame | None: DataFrame contains only the required features, or None if there is an error
        """
        self.logger.info(msg=f"Clean features from {sanitize_path(file_path)}")

        try:
            brfss_dataframe = pd.read_sas(filepath_or_buffer=file_path, encoding="utf-8")
        except Exception as e:
            self.logger.error(msg=f"Error reading SAS file: {e}")
            return None

        selected_features = {}
        for feature_name, metadata in self.feature_metadata.items():
            mapping_variables = metadata['mapping_variables']
            
            selected_col = None
            for col in mapping_variables:
                if col in brfss_dataframe.columns:
                    selected_col = col
                    break

            if selected_col:
                selected_features[feature_name] = brfss_dataframe[selected_col]
                self.logger.debug(msg=f"Feature '{feature_name}' mapped to column '{selected_col}'")
            else:
                self.logger.warning(msg=f"No available columns found for feature '{feature_name}': {mapping_variables}")
                selected_features[feature_name] = None

        if not selected_features:
            self.logger.error(msg="No features could be selected. Returning None.")
            return None

        dataframe = pd.DataFrame(data=selected_features)
        
        null_counts = dataframe.isnull().sum()
        for col, null_count in null_counts.items():
            if null_count > 0:
                self.logger.info(msg=f"Column '{col}' has {null_count} missing values.")

        if dropna:
            # Remove columns that are None (no mapping found)
            dataframe = dataframe.dropna(axis=1, how='all')
            before_drop = len(dataframe)
            dataframe.dropna(inplace=True)
            after_drop = len(dataframe)
            dropped_rows = before_drop - after_drop
            self.logger.info(msg=f"Dropped {dropped_rows} rows due to missing values.")

        dataframe = self._process_features(df=dataframe, original_df=brfss_dataframe)
        return dataframe