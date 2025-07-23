import sys
import requests
import logging
import os
import zipfile
import shutil

from pathlib import Path
from tqdm import tqdm

# Add the project path into the python path
root_dir = str(Path(__file__).parent.parent.parent.absolute())
if not root_dir in sys.path:
    sys.path.insert(0, root_dir)

from src.config import RAW_DATA_DIR, LOG_FORMAT
from src.utils import make_dirs

class BrfssDataLoader:
    """Class for download, extract and load BRFSS dataset"""
    def __init__(self, chunk_size=1024, des_dir=RAW_DATA_DIR, log_file: str | None = None, log_format: str | None = LOG_FORMAT):
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
        make_dirs(os.path.dirname(des_dir))

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

    def _download_data(self, url, file_name, file_path):
        self.logger.info(f"Navigating to URL: {url}")
        try:
            response = requests.get(url=url, stream=True)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to download {file_name} from {url}. Error: {e}")
            raise

        # Get the size of a file
        total_size = int(response.headers.get("Content-Length", 0))
        self.logger.info(f"Starting download: {file_name} ({total_size / 1024:.2f} KB)")

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
            self.logger.info(f"Downloaded: {file_name} -> {file_path}")
        except Exception as e:
            self.logger.error(f"Error writing file {file_name} to {file_path}: {e}")
            raise

    def _clean_filename(self, filename):
        # Remove leading/trailing spaces
        cleaned = filename.strip()
        
        # Remove any spaces in the filename
        cleaned = cleaned.replace(' ', '')
        
        # Remove any other unwanted characters if needed
        # cleaned = re.sub(r'[^\w\-_\.]', '', cleaned)
        
        # Ensure it has .XPT extension
        if not cleaned.upper().endswith('.XPT'):
            # Remove any existing extension and add .XPT
            name_without_ext = os.path.splitext(cleaned)[0]
            cleaned = name_without_ext + '.XPT'
        
        return cleaned

    def _extract_zip_file(self, zip_file_path, custom_filename=None):
        self.logger.info(f"Attempting to extract zip file: {zip_file_path}")
        try:
            with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
                file_list = zip_ref.namelist()
                if not file_list:
                    self.logger.warning(f"No files found in archive: {zip_file_path}")
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
                    final_filename = self._clean_filename(original_file_name)
                
                final_file_path = self.des_dir / final_filename

                if not os.path.exists(final_file_path):
                    self.logger.info(f"Extracting '{original_file_name}' to {self.des_dir}")
                    # Extract the original file first
                    zip_ref.extract(original_file_name, self.des_dir)
                    
                    # If the filename needs to be cleaned up, rename the extracted file
                    if original_file_name != final_filename:
                        original_file_path = self.des_dir / original_file_name
                        os.rename(original_file_path, final_file_path)
                        self.logger.info(f"Renamed '{original_file_name}' to '{final_filename}'")
                    
                    self.logger.info(f"Extracted {final_filename} successfully")
                else:
                    self.logger.info(f"File {final_filename} already exists at {final_file_path}")
        except zipfile.BadZipFile as e:
            self.logger.error(f"Invalid zip file: {zip_file_path}. Error: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error during zip extraction: {e}")
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
        self.logger.info("Starting data loading process...")
        for idx, url in enumerate(urls):
            # Get zip filename from URL
            zip_file_name = url.split('/')[-1]
            zip_file_path = self.des_dir / "zip" / zip_file_name.strip()

            # Get a custom filename for an extracted file
            custom_filename = None
            if filenames and idx < len(filenames):
                custom_filename = filenames[idx]

            make_dirs(os.path.dirname(zip_file_path))

            self.logger.info(f"Processing: {zip_file_name}")
            if not os.path.exists(zip_file_path):
                self.logger.info(f"{zip_file_name} not found locally. Downloading...")
                try:
                    self._download_data(url, zip_file_name, zip_file_path)
                except Exception:
                    self.logger.error(f"Skipping file due to download error: {zip_file_name}")
                    continue
            else:
                self.logger.info(f"{zip_file_name} already exists at {zip_file_path}")

            try:
                self._extract_zip_file(zip_file_path, custom_filename)
            except Exception:
                self.logger.error(f"Skipping file due to extraction error: {zip_file_name}")
                continue

        # Remove zip folder
        zip_folder = self.des_dir / "zip"
        if zip_folder.exists():
            self.logger.info(f"Removing zip folder: {zip_folder}")
            shutil.rmtree(zip_folder)
        else:
            self.logger.warning(f"Zip folder not found: {zip_folder}")

        self.logger.info("Data loading completed successfully.")
