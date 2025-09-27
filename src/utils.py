import os
import pickle
from pathlib import Path
import logging

DEFAULT_ROOT_DIR = Path(__file__).resolve().parents[1]

def make_dirs(path):
    """
    Make directories if they don't exist

    Parameters:
        path (str): Path to the directory
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def save_data(path:str, data):
    """
    Export data variable to pickle file

    Parameters:
        path (str): Path to the pickle file
        data (Any): Data to be saved
    """
    make_dirs(os.path.dirname(path))
    with open(path, "wb") as file:
        pickle.dump(data, file)

def load_data(path):
    """
    Load data from the pickle file

    Parammeters:
        path (str): Path to the pickle file
    """
    with open(path, "rb") as file:
        return pickle.load(file)

def sanitize_path(path: str, root_dir: Path = DEFAULT_ROOT_DIR) -> str:
    """
    Remove or anonymize the base part of a path to avoid exposing absolute directories.

    Parameters:
        path (str): The path of a file or directory.
        root_dir (Path): Root directory to anonymize. Defaults to project root.

    Returns:
        str: Sanitized path
    """
    try:
        rel_path = os.path.relpath(path, start=str(root_dir))
        return os.path.join("...", rel_path)
    except Exception:
        return os.path.join("...", os.path.basename(path))
    
def get_configured_logger(name: str, log_file: str, log_format: str) -> logging.Logger:
    logger = logging.getLogger(name=name)

    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(log_format)

    # File handler
    if log_file:
        make_dirs(path=os.path.dirname(log_file))
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.propagate = False
    return logger
