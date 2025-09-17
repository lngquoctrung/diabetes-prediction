import os
import pickle

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