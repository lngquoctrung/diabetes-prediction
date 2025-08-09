import os

def make_dirs(path):
    """
    Make directories if they don't exist

    Parameters:
        path (str): Path to the directory
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)