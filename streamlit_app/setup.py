# Setup before running app
import os
import subprocess

home_dir = os.path.expanduser("~")
target_dir = os.path.join(home_dir, "services/public/diabetes-prediction-app")
model_filepath = os.path.join(target_dir, "xgboost_model_checkpoint.pkl")
scaler_filepath = os.path.join(target_dir, "xgb_min_max_scaler.pkl")
data_filepath = os.path.join(target_dir, "brfss_dataset.csv")

def setup_app():
    os.makedirs(target_dir, exist_ok=True)

    files_to_download = {
        model_filepath.split('/')[-1]: "https://github.com/lngquoctrung/diabetes-prediction/releases/download/v1.0.0/xgboost_model_checkpoint.pkl",
        scaler_filepath.split('/')[-1]: "https://github.com/lngquoctrung/diabetes-prediction/releases/download/v1.0.0/xgb_min_max_scaler.pkl",
        data_filepath.split('/')[-1]: "https://github.com/lngquoctrung/diabetes-prediction/releases/download/v1.0.0/brfss_dataset.csv"
    }

    for filename, url in files_to_download.items():
        dest_path = os.path.join(target_dir, filename)
        if not os.path.exists(dest_path):
            try:
                subprocess.run(["curl", "-L", url, "-o", dest_path], check=True)
                print(f"Downloaded {filename}")
            except subprocess.CalledProcessError as e:
                print(f"Failed to download {filename}: {e}")
        else:
            print(f"{filename} already exists. Skipping download.")

