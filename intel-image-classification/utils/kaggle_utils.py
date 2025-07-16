import os
import subprocess
import glob as gb
from google.colab import files

def download_kaggle_dataset(kaggle_data_path, is_competition=False):
    dataset_name = kaggle_data_path.split("/")[-1]
    subprocess.run(["pip", "install", "-q", "kaggle"], check=True)
    files.upload()
    os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
    subprocess.run(["mv", "kaggle.json", os.path.expanduser("~/.kaggle/")], check=True)
    subprocess.run(["chmod", "600", os.path.expanduser("~/.kaggle/kaggle.json")], check=True)

    if is_competition:
        subprocess.run(["kaggle", "competitions", "download", "-c", kaggle_data_path], check=True)
    else:
        subprocess.run(["kaggle", "datasets", "download", "-d", kaggle_data_path], check=True)

    subprocess.run(["unzip", "-q", f"{dataset_name}.zip", "-d", dataset_name], check=True)
    os.remove(f"{dataset_name}.zip")

def check_and_download_dataset(kaggle_path, colab_path):
    if gb.glob(pathname=kaggle_path + "*"):
        print("Dataset already available in Kaggle")
        return kaggle_path
    elif gb.glob(pathname=colab_path + "*"):
        print("Dataset already downloaded in Colab")
        return colab_path
    else:
        try:
            import google.colab
            print("Running in Google Colab")
            print("Downloading Kaggle dataset...")
            download_kaggle_dataset("puneet6060/intel-image-classification")
            return colab_path
        except ImportError:
            raise RuntimeError("Not running in Colab and dataset not found.")
