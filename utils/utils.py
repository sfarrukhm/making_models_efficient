import os
import glob as gb
import subprocess
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import torch
def download_kaggle_dataset(kaggle_data_path, is_competition=False):
    dataset_name = kaggle_data_path.split("/")[-1]

    # Install Kaggle API
    subprocess.run(["pip", "install", "-q", "kaggle"], check=True)

    # Upload the Kaggle API key (kaggle.json)
    from google.colab import files
    files.upload()

    # Move kaggle.json to the .kaggle directory and set proper permissions
    os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
    subprocess.run(["mv", "kaggle.json", os.path.expanduser("~/.kaggle/")], check=True)
    subprocess.run(["chmod", "600", os.path.expanduser("~/.kaggle/kaggle.json")], check=True)

    # Download the dataset or competition data
    if is_competition:
        subprocess.run(["kaggle", "competitions", "download", "-c", kaggle_data_path], check=True)
    else:
        subprocess.run(["kaggle", "datasets", "download", "-d", kaggle_data_path], check=True)

    # Unzip the downloaded dataset into the created folder
    subprocess.run(["unzip", "-q", f"{dataset_name}.zip", "-d", dataset_name], check=True)

    # Delete the zip file after unzipping
    os.remove(f"{dataset_name}.zip")


def check_and_download_dataset(kaggle_path, colab_path):
    try:
        # Check if dataset is already available in Kaggle
        if gb.glob(pathname=kaggle_path + "*"):
            print("Dataset already available in Kaggle")
            return kaggle_path

        # Check if dataset is already downloaded in Colab
        elif gb.glob(pathname=colab_path + "*"):
            print("Dataset already downloaded in Colab")
            return colab_path

        # Raise an error if the dataset is not found
        raise FileNotFoundError

    except FileNotFoundError:
        # If running in Colab, download the dataset
        try:
            import google.colab
            print("Running in Google Colab")
            print("Downloading Kaggle dataset...")
            download_kaggle_dataset("puneet6060/intel-image-classification")
            return colab_path
        except ImportError:
            raise RuntimeError("Not running in Colab and dataset not found.")

def get_image_size(path, is_pred=False):
  size = []

  if is_pred:
    folders = [""]
  else:
    folders = os.listdir(path)

  for folder in tqdm(folders):
    for img in gb.glob(pathname= path + folder + "/*.jpg"):
      image = plt.imread(img)
      size.append(image.shape)
  return pd.Series(size).value_counts()

def visualize_Images_samples(data, class_names, title, figsize=(16, 6), rows_cols=(3, 10), is_pred=False):
  plt.figure(figsize=(figsize))
  rows, cols = rows_cols
  plt.suptitle(title, fontsize=16, fontweight='bold')

  for i in range(1, rows * cols + 1):
    plt.subplot(rows, cols, i)
    random_idx = int(torch.randint(0, len(data), size=[1]).item())

    if is_pred:
      image = data[random_idx]
    else:
      image, label = data[random_idx]
      plt.title(class_names[label].title(),fontdict={"color":"blue", "fontsize":12})

    plt.imshow(image.permute(1, 2, 0))
    plt.axis('off')

  plt.tight_layout()
  plt.show()

def plot_predictions(model, data, class_names, device):
    """
    Plots a grid of predictions made by a classification model on random samples from a dataset.

    Parameters:
    -----------
    model : torch.nn.Module
        The trained PyTorch model used for making predictions.

    data : torch.utils.data.Dataset
        The dataset to sample images and labels from.

    class_names : list
        List of class names corresponding to label indices. Defaults to global `class_names`.

    device : torch.device
        The device (CPU or GPU) on which model inference is performed. Defaults to global `device`.

    Notes:
    ------
    - The function selects 36 random samples (4 rows × 9 columns).
    - It shows the predicted and true class labels, coloring the title green if the prediction is correct,
      and red otherwise.
    - The function uses torch.inference_mode() to disable gradient calculations for efficiency.
    """

    # Set the figure size for the plot
    plt.figure(figsize=(16, 9))
    rows, cols = 4, 9  # Number of rows and columns in the subplot grid

    # Loop through each subplot
    for i in range(1, rows * cols + 1):
        plt.subplot(rows, cols, i)

        # Randomly select an index from the dataset
        random_idx = int(torch.randint(0, len(data), size=[1]).item())
        image, label = data[random_idx]  # Get the image and its true label

        model.eval()  # Set model to evaluation mode
        with torch.inference_mode():  # Disable gradient computation for inference
            # Predict the class label for the image
            pred_label = model(image.unsqueeze(0).to(device)).argmax(dim=1)

        # Display the image (convert tensor shape from [C, H, W] to [H, W, C])
        plt.imshow(image.permute(1, 2, 0))
        plt.axis('off')  # Hide axis

        # Format the title with predicted and true labels
        title_text = f"Pred: {class_names[pred_label]} \nTrue: {class_names[label]}".title()

        # Set title color based on prediction correctness
        if pred_label == label:
            plt.title(title_text, fontsize=10, c="g")  # Green for correct prediction
        else:
            plt.title(title_text, fontsize=10, c="r")  # Red for incorrect prediction

    # Adjust subplot spacing and display the full plot
    plt.tight_layout()
    plt.show()

def load_model(model_class, checkpoint_path, model_name, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Load a PyTorch model from a checkpoint.

    Args:
        model_class: A callable that returns an instance of the model architecture.
        checkpoint_path (str): Path to the .pth file.
        device (str): 'cuda' or 'cpu'.

    Returns:
        model: Loaded PyTorch model.
    """
    model_path = os.path.join(checkpoint_path+"/"+model_name)
    model = model_class()  # Instantiate the model architecture
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def save_model(model, model_path, model_name):
  os.makedirs(model_path, exist_ok=True)
  model_save_path = os.path.join(model_path, model_name)

  torch.save(obj=model.state_dict(), f=model_save_path)
  print(f"Model_Saved_Path: '{model_save_path}'")
