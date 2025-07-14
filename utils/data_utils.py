import os
import glob as gb
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import datasets,transforms

def get_image_size(path, is_pred=False):
    size = []
    folders = [""] if is_pred else os.listdir(path)
    for folder in tqdm(folders, desc="Getting image sizes"):
        for img in gb.glob(path + folder + "/*.jpg"):
            image = plt.imread(img)
            size.append(image.shape)
    return pd.Series(size).value_counts()
def make_torch_dataset_from_image_folder(root, transform):
    return datasets.ImageFolder(root=root, transform=transform)
