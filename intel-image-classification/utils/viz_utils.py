import torch
import matplotlib.pyplot as plt

def visualize_image_samples(data, class_names, title, figsize=(16, 6), rows_cols=(3, 10), is_pred=False):
    """
    Display a grid of image samples with class labels.
    """
    plt.figure(figsize=figsize)
    plt.suptitle(title, fontsize=16, fontweight='bold')
    rows, cols = rows_cols

    for i in range(1, rows * cols + 1):
        plt.subplot(rows, cols, i)
        random_idx = int(torch.randint(0, len(data), size=[1]).item())

        if is_pred:
            image = data[random_idx]
        else:
            image, label = data[random_idx]
            plt.title(class_names[label].title(), fontdict={"color": "blue", "fontsize": 12})

        plt.imshow(image.permute(1, 2, 0))
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def plot_predictions(model, data, class_names, device):
    """
    Plot predictions for a batch of images with true vs. predicted labels.
    """
    plt.figure(figsize=(16, 9))
    rows, cols = 4, 9

    for i in range(1, rows * cols + 1):
        plt.subplot(rows, cols, i)
        random_idx = int(torch.randint(0, len(data), size=[1]).item())
        image, label = data[random_idx]

        model.eval()
        with torch.inference_mode():
            pred_label = model(image.unsqueeze(0).to(device)).argmax(dim=1)

        plt.imshow(image.permute(1, 2, 0))
        plt.axis('off')
        title = f"Pred: {class_names[pred_label]} \nTrue: {class_names[label]}".title()
        color = "g" if pred_label == label else "r"
        plt.title(title, fontsize=10, c=color)

    plt.tight_layout()
    plt.show()
