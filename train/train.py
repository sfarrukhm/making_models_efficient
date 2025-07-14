import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, Dataset
from torchvision import datasets, transforms
from torchmetrics.classification import MulticlassAccuracy
from PIL import Image

from models.teacher import VisionEagle
from models.train_evaluate_predict import train_model


# --------------------------------------------
# Custom Dataset for Inference-only Predictions
# --------------------------------------------
class PredictionDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.image_names = os.listdir(root)
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.root, self.image_names[idx])).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


# --------------------------------------------
# Argument Parser
# --------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train VisionEagle Model")
    
    parser.add_argument("--train_path", type=str, required=True, help="Path to training dataset")
    parser.add_argument("--test_path", type=str, required=True, help="Path to testing dataset")
    parser.add_argument("--pred_path", type=str, default=None, help="Path to prediction images (optional)")

    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--img_size", type=int, default=100, help="Image resize size (square)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--checkpoint_path", type=str, default="models/checkpoints", help="Path to save checkpoints")
    parser.add_argument("--model_name", type=str, default="teacher.pth", help="Model filename")

    return parser.parse_args()


# --------------------------------------------
# Main Function
# --------------------------------------------
def main():
    args = parse_args()

    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.checkpoint_path, exist_ok=True)

    # -------------------------
    # Data Transforms
    # -------------------------
    global_transforms = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor()
    ])

    # -------------------------
    # Load Datasets
    # -------------------------
    train_data = datasets.ImageFolder(root=args.train_path, transform=global_transforms)
    test_data = datasets.ImageFolder(root=args.test_path, transform=global_transforms)
    
    class_names = train_data.classes
    class_num = len(class_names)

    # Optional prediction dataset
    if args.pred_path:
        pred_data = PredictionDataset(root=args.pred_path, transform=global_transforms)

    # -------------------------
    # Dataloaders
    # -------------------------
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        sampler=RandomSampler(test_data)
    )

    # -------------------------
    # Model, Loss, Optimizer
    # -------------------------
    model = VisionEagle(num_classes=class_num).to(args.device)
    loss_fn = nn.CrossEntropyLoss()
    accuracy_fn = MulticlassAccuracy(num_classes=class_num).to(args.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

    # -------------------------
    # Train Model
    # -------------------------
    print(f"Training model on {args.device} for {args.epochs} epochs...")

    history = train_model(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=test_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        accuracy_fn=accuracy_fn,
        epochs=args.epochs,
        scheduler=scheduler,
        device=args.device,
        model_save_path=args.checkpoint_path,
        model_name=args.model_name
    )

    print("Training complete. Model saved.")


if __name__ == "__main__":
    main()
