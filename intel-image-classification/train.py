
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, Dataset
from torchvision import datasets, transforms
from torchmetrics.classification import MulticlassAccuracy
from PIL import Image

from utils.train_utils import train_teacher, train_student
from models.teacher import VisionEagle
from models.student import StudentVisionEagle

# --------------------------------------------
# Dynamic Model Import Based on --model_type
# --------------------------------------------
def get_model_class(model_type):
    if model_type == "teacher":
        return VisionEagle
    elif model_type == "student":
        return StudentVisionEagle, VisionEagle
    else:
        raise ValueError("Invalid model type. Choose 'teacher' or 'student'.")

# --------------------------------------------
# Custom Dataset for Prediction-Only Images
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
    parser = argparse.ArgumentParser(description="Train Teacher or Student Model")

    parser.add_argument("--model_type", type=str, choices=["teacher", "student"], default="teacher", help="Specify whether to train the teacher or student model")
    parser.add_argument("--train_path", type=str, required=True, help="Path to training dataset")
    parser.add_argument("--test_path", type=str, required=True, help="Path to testing dataset")
    parser.add_argument("--pred_path", type=str, default=None, help="Path to prediction images (optional)")
    parser.add_argument("--optimizer", type=str, default="SGD", choices=["SGD", "Adam", "AdamW"], help="Optimizer to use: SGD | Adam | AdamW")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--alpha", type=float, default=0.5, help="Alpha parameter for total loss in KD")
    parser.add_argument("--temperature", type=float, default=4.0, help="Temperature for distillation")
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

    optimizer_dict = {
        "SGD": torch.optim.SGD,
        "Adam": torch.optim.Adam,
        "AdamW": torch.optim.AdamW
    }

    # Update model name if default is unchanged
    if args.model_name == "teacher.pth" and args.model_type == "student":
        args.model_name = "student.pth"

    # Create checkpoint directory
    os.makedirs(args.checkpoint_path, exist_ok=True)

    # -------------------------
    # Transforms
    # -------------------------
    global_transforms = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor()
    ])

    # -------------------------
    # Datasets
    # -------------------------
    train_data = datasets.ImageFolder(root=args.train_path, transform=global_transforms)
    test_data = datasets.ImageFolder(root=args.test_path, transform=global_transforms)
    class_names = train_data.classes
    class_num = len(class_names)

    if args.pred_path:
        pred_data = PredictionDataset(root=args.pred_path, transform=global_transforms)

    # -------------------------
    # DataLoaders
    # -------------------------
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, sampler=RandomSampler(test_data))

    # -------------------------
    # Model Setup
    # -------------------------
    if args.model_type == "student":
        StudentClass, TeacherClass = get_model_class(args.model_type)
        model = StudentClass(num_classes=class_num).to(args.device)
        teacher_model = TeacherClass(num_classes=class_num).to(args.device)

        # Load pretrained teacher weights if available
        teacher_path = os.path.join(args.checkpoint_path, "teacher.pth")
        if os.path.exists(teacher_path):
            teacher_model.load_state_dict(torch.load(teacher_path, map_location=args.device))
            print(f"Loaded pretrained teacher model from {teacher_path}")
        else:
            raise FileNotFoundError(f"Teacher weights not found at {teacher_path} — required for student training.")
    else:
        ModelClass = get_model_class(args.model_type)
        model = ModelClass(num_classes=class_num).to(args.device)

    # -------------------------
    # Loss, Optimizer, Scheduler
    # -------------------------
    loss_fn = nn.CrossEntropyLoss()
    accuracy_fn = MulticlassAccuracy(num_classes=class_num).to(args.device)

    opt_class = optimizer_dict[args.optimizer]
    optimizer = opt_class(params=model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

    # -------------------------
    # Train Model
    # -------------------------
    print(f"\nTraining {args.model_type.upper()} model on {args.device} for {args.epochs} epochs...\n")

    if args.model_type == "student":
        history = train_student(
            student_model=model,
            teacher_model=teacher_model,
            train_dataloader=train_loader,
            val_dataloader=test_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            accuracy_fn=accuracy_fn,
            epochs=args.epochs,
            scheduler=scheduler,
            device=args.device,
            alpha=args.alpha,
            temperature=args.temperature,
            model_save_path=args.checkpoint_path,
            model_name=args.model_name
        )
    else:
        history = train_teacher(
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

    print(f"\nTraining complete. Model saved as {args.model_name}.\n")

# --------------------------------------------
# Run Main
# --------------------------------------------
if __name__ == "__main__":
    main()
