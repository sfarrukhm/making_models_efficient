import torch
from tqdm.auto import tqdm
import os
import torch
import torch.nn.functional as F
from utils.train_evaluate_predict import evaluate_model
### Student Training Setup
def train_student(
    student_model: torch.nn.Module,
    teacher_model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    accuracy_fn,
    epochs: int,
    scheduler=None,
    device="cuda",
    alpha=0.5,
    temperature=4.0,
    model_save_path="checkpoints/",
    model_name="student.pth"
):
    """
    Trains a student model using knowledge distillation from a teacher model.

    Args: Same as train_teacher with alpha/temperature.

    Returns:
        dict: training and validation loss/accuracy history
    """

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    teacher_model.eval()  # Teacher never trains

    for epoch in range(epochs):
        student_model.train()
        train_loss = 0.0
        train_acc = 0.0

        loop = tqdm(train_dataloader, desc=f"[Student] Epoch {epoch+1}/{epochs}", leave=False)

        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():
                teacher_logits = teacher_model(images)

            student_logits = student_model(images)

            # Distillation loss
            soft_teacher_probs = F.log_softmax(teacher_logits / temperature, dim=1)
            soft_student_probs = F.log_softmax(student_logits / temperature, dim=1)

            soft_loss = F.kl_div(
                soft_student_probs, 
                soft_teacher_probs.exp(), 
                reduction="batchmean"
            ) * (temperature ** 2)

            hard_loss = loss_fn(student_logits, labels)

            loss = alpha * soft_loss + (1 - alpha) * hard_loss
            acc = accuracy_fn(student_logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += acc.item()
            loop.set_postfix(loss=loss.item(), acc=acc.item())

        train_loss /= len(train_dataloader)
        train_acc /= len(train_dataloader)

        # Validation
        student_model.eval()
        val_loss = 0.0
        val_acc = 0.0

        with torch.inference_mode():
            for val_images, val_labels in val_dataloader:
                val_images, val_labels = val_images.to(device), val_labels.to(device)
                val_outputs = student_model(val_images)
                v_loss = loss_fn(val_outputs, val_labels)
                v_acc = accuracy_fn(val_outputs, val_labels)

                val_loss += v_loss.item()
                val_acc += v_acc.item()

        val_loss /= len(val_dataloader)
        val_acc /= len(val_dataloader)

        # Save metrics
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"[Student Epoch {epoch+1}/{epochs}] "
              f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(model_save_path, model_name)
            torch.save(student_model.state_dict(), save_path)
            print(f"✅ Saved best student model to {save_path}")

        if scheduler:
            scheduler.step()

    return history



### Teacher Training Setup
def train_teacher(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    accuracy_fn,
    epochs: int,
    scheduler=None,
    device="cuda",
    model_save_path="checkpoints/",
    model_name="model.pth"
):
    """
    Trains the model and evaluates on validation data.

    Args:
        model (torch.nn.Module): Model to train.
        train_dataloader (DataLoader): Training data loader.
        val_dataloader (DataLoader): Validation data loader.
        optimizer (torch.optim.Optimizer): Optimizer.
        loss_fn: Loss function.
        accuracy_fn: Accuracy metric function.
        epochs (int): Number of epochs.
        scheduler: Optional learning rate scheduler.
        device (str): "cuda" or "cpu".
        model_save_path (str): Folder to save the model.
        model_name (str): Filename for the saved model.

    Returns:
        dict: training and validation loss/accuracy history
    """

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_acc = 0

        loop = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        for batch in loop:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            # Forward
            y_pred = model(images)
            loss = loss_fn(y_pred, labels)
            acc = accuracy_fn(y_pred, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Metrics
            train_loss += loss.item()
            train_acc += acc.item()

            loop.set_postfix(loss=loss.item(), acc=acc.item())

        train_loss /= len(train_dataloader)
        train_acc /= len(train_dataloader)

        # Validation
        model.eval()
        val_loss = 0
        val_acc = 0

        with torch.inference_mode():
            for val_batch in val_dataloader:
                val_images, val_labels = val_batch
                val_images, val_labels = val_images.to(device), val_labels.to(device)

                val_pred = model(val_images)
                v_loss = loss_fn(val_pred, val_labels)
                v_acc = accuracy_fn(val_pred, val_labels)

                val_loss += v_loss.item()
                val_acc += v_acc.item()

        val_loss /= len(val_dataloader)
        val_acc /= len(val_dataloader)

        # Save history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

        # Checkpointing
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(model_save_path, model_name)
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved new best model to {save_path}")

        # Step the scheduler
        if scheduler:
            scheduler.step()

    return history
