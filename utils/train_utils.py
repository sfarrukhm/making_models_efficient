import torch
from tqdm.auto import tqdm
import os
import torch
import torch.nn.functional as F

### Student Training Setup
def train_student(
    student_model,
    teacher_model,
    train_dataloader,
    val_dataloader,
    optimizer,
    loss_fn,
    accuracy_fn,
    epochs,
    scheduler=None,
    device='cuda',
    alpha=0.5,
    temperature=4.0,
    model_save_path=None,
    model_name="student.pth"
):
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(epochs):
        print(f"\nEpoch [{epoch+1}/{epochs}]")

        # Training phase
        train_loss, train_acc = distillation_train_step(
            student_model, teacher_model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            accuracy_fn=accuracy_fn,
            device=device,
            alpha=alpha,
            temperature=temperature
        )

        # Validation phase
        val_loss, val_acc = evaluate_model(
            model=student_model,
            dataloader=val_dataloader,
            loss_fn=loss_fn,
            accuracy_fn=accuracy_fn,
            device=device
        )

        if scheduler: scheduler.step()

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        # Save best model if desired
        if model_save_path:
            save_model(student_model, model_save_path, model_name)

        # Log results
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

    return history
def distillation_train_step(
    student_model,
    teacher_model,
    dataloader,
    optimizer,
    loss_fn,
    accuracy_fn,
    device,
    alpha=0.5,
    temperature=4.0
):
    student_model.train()
    teacher_model.eval()

    total_loss = 0.0
    total_accuracy = 0.0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass (no grad for teacher)
        with torch.no_grad():
            teacher_logits = teacher_model(images)

        student_logits = student_model(images)

        # Soft targets
        soft_teacher_probs = F.log_softmax(teacher_logits / temperature, dim=1)
        soft_student_probs = F.log_softmax(student_logits / temperature, dim=1)

        # Distillation loss (KL divergence)
        soft_loss = F.kl_div(
            soft_student_probs, 
            soft_teacher_probs.exp(), 
            reduction='batchmean'
        ) * (temperature ** 2)

        # Hard label loss
        hard_loss = loss_fn(student_logits, labels)

        # Combined loss
        loss = alpha * soft_loss + (1 - alpha) * hard_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_accuracy += accuracy_fn(student_logits, labels).item()

    return total_loss / len(dataloader), total_accuracy / len(dataloader)


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
