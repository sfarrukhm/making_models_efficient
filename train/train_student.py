import torch
import torch.nn.functional as F

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
def train_student_with_distillation(
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
