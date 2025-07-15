import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from utils.model_utils import save_model

def train_model(model: torch.nn.Module,
                train_dataloader: DataLoader,
                val_dataloader: DataLoader,
                optimizer: torch.optim.Optimizer,
                loss_fn: torch.nn.Module,
                accuracy_fn,
                epochs: int,
                device: torch.device,
                model_save_path,
                model_name,
                scheduler=None,
                ):
    

  # Set random seeds for reproducibility
  torch.manual_seed(42)
  torch.cuda.manual_seed(42)

  # Lists to store losses and accuracies for training and validation
  train_losses, train_accuracies = [], []
  val_losses, val_accuracies = [], []
  best_val_epoch = 0


  for epoch in range(epochs):
    ### Training Phase ###
    train_loss, train_acc = 0, 0
    model.train()

    for features, labels in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}\t"):
      features, labels = features.to(device), labels.to(device)

      # Forward pass
      labels_preds = model(features)

      # Calculate loss
      loss = loss_fn(labels_preds, labels)
      train_loss += loss.item()

      # Calculate accuracy
      train_acc += accuracy_fn(labels_preds.argmax(dim=1), labels).item()

      # Backward pass and optimization
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

    # Average loss and accuracy over the entire training set
    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    ### Validation Phase ###
    val_loss, val_acc = 0, 0
    model.eval()

    with torch.inference_mode():
      for features, labels in tqdm(val_dataloader, desc="Validation", leave=False):
        features, labels = features.to(device), labels.to(device)

        # Forward pass for validation
        labels_preds = model(features)

        # Calculate loss
        val_loss += loss_fn(labels_preds, labels).item()

        # Calculate validation accuracy
        val_acc += accuracy_fn(labels_preds.argmax(dim=1), labels).item()

    # Average loss and accuracy over the entire validation set
    val_loss /= len(val_dataloader)
    val_acc /= len(val_dataloader)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    if(val_acc>best_val_epoch):
        best_val_epoch = val_acc
        print("saving")
        save_model(model=model, model_path=model_save_path, model_name=model_name)
        
    # Print epoch results
    print(f"[Loss: {train_loss:.4f}, Accuracy: {train_acc*100:.2f}% | "
          f"Val_Loss: {val_loss:.4f}, Val_Accuracy: {val_acc*100:.2f}%]")
    print("—" * 73)

    # Step the learning rate scheduler if it is provided
    if scheduler is not None:
      scheduler.step()
    
  # Store the history of losses and accuracies for plotting or analysis later
  history = {
    "train_losses": train_losses,
    "train_accuracies": train_accuracies,
    "val_losses": val_losses,
    "val_accuracies": val_accuracies
  }

  return history

from torch.utils.data import DataLoader
from tqdm import tqdm
import torch

def evaluate_model(model: torch.nn.Module,
                   dataloader: DataLoader,
                   loss_fn,
                   device,
                   accuracy_fn=None):  # Optional argument

    total_loss = 0.0
    total_accuracy = 0.0
    total_samples = 0

    model.eval()
    with torch.inference_mode():
        for features, labels in tqdm(dataloader, desc="Evaluation"):
            features, labels = features.to(device), labels.to(device)

            outputs = model(features)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item() * features.size(0)  # sum up batch loss

            if accuracy_fn:
                total_accuracy += accuracy_fn(outputs, labels) * features.size(0)
            else:
                preds = outputs.argmax(dim=1)
                total_accuracy += (preds == labels).sum().item()

            total_samples += features.size(0)

    avg_loss = total_loss / total_samples
    avg_acc = total_accuracy / total_samples

    return avg_loss, avg_acc


def make_prediction(model: torch.nn.Module, data_loader: DataLoader, device):
  model.eval()
  pred_labels = []
  true_labels = []

  with torch.inference_mode():
    for features, labels in data_loader:
      features = features.to(device)
      y_pred = model(features).argmax(dim=1)

      pred_labels.append(y_pred)
      true_labels.append(labels)

  pred_labels = torch.cat(pred_labels)
  true_labels = torch.cat(true_labels)

  return pred_labels, true_labels
