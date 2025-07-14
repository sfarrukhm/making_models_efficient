import os
import torch

def save_model(model, model_path, model_name):
    """
    Save a PyTorch model's state_dict to the specified path.
    """
    os.makedirs(model_path, exist_ok=True)
    model_save_path = os.path.join(model_path, model_name)
    torch.save(model.state_dict(), model_save_path)
    print(f"✅ Model saved to: '{model_save_path}'")

def load_model(model_class, checkpoint_path, model_name, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Load a saved PyTorch model.
    """
    model_path = os.path.join(checkpoint_path, model_name)
    model = model_class()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model
