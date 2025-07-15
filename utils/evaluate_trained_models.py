import torch
import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score

def evaluate_accuracy(model, dataloader, device):
    """
    Calculates top-1 accuracy of a given model on the provided dataloader.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating Accuracy"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    return acc

def measure_latency(model, input_size=(1, 3, 150, 150), device='cuda', num_runs=100):
    """
    Measures average inference latency per image (in milliseconds).
    """
    model = model.to(device)
    model.eval()
    dummy_input = torch.randn(input_size).to(device)

    # Warm-up
    for _ in range(10):
        _ = model(dummy_input)

    # Measure
    torch.cuda.synchronize() if device == 'cuda' else None
    start = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
    torch.cuda.synchronize() if device == 'cuda' else None
    end = time.time()

    avg_latency_ms = (end - start) / num_runs * 1000  # milliseconds
    return avg_latency_ms
