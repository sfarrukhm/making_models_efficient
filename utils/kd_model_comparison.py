import os
import time
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score

def evaluate_accuracy(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating Accuracy"):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return accuracy_score(all_labels, all_preds)

def measure_latency(model, input_size=(1, 3, 100, 100), device='cuda', num_runs=100):
    model.eval()
    dummy_input = torch.randn(input_size).to(device)
    model.to(device)

    # Warm-up
    for _ in range(10):
        _ = model(dummy_input)

    torch.cuda.synchronize() if device == 'cuda' else None
    start_time = time.time()

    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)

    torch.cuda.synchronize() if device == 'cuda' else None
    end_time = time.time()

    return ((end_time - start_time) / num_runs) * 1000  # ms per image

def get_model_size(model_path):
    size_bytes = os.path.getsize(model_path)
    return size_bytes / (1024 ** 2)  # Convert to MB
def count_params(model):
    return sum(p.numel() for p in model.parameters())
def generate_report(teacher_model, student_model, test_loader, device,
                    teacher_path, student_path):
    print("\n🚀 Running Knowledge Distillation Evaluation...\n")

    # Accuracy
    teacher_acc = evaluate_accuracy(teacher_model, test_loader, device)
    student_acc = evaluate_accuracy(student_model, test_loader, device)

    # Latency
    teacher_latency = measure_latency(teacher_model, device=device)
    student_latency = measure_latency(student_model, device=device)

    # Size
    teacher_size = get_model_size(teacher_path)
    student_size = get_model_size(student_path)

    # Derived metrics
    accuracy_gap = teacher_acc - student_acc
    speedup = teacher_latency / student_latency
    size_reduction = teacher_size / student_size


    teacher_params = count_params(teacher_model)
    student_params = count_params(student_model)
    param_reduction = ((teacher_params - student_params) / teacher_params) * 100

    # Report
    print("\n📊 KD Performance Report:")
    print(f"{'Metric':<25}{'Teacher':<15}{'Student':<15}{'Difference/Ratio'}")
    print("-" * 70)
    print(f"{'Accuracy (%)':<25}{teacher_acc*100:<15.2f}{student_acc*100:<15.2f}{(teacher_acc - student_acc)*100:.2f} ↓")
    print(f"{'Latency (ms)':<25}{teacher_latency*1000:<15.2f}{student_latency*1000:<15.2f}{(teacher_latency - student_latency)*1000:.2f} ms ↓")
    print(f"{'Speedup':<25}{'-':<15}{'-':<15}{teacher_latency / student_latency:.2f}× ↑")
    print(f"{'Model Size (MB)':<25}{teacher_size:<15.2f}{student_size:<15.2f}{teacher_size/student_size:.2f}× ↓")
    print(f"{'#Params (Millions)':<25}{teacher_params/1e6:<15.2f}{student_params/1e6:<15.2f}{(teacher_params - student_params)/1e6:.2f}M ↓")
    print(f"{'Param Reduction (%)':<25}{'-':<15}{'-':<15}{param_reduction:.2f}% ↓")
    print("-" * 70)


