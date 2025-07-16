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

    # Report
    print("\n📊 KD Performance Report:")
    print("=" * 40)
    print(f"{'Metric':<20} {'Teacher':<10} {'Student':<10}")
    print("-" * 40)
    print(f"{'Accuracy (%)':<20} {teacher_acc*100:>8.2f}   {student_acc*100:>8.2f}")
    print(f"{'Latency (ms)':<20} {teacher_latency:>8.2f}   {student_latency:>8.2f}")
    print(f"{'Size (MB)':<20} {teacher_size:>8.2f}   {student_size:>8.2f}")
    print("=" * 40)
    print(f"{'Accuracy Gap':<20}: {accuracy_gap:.4f}")
    print(f"{'Speedup':<20}: {speedup:.2f}x")
    print(f"{'Size Reduction':<20}: {size_reduction:.2f}x smaller")
    print("=" * 40)

