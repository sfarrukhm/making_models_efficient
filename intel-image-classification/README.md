🧠 Knowledge Distillation for Intel Image Classification

This repo implements a knowledge distillation pipeline where a large VisionEagle-based teacher model guides a lightweight CNN student for image classification on the [Intel Image Classification Dataset](https://www.kaggle.com/datasets/puneet6060/intel-image-classification).

---

## 📦 Models

### 🧑‍🏫 Teacher Model

- Based on [Vision Eagle Attention (VEA)](https://arxiv.org/abs/2411.10564) with custom architectural tweaks.

- ResNet18 backbone with spatial attention modules.

- Trained from scratch on 100×100 resolution input.

### 👶 Student Model

- Custom lightweight CNN (~90K parameters).

- 3 conv blocks + global avg pooling + dropout.

- Designed for fast inference and minimal memory footprint.

- Trained using logits + labels from the teacher (KD).

```text
Conv → BN → ReLU → MaxPool → x3 → AdaptiveAvgPool → FC → Output
```

---

## 📊 KD Performance Report

| Metric                  | Teacher  | Student | Difference/Ratio |
| ----------------------- | -------- | ------- | ---------------- |
| **Accuracy (%)**        | 89.50    | 81.13   | 8.37 ↓           |
| **Latency (ms)**        | 23088.15 | 5794.44 | 17293.71 ↓       |
| **Speedup**             | -        | -       | 3.98× ↑          |
| **Model Size (MB)**     | 91.08    | 0.37    | 246.75× ↓        |
| **#Params (Millions)**  | 23.83    | 0.09    | 23.73M ↓         |
| **Param Reduction (%)** | -        | -       | 99.60% ↓         |

---

## 🔍 Predictions

### Teacher Predictions

<img src="file:///C:/Users/HP/Git/repos/snapshots/teacher_predictions.png" title="" alt="Teacher Predictions" width="792">



### Teacher's Confusion <img title="" src="file:///C:/Users/HP/Git/repos/snapshots/confusion_teacher.png" alt="Confusion" width="463">

### 

### Student's Predictions ![Predictions](C:\Users\HP\Git\repos\snapshots\student_predictions.png)

### Student's Confusion<img src="file:///C:/Users/HP/Git/repos/snapshots/confusion_student.png" title="" alt="Confusion" width="499">

---

## 🗂️ Repo Structure

```bash
.
├── models/                  # Teacher and student architectures
├── train.py                # CLI training script
├── evaluate.py             # Evaluation and reporting
├── utils/                  # Helpers: KD loss, metrics, loaders
├── checkpoints/            # Saved model weights
├── images/                 # Prediction snapshots & model diagrams
└── README.md
```




