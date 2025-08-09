# Fine-Grained Pruning of ResNet-18

## Overview

This project implements **magnitude-based fine-grained pruning** on a ResNet-18 model to reduce its parameter count while preserving (and in some cases improving) accuracy. The pruning selectively zeroes out less important weights in convolutional, downsampling, and fully connected layers.

## Motivation

Model pruning is a powerful technique for:

- **Reducing model size** for deployment on resource-constrained devices.

- **Improving inference speed** by decreasing computational load.

- **Potentially enhancing generalization** by removing redundant parameters.

Fine-grained pruning operates at the **individual weight level**, as opposed to structured pruning, which removes entire filters or channels.

## Methodology

- **Base model:** ResNet-18 pretrained on the target dataset.

- **Pruning approach:** Magnitude-based fine-grained pruning — weights with the smallest absolute values are set to zero according to layer-specific sparsity ratios.

- **Target layers:** All convolutional layers, downsampling layers, and the fully connected layer.

- **Sparsity setting:** Ratios are defined per layer, allowing flexible control over pruning intensity.

- **Training procedure:** The pruned model is retrained (fine-tuned) to recover performance lost during pruning.

- **Evaluation metric:** Classification accuracy on the validation/test set.

---

## Comparative Stats

|                | MACs (M) | Params (M) | Sparsity (%) | Size (MB) | Accuracy (%) |
| -------------- | -------- | ---------- | ------------ | --------- | ------------ |
| Before Pruning | 1816.05  | 11.18      | 0.00         | 42.72     | 90.23        |
| After Pruning  | 1816.05  | 11.18      | 79.93        | 42.72     | 91.47        |

---

## Interpretation

- **Accuracy Gain** — The small bump in accuracy (+1.24%) suggests that pruning helped the model generalize better by removing noise or overfitting parameters.

- **No Size/MACs Reduction** — Without converting to sparse tensor formats or hardware-aware pruning, the computational cost and storage footprint are unaffected.

---

## Takeaway

Fine-grained pruning is a **lightweight regularization tool** that can improve accuracy and model sparsity without retraining from scratch.  
However, for actual runtime speedups or storage savings, **structured pruning or sparse-aware deployment** is required.


