import torch
import torch.nn as nn
from torchprofile import profile_macs
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.prune_utils import fine_grained_prune
from utils.train_evaluate import evaluate


@torch.no_grad()
def sensitivity_scan(model, dataloader, scan_step=0.1, scan_start=0.4, scan_end=1.0):
    sparsities = np.arange(start=scan_start, stop=scan_end, step=scan_step)
    accuracies = []
    named_conv_weights = [(name, param) for (name, param) in model.named_parameters() if param.dim() > 1]

    for i_layer, (name, param) in enumerate(named_conv_weights):
        param_clone = param.detach().clone()
        layer_accuracies = []

        for sparsity in tqdm(sparsities, desc=f'Layer {i_layer+1}/{len(named_conv_weights)}: {name}'):
            fine_grained_prune(param.detach(), sparsity=sparsity)
            acc = evaluate(model, dataloader, verbose=False)
            param.copy_(param_clone)  # restore
            layer_accuracies.append(acc)

        accuracies.append(layer_accuracies)

    return sparsities, accuracies


