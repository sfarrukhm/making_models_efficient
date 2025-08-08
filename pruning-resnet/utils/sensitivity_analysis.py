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
def sensitivity_scan(model, dataloader, scan_step=0.1, scan_start=0.4, scan_end=1.0, verbose=True):
    """
    Performs a sensitivity analysis for each layer of the model by applying different sparsity levels 
    and evaluating how it affects accuracy.

    Args:
        model (torch.nn.Module): The neural network to prune and evaluate.
        dataloader (DataLoader): DataLoader for the evaluation dataset.
        scan_step (float): Step size between sparsity levels (default: 0.1).
        scan_start (float): Starting sparsity level (default: 0.4).
        scan_end (float): Ending sparsity level (default: 1.0).
        verbose (bool): Whether to print detailed scan results (default: True).

    Returns:
        tuple:
            - sparsities (np.ndarray): Array of sparsity levels tested.
            - accuracies (list of list): Layer-wise list of accuracies at each sparsity level.
    """

    # Create an array of sparsity levels to scan
    sparsities = np.arange(start=scan_start, stop=scan_end, step=scan_step)
    
    # Container to hold accuracy values for each layer and sparsity
    accuracies = []

    # Get list of all prunable parameters (weight matrices only)
    named_conv_weights = [
        (name, param) for (name, param) in model.named_parameters() if param.dim() > 1
    ]

    # Iterate over each prunable parameter (Conv/Linear layers)
    for i_layer, (name, param) in enumerate(named_conv_weights):
        # Clone the original parameter so we can restore it after each test
        param_clone = param.detach().clone()
        accuracy = []

        # Loop over each sparsity level for current layer
        for sparsity in tqdm(sparsities, desc=f'Scanning layer {i_layer}/{len(named_conv_weights)} - {name}'):
            # Apply fine-grained pruning to the parameter in-place
            fine_grained_prune(param.detach(), sparsity=sparsity)

            # Evaluate model accuracy on the validation/test set
            acc = evaluate(model, dataloader, verbose=False)
            accuracy.append(acc)

            # Optionally print real-time progress
            if verbose:
                print(f'\r    sparsity={sparsity:.2f}: accuracy={acc:.2f}%', end='')

            # Restore original unpruned weights for the next iteration
            param.copy_(param_clone)

        # After scanning one layer, print summary if verbose
        if verbose:
            sparsity_str = ", ".join(["{:.2f}".format(x) for x in sparsities])
            accuracy_str = ", ".join(["{:.2f}%".format(x) for x in accuracy])
            print(f'\r    sparsity=[{sparsity_str}]: accuracy=[{accuracy_str}]', end='')

        # Store accuracy curve for this layer
        accuracies.append(accuracy)

    return sparsities, accuracies

