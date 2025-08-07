import torch
import matplotlib.pyplot as plt
from utils.model_utils import get_sparsity
# Fine-grained pruning function
def fine_grained_prune(tensor: torch.Tensor, sparsity: float) -> torch.Tensor:
    sparsity = min(max(0.0, sparsity), 1.0)
    if sparsity == 1.0:
        tensor.zero_()
        return torch.zeros_like(tensor)
    elif sparsity == 0.0:
        return torch.ones_like(tensor)

    num_elements = tensor.numel()
    num_zeros = round(sparsity * num_elements)

    importance = torch.abs(tensor)
    threshold = torch.kthvalue(importance.view(-1), k=num_zeros)[0]
    mask = importance > threshold
    tensor.mul_(mask)

    return mask

# Histogram of weight distribution
def plot_weight_distribution(model, bins=256, count_nonzero_only=False):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 3, figsize=(10, 6))
    axes = axes.ravel()
    plot_index = 0
    for name, param in model.named_parameters():
        if param.dim() > 1:
            ax = axes[plot_index]
            param_cpu = param.detach().view(-1).cpu()
            if count_nonzero_only:
                param_cpu = param_cpu[param_cpu != 0]
            ax.hist(param_cpu, bins=bins, density=True, color='blue', alpha=0.5)
            ax.set_xlabel(name)
            ax.set_ylabel('density')
            plot_index += 1
            if plot_index >= len(axes):  # avoid index error
                break
    fig.suptitle('Histogram of Weights')
    fig.tight_layout()
    fig.subplots_adjust(top=0.925)
    plt.show()



def test_fine_grained_prune(
    test_tensor=torch.tensor([[-0.46, -0.40, 0.39, 0.19, 0.37],
                              [0.00, 0.40, 0.17, -0.15, 0.16],
                              [-0.20, -0.23, 0.36, 0.25, 0.03],
                              [0.24, 0.41, 0.07, 0.13, -0.15],
                              [0.48, -0.09, -0.36, 0.12, 0.45]]),
    test_mask=torch.tensor([[True, True, False, False, False],
                            [False, True, False, False, False],
                            [False, False, False, False, False],
                            [False, True, False, False, False],
                            [True, False, False, False, True]]),
    target_sparsity=0.75, target_nonzeros=None):
    
    def plot_matrix(tensor, ax, title):
        ax.imshow(tensor.cpu().numpy() == 0, vmin=0, vmax=1, cmap='tab20c')
        ax.set_title(title)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        for i in range(tensor.shape[1]):
            for j in range(tensor.shape[0]):
                text = ax.text(j, i, f'{tensor[i, j].item():.2f}',
                               ha="center", va="center", color="k")

    test_tensor = test_tensor.clone()
    fig, axes = plt.subplots(1, 2, figsize=(6, 10))
    ax_left, ax_right = axes.ravel()
    plot_matrix(test_tensor, ax_left, 'dense tensor')

    sparsity_before_pruning = get_sparsity(test_tensor)
    mask = fine_grained_prune(test_tensor, target_sparsity)
    sparsity_after_pruning = get_sparsity(test_tensor)
    sparsity_of_mask = get_sparsity(mask)

    plot_matrix(test_tensor, ax_right, 'sparse tensor')
    fig.tight_layout()
    plt.show()

    print('* Test fine_grained_prune()')
    print(f'    target sparsity: {target_sparsity:.2f}')
    print(f'        sparsity before pruning: {sparsity_before_pruning:.2f}')
    print(f'        sparsity after pruning: {sparsity_after_pruning:.2f}')
    print(f'        sparsity of pruning mask: {sparsity_of_mask:.2f}')

    if target_nonzeros is None:
        if test_mask.equal(mask):
            print('* Test passed.')
        else:
            print('* Test failed.')
    else:
        if mask.count_nonzero() == target_nonzeros:
            print('* Test passed.')
        else:
            print('* Test failed.')
import torch
import numpy as np
from tqdm import tqdm

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
