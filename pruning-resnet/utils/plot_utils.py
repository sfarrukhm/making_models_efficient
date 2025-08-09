import math
import matplotlib.pyplot as plt
import numpy as np

# Histogram of weight distribution
def plot_weight_distribution(model, bins=256, count_nonzero_only=False):
    # Get all params with dim > 1
    weight_params = [(name, p) for name, p in model.named_parameters() if p.dim() > 1]
    
    # Make grid big enough
    cols = 3
    rows = (len(weight_params) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*3))
    axes = axes.ravel()

    for idx, (name, param) in enumerate(weight_params):
        ax = axes[idx]
        param_cpu = param.detach().view(-1).cpu()
        if count_nonzero_only:
            param_cpu = param_cpu[param_cpu != 0]
        ax.hist(param_cpu, bins=bins, density=True, color='blue', alpha=0.5)
        ax.set_xlabel(name, fontsize=8)
        ax.set_ylabel('density')

    # Hide unused subplots
    for ax in axes[len(weight_params):]:
        ax.axis('off')

    fig.suptitle('Histogram of Weights', fontsize=16)
    fig.tight_layout()
    fig.subplots_adjust(top=0.92)
    plt.show()

def plot_num_parameters_distribution(model):
    num_parameters = {name: param.numel() for name, param in model.named_parameters() if param.dim() > 1}
    
    # Dynamic figure width: 0.5 inch per layer, minimum 12 inches
    fig_width = max(12, 0.5 * len(num_parameters))
    fig_height = 6
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    plt.grid(axis='y')
    plt.bar(list(num_parameters.keys()), list(num_parameters.values()))
    plt.title('#Parameter Distribution')
    plt.ylabel('Number of Parameters')
    plt.xticks(rotation=60, ha='right')  # rotate & align right for clarity
    plt.tight_layout()
    plt.show()


def plot_sensitivity_scan(sparsities, accuracies, dense_model_accuracy, model):
    # Collect names of all parameters with dim > 1 (weights of conv, fc, etc.)
    layer_names = [name for name, param in model.named_parameters() if param.dim() > 1]
    n_plots = len(layer_names)
    
    # Set layout: 3 columns, enough rows to fit all plots
    n_cols = 3
    n_rows = math.ceil(n_plots / n_cols)
    
    # Adjust figure size dynamically (5x4 inches per subplot is a good start)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = axes.ravel()  # Flatten axes array for easy indexing
    
    # Calculate lower bound line for plotting
    lower_bound_accuracy = 100 - (100 - dense_model_accuracy) * 1.5
    
    for idx, name in enumerate(layer_names):
        ax = axes[idx]
        ax.plot(sparsities, accuracies[idx], label="accuracy after pruning")
        ax.plot(sparsities, [lower_bound_accuracy]*len(sparsities), 
                label=f'{lower_bound_accuracy / dense_model_accuracy * 100:.0f}% of dense accuracy')
        ax.set_xticks(np.arange(sparsities[0], sparsities[-1]+0.01, step=0.1))
        ax.set_ylim(lower_bound_accuracy - 1, dense_model_accuracy+2)  
        ax.set_title(name)
        ax.set_xlabel('Sparsity')
        ax.set_ylabel('Top-1 Accuracy')
        ax.legend()
        ax.grid(axis='x')
    
    # Hide unused axes (if any)
    for i in range(n_plots, n_rows * n_cols):
        fig.delaxes(axes[i])
    
    fig.suptitle('Sensitivity Curves: Validation Accuracy vs. Pruning Sparsity', fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()