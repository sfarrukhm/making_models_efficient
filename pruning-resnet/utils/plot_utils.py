import matplotlib.pyplot as plt
import math
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
    """
    Plots number of parameters for each prunable layer in the model.
    Useful for deciding sparsity levels.
    """
    num_parameters = {
        name: param.numel()
        for name, param in model.named_parameters()
        if param.dim() > 1  # Only conv/linear layers
    }

    plt.figure(figsize=(8, 6))
    plt.bar(num_parameters.keys(), num_parameters.values())
    plt.grid(axis='y')
    plt.title('#Parameter Distribution')
    plt.ylabel('Number of Parameters')
    plt.xticks(rotation=60)
    plt.tight_layout()
    plt.show()

def plot_sensitivity_scan(sparsities, accuracies, dense_model_accuracy):
    """
    Plots layer-wise sensitivity curves: sparsity vs. validation accuracy.
    """
    lower_bound_accuracy = 100 - (100 - dense_model_accuracy) * 1.5

    fig, axes = plt.subplots(3, math.ceil(len(accuracies) / 3), figsize=(15, 8))
    axes = axes.ravel()
    plot_index = 0

    for name, param in model.named_parameters():
        if param.dim() > 1:
            ax = axes[plot_index]

            # Accuracy curve for this layer
            ax.plot(sparsities, accuracies[plot_index], label='accuracy after pruning')
            # Horizontal line: acceptable accuracy drop
            ax.plot(sparsities, [lower_bound_accuracy] * len(sparsities),
                    label=f'{lower_bound_accuracy / dense_model_accuracy * 100:.0f}% of dense acc')

            ax.set_xticks(np.arange(0.4, 1.0, 0.1))
            ax.set_ylim(80, 95)
            ax.set_title(name)
            ax.set_xlabel('sparsity')
            ax.set_ylabel('top-1 accuracy')
            ax.legend()
            ax.grid(axis='x')

            plot_index += 1

    fig.suptitle('Sensitivity Curves: Validation Accuracy vs. Pruning Sparsity')
    fig.tight_layout()
    fig.subplots_adjust(top=0.925)
    plt.show()
  
def plot_sensitivity_scan(sparsities, accuracies, dense_model_accuracy):
    lower_bound_accuracy = 100 - (100 - dense_model_accuracy) * 1.5
    fig, axes = plt.subplots(3, int(math.ceil(len(accuracies) / 3)),figsize=(15,8))
    axes = axes.ravel()
    plot_index = 0
    for name, param in model.named_parameters():
        if param.dim() > 1:
            ax = axes[plot_index]
            curve = ax.plot(sparsities, accuracies[plot_index])
            line = ax.plot(sparsities, [lower_bound_accuracy] * len(sparsities))
            ax.set_xticks(np.arange(start=0.4, stop=1.0, step=0.1))
            ax.set_ylim(80, 95)
            ax.set_title(name)
            ax.set_xlabel('sparsity')
            ax.set_ylabel('top-1 accuracy')
            ax.legend([
                'accuracy after pruning',
                f'{lower_bound_accuracy / dense_model_accuracy * 100:.0f}% of dense model accuracy'
            ])
            ax.grid(axis='x')
            plot_index += 1
    fig.suptitle('Sensitivity Curves: Validation Accuracy vs. Pruning Sparsity')
    fig.tight_layout()
    fig.subplots_adjust(top=0.925)
    plt.show()
