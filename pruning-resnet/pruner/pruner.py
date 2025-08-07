import torch

class FineGrainedPruner:
    """
    This class performs fine-grained pruning on a PyTorch model.
    Fine-grained pruning zeros out individual weights based on their magnitude.

    Attributes:
        masks (dict): Dictionary of binary masks (1: keep, 0: prune) for each pruned parameter.
    """

    def __init__(self, model, sparsity_dict):
        """
        Initializes the pruner by generating masks for the given model and target sparsities.

        Args:
            model (nn.Module): The PyTorch model to prune.
            sparsity_dict (dict): A dictionary where keys are parameter names and values are target sparsities (0 to 1).
        """
        self.masks = FineGrainedPruner.prune(model, sparsity_dict)

    @torch.no_grad()
    def apply(self, model):
        """
        Applies the stored masks to the model to enforce pruning (zeros out pruned weights).

        Args:
            model (nn.Module): The model on which the masks should be applied.
        """
        for name, param in model.named_parameters():
            if name in self.masks:
                param *= self.masks[name]  # Element-wise multiplication to zero out pruned weights

    @staticmethod
    @torch.no_grad()
    def prune(model, sparsity_dict):
        """
        Generates masks for each prunable parameter based on the target sparsity.

        Args:
            model (nn.Module): The model whose weights will be pruned.
            sparsity_dict (dict): Dictionary specifying the sparsity for each parameter.

        Returns:
            dict: A dictionary of binary masks (same shape as weights).
        """
        masks = dict()
        for name, param in model.named_parameters():
            if param.dim() > 1:  # Only prune weight matrices (e.g., Conv and Linear), skip biases
                masks[name] = fine_grained_prune(param, sparsity_dict[name])
        return masks
