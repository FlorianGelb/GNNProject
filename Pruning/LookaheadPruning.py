import torch
from Pruning.laprune import LAP  # Import LAP function


def calculate_sparsity(model):
    total_weights = 0
    total_zero_weights = 0
    for param in model.parameters():
        if param.dim() > 1:  # Only consider weights of layers, not biases
            total_weights += param.numel()
            total_zero_weights += torch.sum(param == 0).item()
    sparsity_level = total_zero_weights / total_weights
    return sparsity_level


def apply_lap(model, prune_ratios, bn_factors=None):
    weights = [param.data for param in model.parameters() if len(param.data.size()) > 1]
    masks = [torch.ones_like(weight) for weight in weights]

    # Assuming LAP function is correctly implemented and available
    new_masks = LAP(weights, masks, prune_ratios, bn_factors=bn_factors)

    # Apply new masks
    for param, mask in zip([param for param in model.parameters() if len(param.data.size()) > 1], new_masks):
        param.data.mul_(mask)