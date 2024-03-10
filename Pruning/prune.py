import torch
import torch.nn as nn

"""
@inproceedings{
  park2020lookahead,
  title={Lookahead: A Far-sighted Alternative of Magnitude-based Pruning},
  author={Sejun Park and Jaeho Lee and Sangwoo Mo and Jinwoo Shin},
  booktitle={International Conference on Learning Representations},
  year={2020},
  url={https://openreview.net/forum?id=ryl3ygHYDB}
}
"""

def MP(weights, masks, prune_ratios):
    """Magnitude pruning"""

    def score_func(weights, layer):
        score = torch.abs(weights[layer])
        return score

    new_masks = _score_based_pruning(weights, masks, prune_ratios, score_func)
    return new_masks

def RP(weights, masks, prune_ratios):
    """Random pruning"""

    def score_func(weights, layer):
        # Explicitly ensure `weights[layer]` results in a tensor
        assert isinstance(weights[layer], torch.Tensor), f"weights[{layer}] must be a tensor, got {type(weights[layer])}"
        score = torch.abs(torch.randn(weights[layer].size()))
        return score

    new_masks = _score_based_pruning(weights, masks, prune_ratios, score_func)
    return new_masks

def _score_based_pruning(weights, masks, prune_ratios, score_func, mode='base', split=1):
    """Abstract function for score-based pruning"""
    if mode == 'base':
        layers = list(range(len(prune_ratios)))  # Convert to list for reusability
    elif mode == 'forward':
        layers = list(range(len(prune_ratios)))
    elif mode == 'backward':
        layers = list(reversed(range(len(prune_ratios))))
    else:
        raise ValueError('Unknown pruning method')

    new_masks = []
    for s in range(split):
        new_masks = []
        for layer in layers:
            score = score_func(weights, layer)  # score for the current layer
            # Debugging check
            assert isinstance(score, torch.Tensor), f"Score for layer {layer} is not a tensor after score_func."
            prune_rate = prune_ratios[layer] / split / (1 - prune_ratios[layer] * s / split)
            new_mask = _score_based_mask(score, masks[layer], prune_rate)
            new_masks.append(new_mask)
            # Apply mask updates according to the mode
            if mode in ['forward', 'backward']:
                weights[layer] *= new_mask
                masks[layer] *= new_mask
        # Base mode implies updating all layers' weights and masks
        if mode == 'base':
            for layer, new_mask in zip(layers, new_masks):
                weights[layer] *= new_mask
                masks[layer] *= new_mask

    if mode == 'backward':
        new_masks.reverse()

    return new_masks



def _score_based_mask(score, mask, prune_ratio):
    """Generate a new mask for the current layer based on pruning ratio and score"""
    assert (prune_ratio >= 0) and (prune_ratio <= 1), "Prune ratio must be between 0 and 1."

    # Flatten the score and mask to ensure compatibility for operations.
    score_flat = score.view(-1)
    mask_flat = mask.view(-1)
    # Previous weights pruning mask
    score_flat[mask_flat <= 0] = float('-inf')

    surv_ratio = 1 - prune_ratio
    num_surv = int(torch.sum(mask_flat).item() * surv_ratio)
    
    _, idx = score_flat.sort(descending=True)
    cutoff_idx = idx[num_surv] if num_surv < len(idx) else idx[-1]

    new_mask_flat = torch.ones_like(score_flat)
    new_mask_flat[idx[num_surv:]] = 0  # Prune the lowest scored weights

    new_mask = new_mask_flat.view(mask.shape)  # Reshape new mask to original

    return new_mask