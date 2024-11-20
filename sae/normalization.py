from functools import partial
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader


def estimate_norm_scaling_factor(
    data_loader: DataLoader,
    model: torch.nn.Module,
    max_tokens: int,
    hook: Callable[
        [nn.Module, Tuple[Any, ...], Any, Dict[nn.Module, str], Dict[str, Tensor]],
        Optional[Any],
    ],
    module_to_name: Dict[torch.nn.Module, str],
    target_norm: float = 1.0,
    device: str | torch.device = "cuda",
) -> Dict[str, float]:
    """
    Estimate the normalization factor for the given dataset.
    The normalization factor is the value that should be used to scale the logits of the model
    to have a norm of `target_norm` on average.

    Args:
        data_loader: The data loader for the dataset.
        model: The model to estimate the normalization factor for.
        max_tokens: The maximum number of tokens to process.
        hook: The hook to use to extract the hidden states.
        module_to_name: A dictionary mapping the modules to the names of the hidden states.
        target_norm: The target norm for the logits.

    Returns:
        A dictionary mapping the names of the hidden states to the normalization factors.
    """
    print("Estimating normalization factor")
    num_tokens = 0
    hidden_dict = {}
    mean_l2_per_hook = {name: [] for name in module_to_name.values()}
    for batch in data_loader:
        # Forward pass on the model to get the next batch of activations
        handles = [
            mod.register_forward_hook(
                partial(
                    hook,
                    module_to_name=module_to_name,
                    hidden_dict=hidden_dict,
                )
            )
            for mod in module_to_name.keys()
        ]
        try:
            with torch.no_grad():
                model(batch["input_ids"].to(device))
        finally:
            for handle in handles:
                handle.remove()
        for i, (name, hidden) in enumerate(hidden_dict.items()):
            if i == 0:
                num_tokens += hidden.size(0)
            mean_l2_per_hook[name].append(hidden.norm(p=2, dim=-1).mean().item())
        if num_tokens >= max_tokens:
            break
    return {
        name: target_norm / np.mean(values).item() for name, values in mean_l2_per_hook.items()
    }
