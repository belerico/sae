from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Protocol, Sequence, Tuple, TypeVar

import torch
from torch import Tensor, nn
from torch.nn import Module

R = TypeVar("R")


class HookWithKwargs(Protocol):
    def __call__(
        self,
        module: Module,
        inputs: Tuple[Any, ...],
        outputs: Any,
        *,
        module_to_name: Dict[Module, str],
        hidden_dict: Dict[str, Tensor],
    ) -> None:
        ...


def forward_hook_wrapper(
    hook: Callable[..., R] | None, *hook_args: Any, **hook_kwargs: Any
) -> Callable[[Module, Tuple[Any, ...], Any], R]:
    """
    Return a function that matches the PyTorch forward hook signature:
      (module, inputs, outputs) -> R
    but calls the given 'hook' with additional arguments.

    Args:
        hook (Callable): The hook to call.
        *hook_args (Any): Additional positional arguments to pass to the hook
        **hook_kwargs (Any): Additional keyword arguments to pass to the hook

    Returns:
        Callable[[Module, Tuple[Any, ...], Any], R]: The wrapper function.
    """

    def wrapper(module: Module, inputs: Tuple[Any, ...], outputs: Any) -> R:
        # Forward the standard hook signature + any extras
        if hook is None:
            raise ValueError("No hook provided")
        return hook(module, inputs, outputs, *hook_args, **hook_kwargs)

    return wrapper


def standard_hook(
    module: nn.Module,
    inputs: Tuple[Any, ...],
    outputs: Any,
    *,
    module_to_name: Dict[nn.Module, str],
    hidden_dict: Dict[str, Tensor],
):
    # Maybe unpack tuple outputs
    if isinstance(outputs, tuple):
        outputs = outputs[0]

    name = module_to_name[module]
    hidden_dict[name] = outputs.flatten(0, 1)


def from_tokens(examples: Sequence[Mapping[str, Any]]) -> Mapping[str, Tensor]:
    return {
        "input_ids": torch.stack(
            list(torch.tensor(example["tokens"]) for example in examples), dim=0
        )
    }
