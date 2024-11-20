from __future__ import annotations

import math
import os
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    Optional,
    Tuple,
    Type,
    TypeVar,
    cast,
)

import torch
import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from accelerate.utils import send_to_device
from torch import Tensor, nn
from torch.utils.data import DataLoader
from transformers import PreTrainedModel

if TYPE_CHECKING:
    from .config import TrainConfig

T = TypeVar("T")


#  Constant
#  Cosine Annealing with Warmup
#  Cosine Annealing with Warmup / Restarts
#  No default values specified so the type-checker can verify we don't forget any arguments.
def get_lr_scheduler(
    scheduler_name: str,
    optimizer: optim.Optimizer,
    training_steps: int,
    lr: float,
    warm_up_steps: int,
    decay_steps: int,
    lr_end: float,
    num_cycles: int,
) -> lr_scheduler.LRScheduler:
    """
    Loosely based on this, seemed simpler write this than import
    transformers: https://huggingface.co/docs/transformers/main_classes/optimizer_schedules

    Args:
        scheduler_name (str): Name of the scheduler to use, one of "constant", "cosineannealing", "cosineannealingwarmrestarts"
        optimizer (optim.Optimizer): Optimizer to use
        training_steps (int): Total number of training steps
        warm_up_steps (int, optional): Number of linear warm up steps. Defaults to 0.
        decay_steps (int, optional): Number of linear decay steps to 0. Defaults to 0.
        num_cycles (int, optional): Number of cycles for cosine annealing with warm restarts. Defaults to 1.
        lr_end (float, optional): Final learning rate multiplier before decay. Defaults to 0.0.
    """
    if scheduler_name.lower() not in {
        "constant",
        "cosineannealing",
        "cosineannealingwarmrestarts",
    }:
        raise ValueError(
            "`scheduler_name` must be one of `constant`, `cosineannealing`,"
            "and `cosineannealingwarmrestarts`. "
            f"Given: {scheduler_name.lower()}."
        )
    base_scheduler_steps = training_steps - warm_up_steps - decay_steps
    norm_scheduler_name = scheduler_name.lower()
    main_scheduler = _get_main_lr_scheduler(
        norm_scheduler_name,
        optimizer,
        steps=base_scheduler_steps,
        lr_end=lr_end,
        num_cycles=num_cycles,
    )
    if norm_scheduler_name == "constant":
        # constant scheduler ignores lr_end, so decay needs to start at lr
        lr_end = lr
    schedulers: list[lr_scheduler.LRScheduler] = []
    milestones: list[int] = []
    if warm_up_steps > 0:
        schedulers.append(
            lr_scheduler.LinearLR(
                optimizer,
                start_factor=1 / warm_up_steps,
                end_factor=1.0,
                total_iters=warm_up_steps - 1,
            ),
        )
        milestones.append(warm_up_steps)
    schedulers.append(main_scheduler)
    if decay_steps > 0:
        if lr_end == 0.0:
            raise ValueError(
                "Cannot have decay_steps with lr_end=0.0, this would decay from 0 to 0 and be a waste."
            )
        schedulers.append(
            lr_scheduler.LinearLR(
                optimizer,
                start_factor=lr_end / lr,
                end_factor=0.0,
                total_iters=decay_steps,
            ),
        )
        milestones.append(training_steps - decay_steps)
    return lr_scheduler.SequentialLR(
        schedulers=schedulers,
        optimizer=optimizer,
        milestones=milestones,
    )


def standard_hook(
    module: nn.Module,
    _,
    outputs,
    module_to_name: Dict[nn.Module, str],
    hidden_dict: Dict[str, Tensor],
):
    # Maybe unpack tuple outputs
    if isinstance(outputs, tuple):
        outputs = outputs[0]

    name = module_to_name[module]
    hidden_dict[name] = outputs.flatten(0, 1)


def _get_main_lr_scheduler(
    scheduler_name: str,
    optimizer: optim.Optimizer,
    steps: int,
    lr_end: float,
    num_cycles: int,
) -> lr_scheduler.LRScheduler:
    if scheduler_name == "constant":
        return lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda steps: 1.0)
    elif scheduler_name == "cosineannealing":
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=lr_end)  # type: ignore
    elif scheduler_name == "cosineannealingwarmrestarts":
        return lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=steps // num_cycles, eta_min=lr_end  # type: ignore
        )
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")


class L1Scheduler:

    def __init__(
        self,
        l1_warmup_steps: float,
        total_steps: int,
        final_l1_coefficient: float,
    ):

        self.l1_warmup_steps = l1_warmup_steps
        # assume using warm-up
        if self.l1_warmup_steps != 0:
            self.current_l1_coefficient = 0.0
        else:
            self.current_l1_coefficient = final_l1_coefficient

        self.final_l1_coefficient = final_l1_coefficient

        self.current_step = 0
        self.total_steps = total_steps
        assert isinstance(self.final_l1_coefficient, float | int)

    def __repr__(self) -> str:
        return (
            f"L1Scheduler(final_l1_value={self.final_l1_coefficient}, "
            f"l1_warmup_steps={self.l1_warmup_steps}, "
            f"total_steps={self.total_steps})"
        )

    def step(self):
        """
        Updates the l1 coefficient of the sparse autoencoder.
        """
        step = self.current_step
        if step < self.l1_warmup_steps:
            self.current_l1_coefficient = self.final_l1_coefficient * (
                (1 + step) / self.l1_warmup_steps
            )  # type: ignore
        else:
            self.current_l1_coefficient = self.final_l1_coefficient  # type: ignore

        self.current_step += 1

    def state_dict(self):
        """State dict for serializing as part of an SAETrainContext."""
        return {
            "l1_warmup_steps": self.l1_warmup_steps,
            "total_steps": self.total_steps,
            "current_l1_coefficient": self.current_l1_coefficient,
            "final_l1_coefficient": self.final_l1_coefficient,
            "current_step": self.current_step,
        }

    def load_state_dict(self, state_dict: dict[str, Any]):
        """Loads all state apart from attached SAE."""
        for k in state_dict:
            setattr(self, k, state_dict[k])


class CycleIterator:
    """An iterator that cycles through an iterable indefinitely.

    Example:
        >>> iterator = CycleIterator([1, 2, 3])
        >>> [next(iterator) for _ in range(5)]
        [1, 2, 3, 1, 2]

    Note:
        Unlike ``itertools.cycle``, this iterator does not cache the values of the iterable.
    """

    def __init__(self, iterable: Iterable) -> None:
        self.iterable = iterable
        self.epoch = 0
        self._iterator = None

    def __next__(self) -> Any:
        if self._iterator is None:
            self._iterator = iter(self.iterable)
        try:
            return next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.iterable)
            self.epoch += 1
            return next(self._iterator)

    def __iter__(self) -> "CycleIterator":
        return self


class Norm1Normalizer(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super(Norm1Normalizer, self).__init__()
        self.eps = eps

    def forward(self, x):
        # Compute L2 norms and mean squared norm for current batch
        l2_norms = torch.norm(x, p=2, dim=1)

        # Normalize activations
        normalized_activations = x / (l2_norms.unsqueeze(1) + self.eps)
        return normalized_activations


class ExpectedNorm1Normalizer(nn.Module):
    def __init__(self, eta: float = 0.05, epsilon=1e-8):
        super(ExpectedNorm1Normalizer, self).__init__()
        self.eta = eta
        self.epsilon = epsilon
        self.running_mean_sqaured_norm = None
        self.dist_avail = dist.is_available() and dist.is_initialized()

    def forward(self, activations):
        # activations: Tensor of shape (batch_size, feature_size)

        squared_norms = torch.sum(activations**2, dim=-1)  # Shape: (batch_size,)
        mean_squared_norm = torch.mean(squared_norms)
        if self.dist_avail:
            mean_squared_norm = dist.all_reduce(mean_squared_norm)
            mean_squared_norm /= dist.get_world_size()

        if self.running_mean_sqaured_norm is None:
            self.running_mean_sqaured_norm = mean_squared_norm
        self.running_mean_sqaured_norm = (
            self.running_mean_sqaured_norm * (1 - self.eta)
            + mean_squared_norm * self.eta
        )

        # Step 3: Normalize the activations
        scaling_factor = torch.sqrt(mean_squared_norm + self.epsilon)
        normalized_activations = activations / scaling_factor

        return normalized_activations


def assert_type(typ: Type[T], obj: Any) -> T:
    """Assert that an object is of a given type at runtime and return it."""
    if not isinstance(obj, typ):
        raise TypeError(f"Expected {typ.__name__}, got {type(obj).__name__}")

    return cast(typ, obj)


@torch.no_grad()
def geometric_median(points: Tensor, max_iter: int = 100, tol: float = 1e-5):
    """Compute the geometric median `points`. Used for initializing decoder bias."""
    # Initialize our guess as the mean of the points
    guess = points.mean(dim=0)
    prev = torch.zeros_like(guess)

    # Weights for iteratively reweighted least squares
    weights = torch.ones(len(points), device=points.device)

    for _ in range(max_iter):
        prev = guess

        # Compute the weights
        weights = 1 / torch.norm(points - guess, dim=1)

        # Normalize the weights
        weights /= weights.sum()

        # Compute the new geometric median
        guess = (weights.unsqueeze(1) * points).sum(dim=0)

        # Early stopping condition
        if torch.norm(guess - prev) < tol:
            break

    return guess


def get_layer_list(model: PreTrainedModel) -> tuple[str, nn.ModuleList]:
    """Get the list of layers to train SAEs on."""
    N = assert_type(int, model.config.num_hidden_layers)
    candidates = [
        (name, mod)
        for (name, mod) in model.named_modules()
        if isinstance(mod, nn.ModuleList) and len(mod) == N
    ]
    assert len(candidates) == 1, "Could not find the list of layers."

    return candidates[0]


@torch.inference_mode()
def resolve_widths(
    cfg: TrainConfig,
    model: PreTrainedModel,
    module_names: list[str],
    dim: int = -1,
    dl: DataLoader | None = None,
) -> dict[str, int]:
    """Find number of output dimensions for the specified modules."""
    module_to_name = {model.get_submodule(name): name for name in module_names}
    hidden_dict: Dict[str, Tensor] = {}

    hook = partial(
        cfg.hook,
        module_to_name=module_to_name,
        hidden_dict=hidden_dict,
    )

    handles = [mod.register_forward_hook(hook) for mod in module_to_name]
    if dl is None:
        dummy = model.dummy_inputs
    else:
        dummy = next(iter(dl))
    dummy = send_to_device(dummy, model.device)
    try:
        model(**dummy)
    finally:
        for handle in handles:
            handle.remove()

    shapes = {name: hidden.shape[dim] for name, hidden in hidden_dict.items()}
    return shapes


# Fallback implementation of SAE decoder
def eager_decode(top_indices: Tensor, top_acts: Tensor, W_dec: Tensor):
    buf = top_acts.new_zeros(top_acts.shape[:-1] + (W_dec.shape[-1],))
    acts = buf.scatter_(dim=-1, index=top_indices, src=top_acts)
    return acts @ W_dec.mT


# Triton implementation of SAE decoder
def triton_decode(top_indices: Tensor, top_acts: Tensor, W_dec: Tensor):
    return TritonDecoder.apply(top_indices, top_acts, W_dec)


try:
    from .kernels import TritonDecoder
except ImportError:
    decoder_impl = eager_decode
    print("Triton not installed, using eager implementation of SAE decoder.")
else:
    if os.environ.get("SAE_DISABLE_TRITON") == "1":
        print("Triton disabled, using eager implementation of SAE decoder.")
        decoder_impl = eager_decode
    else:
        decoder_impl = triton_decode


def equiangular_init(
    N: int,
    M: int,
    max_iters: int = 1000,
    lr: float = 1e-2,
    loss_thr: float = 1e-6,
    device: str | torch.device ="cpu",
) -> Tensor:
    """Initialize N equiangular unit vectors in M-dimensional space."""

    vectors = torch.randn(N, M, requires_grad=False, device=device)
    vectors = vectors / vectors.norm(dim=1, keepdim=True)
    vectors.requires_grad = True

    # Desired cosine similarity
    desired_cosine = math.cos(2 * torch.pi / N)

    optimizer = optim.Adam([vectors], lr=lr)

    for _ in range(max_iters):
        optimizer.zero_grad()

        # Compute cosine similarities between all pairs
        cosine_matrix = torch.mm(vectors, vectors.t())
        # Subtract identity to exclude self-similarities
        cosine_matrix = cosine_matrix - torch.eye(N, device=device)

        # Compute penalties
        penalties = torch.clamp(cosine_matrix - desired_cosine, min=0)
        loss = penalties.sum()

        loss.backward()
        optimizer.step()

        # Re-normalize vectors to enforce unit norm
        vectors.data = vectors.data / vectors.data.norm(dim=1, keepdim=True)

        if loss.item() < loss_thr:
            break

    return vectors.detach()
