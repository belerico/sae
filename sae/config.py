from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

from simple_parsing import Serializable, list_field
from torch import Tensor, nn

from .utils import standard_hook


@dataclass
class SaeConfig(Serializable):
    """
    Configuration for training a sparse autoencoder on a language model.
    """

    expansion_factor: int = 32
    """Multiple of the input dimension to use as the SAE dimension."""

    normalize_decoder: bool = True
    """Normalize the decoder weights to have unit norm."""

    num_latents: int = 0
    """Number of latents to use. If 0, use `expansion_factor`."""

    k: int = 32
    """Number of nonzero features."""

    multi_topk: bool = False
    """Use Multi-TopK loss."""

    jumprelu: bool = False
    """Whether to use the JumpReLU or not"""

    jumprelu_init_threshold: float = 0.001
    """JumpReLU initial threshold"""

    jumprelu_bandwidth: float = 0.001
    """JumpReLU bandwidth"""

    jumprelu_target_l0: float | None = None
    """JumpReLU target L0"""

    init_enc_as_dec_transpose: bool = False
    """Whether to initialize the encoder matrix as the decoder transpose"""

    init_b_dec_as_zeros: bool = False
    """Whether to initialize the decoder bias as zeros.
    If False, it is initialized as the estimated geometric median."""


@dataclass
class TrainConfig(Serializable):
    sae: SaeConfig

    batch_size: int = 8
    """Batch size measured in sequences.
    The effective SAE batch-size is `cfg.batch_size * cfg.max_seq_len`"""

    max_seq_len: int = 1024
    """The maximum sequence length"""

    num_training_tokens: int = 1_000_000
    """Number of total training tokens"""

    cycle_iterator: bool = True
    """Whether to use a CycleIterator"""

    grad_acc_steps: int = 1
    """Number of steps over which to accumulate gradients."""

    micro_acc_steps: int = 1
    """Chunk the activations into this number of microbatches for SAE training."""

    adam_8bit: bool = False
    """Whether to use 8bit Adam"""

    adam_epsilon: float = 1e-8
    """Adam epsilon"""

    adam_betas: tuple[float, float] = (0.0, 0.999)
    """Adam betas"""

    lr: dict[str, float] | float | None = None
    """Base lr. If None, it is automatically chosen based on the number of latents."""

    lr_scheduler_name: str = "constant"
    """LR scheduler name. One of `constant`, `linear`, `cosine`."""

    lr_warmup_steps: float = 0.01
    """Percentage (in [0;1]) of total steps to warm-up the learning rate"""

    l1_coefficient: float = 0.0
    """Sparsity coefficient"""

    l1_warmup_steps: float = 0.0
    """Percentage (in [0;1]) of total steps to warm-up the sparsity coefficient"""

    use_l2_loss: bool = True
    """Whether to use the L2 loss as the reconstruction loss instead of the FVU"""

    auxk_alpha: float = 0.0
    """Weight of the auxiliary loss term."""

    dead_feature_threshold: int = 10_000_000
    """Number of tokens after which a feature is considered dead."""

    hookpoints: list[str] = list_field()
    """List of hookpoints to train SAEs on."""

    layers: list[int] = list_field()
    """List of layer indices to train SAEs on."""

    layer_stride: int = 1
    """Stride between layers to train SAEs on."""

    distribute_modules: bool = False
    """Store a single copy of each SAE, instead of copying them across devices."""

    save_every: int = 1000
    """Save SAEs every `save_every` steps."""

    normalize_activations: float | None = None
    """Normalize the activations to have an expected norm of `normalize_activations`."""

    num_norm_estimation_tokens: int = 1_000_000
    """Number of tokens to use for estimating the normalization factor."""

    hook: Callable[
        [nn.Module, Tuple[Any, ...], Any, Dict[nn.Module, str], Dict[str, Tensor]],
        Optional[Any],
    ] = standard_hook
    """The hook function to be used to collect model activations"""

    log_to_wandb: bool = True
    run_name: str | None = None
    wandb_log_frequency: int = 1

    def __post_init__(self):
        assert not (
            self.layers and self.layer_stride != 1
        ), "Cannot specify both `layers` and `layer_stride`."
        # if (
        #     self.normalize_activations is not None
        #     and self.normalize_activations.lower() not in {"expected_norm_1", "norm_1"}
        # ):
        #     raise ValueError(
        #         "`normalize_activations` must be one of `expected_norm_1` or `norm_1`. "
        #         "To disable activations normalization pass None instead."
        #     )
