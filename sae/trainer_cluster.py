import copy
import os
import shutil
import warnings
from collections import defaultdict
from dataclasses import asdict

import torch
import torch.distributed as dist
from safetensors.torch import load_model
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import PreTrainedModel

from .config import TrainConfig
from .hooks import forward_hook_wrapper, standard_hook
from .normalization import estimate_norm_scaling_factor
from .sae import Sae
from .utils import (
    CycleIterator,
    L1Scheduler,
    geometric_median,
    get_layer_list,
    get_lr_scheduler,
    resolve_widths,
)


class ClusterSaeTrainer:
    def __init__(
        self,
        cfg: TrainConfig,
        dataloader: DataLoader,
        model: PreTrainedModel,
    ):
        if cfg.hook is None:
            warnings.warn("No hook function specified, using the standard hook.")
            cfg.hook = standard_hook
        if cfg.distribute_modules:
            raise ValueError(
                "Distributing modules is not supported in `SaeTrainer` with clusters. "
                "Please set `distribute_modules=False`."
            )
        if cfg.hookpoints or cfg.layers:
            raise ValueError(
                "The `hookpoints` and `layers` parameters "
                "are not supported in `SaeTrainer` with clusters. "
                "Please use the `clusters` parameter instead."
            )
        else:
            # If no layers are specified, train on all of them
            if cfg.clusters is not None:
                N = max(max(cluster) for cluster in cfg.clusters.values()) + 1
                cfg.layers = list(range(0, N, cfg.layer_stride))
            else:
                raise ValueError("No clusters specified. Please specify the `clusters` parameter.")

            # Now convert layers to hookpoints
            layers_name, _ = get_layer_list(model)
            cfg.hookpoints = [f"{layers_name}.{i}" for i in cfg.layers]
            cfg.cluster_hookpoints = {
                cluster_name: [f"{layers_name}.{i}" for i in layers]
                for cluster_name, layers in cfg.clusters.items()
            }

        # Distribute modules
        self.cfg = cfg
        self.distribute_modules()

        # Check shapes
        cluster_shapes = {}
        input_shapes = resolve_widths(cfg, model, cfg.hookpoints, dataloader=dataloader)
        for cluster_name in cfg.clusters:
            cluster_shapes[cluster_name] = [
                input_shapes[hook] for hook in cfg.cluster_hookpoints[cluster_name]
            ]
            if len(set(cluster_shapes[cluster_name])) != 1:
                raise ValueError(
                    f"Cluster {cluster_name} has different input shapes: "
                    f"{cluster_shapes[cluster_name]}"
                )
        unique_shapes = set(input_shapes.values())
        if cfg.distribute_modules and len(unique_shapes) > 1:
            # dist.all_to_all requires tensors to have the same shape across ranks
            raise ValueError(
                f"All modules must output tensors of the same shape when using "
                f"`distribute_modules=True`, got {unique_shapes}"
            )
        # Moreover, we request that the hook returns 2D tensors of shape B x D
        for shape in input_shapes.values():
            if len(shape) != 2:
                raise ValueError(f"The hook must return 2D tensors of shape B x D, got {shape}")

        # SAEs
        self.model = model
        device = model.device
        self.saes = {
            cluster_name: Sae(cluster_shapes[cluster_name][0][-1], cfg.sae, device)
            for cluster_name in cfg.cluster_hookpoints
        }  # We suppose the input shapes are the same

        # Dataloader
        self.dataloader = dataloader
        real_seq_len = list(unique_shapes)[0][0] // cfg.batch_size
        print(
            f"The specified maximum sequence length is {cfg.max_seq_len}. "
            f"The real sequence length after the SAE hook is {real_seq_len}"
        )
        self.num_training_tokens = cfg.num_training_tokens
        self.dataloader = CycleIterator(self.dataloader)
        self.tokens_per_batch = cfg.batch_size * real_seq_len
        self.training_steps = self.num_training_tokens // self.tokens_per_batch
        self.training_steps //= dist.get_world_size() if dist.is_initialized() else 1

        # Variables for global stats
        self.global_step = 0
        self.num_tokens_since_fired = {
            name: torch.zeros(sae.num_latents, device=device, dtype=torch.long)
            for name, sae in self.saes.items()
        }

        # Optimizer and schedulers (l1 coefficient and lr)
        # Handle different types of lr: dict, float, or None
        self.lrs = {}
        self.pgs = []
        for hook, sae in self.saes.items():
            if cfg.lr is not None:
                lr = cfg.lr
            else:
                num_latents = sae.num_latents
                # Compute default lr based on num_latents
                lr = 2e-4 / (num_latents / (2**14)) ** 0.5
            self.pgs.append(
                {
                    "params": sae.parameters(),
                    "lr": lr,
                }
            )
            self.lrs[hook] = lr

        # Deduplicate and sort the learning rates for logging
        lrs_set = sorted(set(self.lrs.values()))
        lrs_formatted = [f"{lr:.2e}" for lr in lrs_set]
        if len(lrs_set) > 1:
            print(f"Learning rates: {lrs_formatted}")
        else:
            print(f"Learning rate: {lrs_formatted[0]}")

        # Initialize the optimizer
        if cfg.adam_8bit:
            try:
                from bitsandbytes.optim import Adam8bit as Adam

                print("Using 8-bit Adam from bitsandbytes")
            except ImportError:
                from torch.optim import Adam

                print("bitsandbytes 8-bit Adam not available, using torch.optim.Adam")
                print("Run `pip install bitsandbytes` for less memory usage.")
        else:
            from torch.optim import Adam
        self.optimizer = Adam(self.pgs, betas=cfg.adam_betas, eps=cfg.adam_epsilon)

        # LR scheduler setup
        if not (0 <= cfg.lr_warmup_steps <= 1):
            raise ValueError(
                "`lr_warmup_steps` must be a float between 0 and 1. "
                f"Given: {cfg.lr_warmup_steps}"
            )
        lr_warmup_steps = int(cfg.lr_warmup_steps * self.training_steps)
        self.lr_scheduler = get_lr_scheduler(
            cfg.lr_scheduler_name,
            optimizer=self.optimizer,
            num_warmup_steps=lr_warmup_steps,
            num_training_steps=self.training_steps,
        )

        # L1 coefficient scheduler
        self.l1_scheduler = None
        if self.cfg.sae.k <= 0:
            if not (0 <= cfg.l1_warmup_steps <= 1):
                raise ValueError(
                    "`l1_warmup_steps` must be a float between 0 and 1. "
                    f"Given: {cfg.l1_warmup_steps}"
                )
            l1_warmup_steps = int(cfg.l1_warmup_steps * self.training_steps)
            self.l1_scheduler = L1Scheduler(
                l1_warmup_steps=l1_warmup_steps,  # type: ignore
                total_steps=self.training_steps,
                final_l1_coefficient=cfg.l1_coefficient,
            )

        # Resume from checkpoint
        if cfg.resume_from is not None:
            self.load_state(cfg.resume_from)

        # Normalize activations
        if cfg.resume_from is None:
            self.estimate_norm_scaling_factor()

    def estimate_norm_scaling_factor(self):
        if not self.cfg.normalize_activations:
            return
        name_to_module = {name: self.model.get_submodule(name) for name in self.cfg.hookpoints}
        module_to_name = {v: k for k, v in name_to_module.items()}
        scaling_factors = estimate_norm_scaling_factor(
            self.dataloader,
            self.model,
            (
                self.cfg.num_norm_estimation_tokens // dist.get_world_size()
                if dist.is_initialized()
                else 1
            ),
            self.cfg.hook,
            module_to_name=module_to_name,
            target_norm=self.cfg.normalize_activations,
            device=self.model.device,
        )
        if dist.is_available() and dist.is_initialized():
            all_scaling_factors = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(all_scaling_factors, scaling_factors)
            # Average the scaling factors across all ranks
            for hook_name in scaling_factors.keys():
                scaling_factors[hook_name] = sum(
                    sf[hook_name] for sf in all_scaling_factors if sf is not None
                ) / len(all_scaling_factors)
        self.scaling_factors = scaling_factors

    def load_state(self, path: str):
        """Load the trainer state from disk."""
        device = self.model.device

        # Load the train state first so we can print the step number
        train_state = torch.load(f"{path}/state.pt", map_location=device, weights_only=True)
        self.global_step = train_state["global_step"]
        self.num_tokens_since_fired = train_state["num_tokens_since_fired"]

        print(f"\033[92mResuming training at step {self.global_step} from '{path}'\033[0m")

        lr_state = torch.load(f"{path}/lr_scheduler.pt", map_location=device, weights_only=True)
        opt_state = torch.load(f"{path}/optimizer.pt", map_location=device, weights_only=True)
        self.optimizer.load_state_dict(opt_state)
        self.lr_scheduler.load_state_dict(lr_state)
        if self.l1_scheduler is not None and self.cfg.sae.k <= 0:
            l1_state = torch.load(
                f"{path}/l1_scheduler.pt", map_location=device, weights_only=True
            )
            self.l1_scheduler.load_state_dict(l1_state)

        for name, sae in self.saes.items():
            load_model(sae, f"{path}/{name}/sae.safetensors", device=str(device))

    def fit(self):
        # Use Tensor Cores even for fp32 matmuls
        torch.set_float32_matmul_precision("high")

        rank_zero = not dist.is_initialized() or dist.get_rank() == 0
        ddp = dist.is_initialized() and not self.cfg.distribute_modules

        if self.cfg.log_to_wandb and rank_zero:
            try:
                import wandb

                wandb.init(
                    name=self.cfg.run_name,
                    project="sae",
                    config=asdict(self.cfg),
                    save_code=True,
                )
            except ImportError:
                print("Weights & Biases not installed, skipping logging.")
                self.cfg.log_to_wandb = False

        num_sae_params = sum(p.numel() for s in self.saes.values() for p in s.parameters())
        num_model_params = sum(p.numel() for p in self.model.parameters())
        print(f"Number of SAE parameters: {num_sae_params:_}")
        print(f"Number of model parameters: {num_model_params:_}")

        device = self.model.device

        pbar = tqdm(
            desc="Training",
            disable=not rank_zero,
            initial=self.global_step,
            total=self.training_steps,
        )

        did_fire = {
            name: torch.zeros(sae.num_latents, device=device, dtype=torch.bool)
            for name, sae in self.saes.items()
        }
        num_tokens_in_step = 0

        # For logging purposes
        elapsed_tokens = 0
        avg_l1 = defaultdict(float)
        avg_l0 = defaultdict(float)
        avg_l2 = defaultdict(float)
        avg_fvu = defaultdict(float)
        avg_auxk_loss = defaultdict(float)
        avg_multi_topk_fvu = defaultdict(float)
        avg_act_norm = defaultdict(float)
        running_mean_act_norm = {}

        # Function to update the running mean
        def update_running_mean(running_mean, new_value, count):
            return (running_mean * (count - 1) + new_value) / count

        hidden_dict: dict[str, Tensor] = {}
        name_to_module = {name: self.model.get_submodule(name) for name in self.cfg.hookpoints}
        maybe_wrapped: dict[str, DDP] | dict[str, Sae] = {}
        module_to_name = {v: k for k, v in name_to_module.items()}

        for batch_idx, batch in enumerate(self.dataloader):
            if self.global_step >= self.training_steps:
                break

            hidden_dict.clear()

            # Bookkeeping for dead feature detection
            tokens_in_batch = batch["input_ids"].numel()
            elapsed_tokens += tokens_in_batch * (dist.get_world_size() if ddp else 1)
            num_tokens_in_step += tokens_in_batch

            # Forward pass on the model to get the next batch of activations
            handles = [
                mod.register_forward_hook(
                    forward_hook_wrapper(
                        self.cfg.hook,
                        module_to_name=module_to_name,
                        hidden_dict=hidden_dict,
                    )
                )
                for mod in name_to_module.values()
            ]
            try:
                with torch.no_grad():
                    self.model(batch["input_ids"].to(device))
            finally:
                for handle in handles:
                    handle.remove()

            if self.cfg.distribute_modules:
                hidden_dict = self.scatter_hiddens(hidden_dict)

            # Normalize the activations
            if self.cfg.normalize_activations:
                with torch.no_grad():
                    for name, activations in hidden_dict.items():
                        hidden_dict[name] = activations * self.scaling_factors[name]

            # Save the running mean of the L2 norm of the activations
            for name, hiddens in hidden_dict.items():
                l2_norm = hiddens.norm(p=2, dim=-1).mean()
                if name not in running_mean_act_norm:
                    running_mean_act_norm[name] = l2_norm
                else:
                    running_mean_act_norm[name] = update_running_mean(
                        running_mean_act_norm[name], l2_norm, batch_idx + 1
                    )

            # For every cluster of layers, sample one activation per layer
            cluster_layers_dict = {}
            cluster_activations_dict = {}
            if self.cfg.clusters is not None:
                # Collect the activations for each layer in a single tensor
                activations_by_layer = torch.stack(
                    [hidden_dict[hook] for hook in self.local_hookpoints()], dim=1
                )  # B x L x D
                B, L, D = activations_by_layer.shape

                for cluster_name, cluster in self.cfg.clusters.items():
                    cluster_activations = activations_by_layer[:, cluster, :]  # B x C x D
                    indices = torch.randint(0, cluster_activations.shape[1], (B,), device=device)
                    cluster_layers_dict[cluster_name] = indices + min(cluster)
                    indices = indices.view(-1, 1, 1).expand(B, 1, D)
                    sampled_cluster_activations = torch.gather(
                        cluster_activations, 1, indices
                    )  # B x 1 x D
                    cluster_activations_dict[cluster_name] = sampled_cluster_activations

            for name, hiddens in cluster_activations_dict.items():
                raw = self.saes[name]  # 'raw' never has a DDP wrapper

                # On the first iteration, initialize the decoder bias
                if self.global_step == 0 and not self.cfg.sae.init_b_dec_as_zeros:
                    # NOTE: The all-cat here could conceivably cause an OOM in some
                    # cases, but it's unlikely to be a problem with small world sizes.
                    # We could avoid this by "approximating" the geometric median
                    # across all ranks with the mean (median?) of the geometric medians
                    # on each rank. Not clear if that would hurt performance.
                    median = geometric_median(self.maybe_all_cat(hiddens))
                    raw.b_dec.data = median.to(raw.dtype)

                if not maybe_wrapped:
                    # Wrap the SAEs with Distributed Data Parallel. We have to do this
                    # after we set the decoder bias, otherwise DDP will not register
                    # gradients flowing to the bias after the first step.
                    maybe_wrapped = (
                        {
                            name: DDP(sae, device_ids=[dist.get_rank()])
                            for name, sae in self.saes.items()
                        }
                        if ddp
                        else self.saes
                    )

                # Make sure the W_dec is still unit-norm
                if raw.cfg.normalize_decoder:
                    raw.set_decoder_norm_to_unit_norm()

                acc_steps = self.cfg.grad_acc_steps * self.cfg.micro_acc_steps
                denom = acc_steps * self.cfg.wandb_log_frequency
                wrapped = maybe_wrapped[name]

                # Save memory by chunking the activations
                for chunk in hiddens.chunk(self.cfg.micro_acc_steps):
                    out = wrapped(
                        chunk,
                        dead_mask=(
                            self.num_tokens_since_fired[name] > self.cfg.dead_feature_threshold
                            if self.cfg.auxk_alpha > 0
                            else None
                        ),
                    )

                    avg_fvu[name] += float(self.maybe_all_reduce(out.fvu.detach()) / denom)
                    avg_l0[name] += float(
                        self.maybe_all_reduce(
                            (out.feature_acts.detach() > 0).float().sum(-1).mean()
                        )
                        / denom
                    )
                    avg_l2[name] += float(self.maybe_all_reduce(out.l2_loss.detach()) / denom)
                    if self.cfg.auxk_alpha > 0:
                        avg_auxk_loss[name] += float(
                            self.maybe_all_reduce(out.auxk_loss.detach()) / denom
                        )
                    if self.cfg.sae.multi_topk:
                        avg_multi_topk_fvu[name] += float(
                            self.maybe_all_reduce(out.multi_topk_fvu.detach()) / denom
                        )
                    if self.cfg.sae.k <= 0:
                        avg_l1[name] += float(self.maybe_all_reduce(out.l1_loss.detach()) / denom)

                    if self.cfg.use_l2_loss:
                        recon_loss = out.l2_loss
                    else:
                        recon_loss = out.fvu
                    if self.cfg.sae.k <= 0:
                        if self.cfg.sae.jumprelu and self.cfg.sae.jumprelu_target_l0 is not None:
                            l0 = (out.l1_loss / self.cfg.sae.jumprelu_target_l0 - 1) ** 2
                        else:
                            l0 = out.l1_loss
                        sparsity_loss = l0
                        if self.l1_scheduler is not None:
                            sparsity_loss *= self.l1_scheduler.current_l1_coefficient
                    else:
                        sparsity_loss = 0.0

                    loss = (
                        recon_loss
                        + sparsity_loss
                        + self.cfg.auxk_alpha * out.auxk_loss
                        + out.multi_topk_fvu / 8
                    )
                    loss.div(acc_steps).backward()

                    # Update the did_fire mask
                    if out.topk_indices is not None:
                        did_fire[name][out.topk_indices.flatten()] = True
                        self.maybe_all_reduce(did_fire[name], "max")  # max is boolean "any"

                # Clip gradient norm independently for each SAE
                torch.nn.utils.clip_grad_norm_(raw.parameters(), 1.0)

            # Check if we need to actually do a training step
            step, substep = divmod(self.global_step + 1, self.cfg.grad_acc_steps)
            if substep == 0:
                if self.cfg.sae.normalize_decoder:
                    for sae in self.saes.values():
                        sae.remove_gradient_parallel_to_decoder_directions()

                self.optimizer.step()
                self.optimizer.zero_grad()
                self.lr_scheduler.step()
                if self.l1_scheduler is not None and self.cfg.sae.k <= 0:
                    self.l1_scheduler.step()

                ###############
                with torch.no_grad():
                    # Update the dead feature mask
                    for name, counts in self.num_tokens_since_fired.items():
                        counts += num_tokens_in_step
                        counts[did_fire[name]] = 0

                    # Reset stats for this step
                    num_tokens_in_step = 0
                    for mask in did_fire.values():
                        mask.zero_()

                if self.cfg.log_to_wandb and (step + 1) % self.cfg.wandb_log_frequency == 0:
                    info = {}

                    for name in self.saes:
                        mask = self.num_tokens_since_fired[name] > self.cfg.dead_feature_threshold

                        info.update(
                            {
                                f"fvu/{name}": avg_fvu[name],
                                f"l1/{name}": avg_l1[name],
                                "l1/l1_coefficient": (
                                    self.l1_scheduler.current_l1_coefficient
                                    if self.l1_scheduler is not None
                                    else 0.0
                                ),
                                f"l0/{name}": avg_l0[name],
                                f"l2/{name}": avg_l2[name],
                                f"dead_pct/{name}": mask.mean(dtype=torch.float32).item(),
                            }
                        )
                        info.update(
                            {
                                f"lr/group_{g}": lr
                                for g, lr in enumerate(self.lr_scheduler.get_last_lr())
                            }
                        )

                        if self.cfg.auxk_alpha > 0:
                            info[f"auxk/{name}"] = avg_auxk_loss[name]
                        if self.cfg.sae.multi_topk:
                            info[f"multi_topk_fvu/{name}"] = avg_multi_topk_fvu[name]

                    info.update(
                        {
                            f"norm/running_mean_act_norm_{name}": running_mean_act_norm[name]
                            for name in self.cfg.hookpoints
                        }
                    )
                    info.update(
                        {"tokens/elapsed": elapsed_tokens * (dist.get_world_size() if ddp else 1)}
                    )

                    avg_auxk_loss.clear()
                    avg_fvu.clear()
                    avg_l1.clear()
                    avg_l0.clear()
                    avg_l2.clear()
                    avg_multi_topk_fvu.clear()
                    avg_act_norm.clear()

                    if self.cfg.distribute_modules:
                        outputs = [{} for _ in range(dist.get_world_size())]
                        dist.gather_object(info, outputs if rank_zero else None)
                        info.update({k: v for out in outputs for k, v in out.items()})

                    if rank_zero:
                        wandb.log(info, step=step)

                if (step + 1) % self.cfg.save_every == 0:
                    self.save()

            self.global_step += 1
            pbar.update()

        self.save()
        pbar.close()

    def local_hookpoints(self) -> list[str]:
        return self.module_plan[dist.get_rank()] if self.module_plan else self.cfg.hookpoints

    def maybe_all_cat(self, x: Tensor) -> Tensor:
        """Concatenate a tensor across all processes."""
        if not dist.is_initialized() or self.cfg.distribute_modules:
            return x

        buffer = x.new_empty([dist.get_world_size() * x.shape[0], *x.shape[1:]])
        dist.all_gather_into_tensor(buffer, x)
        return buffer

    def maybe_all_reduce(self, x: Tensor, op: str = "mean") -> Tensor:
        if not dist.is_initialized() or self.cfg.distribute_modules:
            return x

        if op == "sum":
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
        elif op == "mean":
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
            x /= dist.get_world_size()
        elif op == "max":
            dist.all_reduce(x, op=dist.ReduceOp.MAX)
        else:
            raise ValueError(f"Unknown reduction op '{op}'")

        return x

    def distribute_modules(self):
        """Prepare a plan for distributing modules across ranks."""
        if not self.cfg.distribute_modules:
            self.module_plan = []
            print(f"Training on modules: {self.cfg.hookpoints}")
            return

        layers_per_rank, rem = divmod(len(self.cfg.hookpoints), dist.get_world_size())
        assert rem == 0, "Number of modules must be divisible by world size"

        # Each rank gets a subset of the layers
        self.module_plan = [
            self.cfg.hookpoints[start : start + layers_per_rank]
            for start in range(0, len(self.cfg.hookpoints), layers_per_rank)
        ]
        for rank, modules in enumerate(self.module_plan):
            print(f"Rank {rank} modules: {modules}")

    def scatter_hiddens(self, hidden_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        """Scatter & gather the hidden states across ranks."""
        outputs = [
            # Add a new leading "layer" dimension to each tensor
            torch.stack([hidden_dict[hook] for hook in hookpoints], dim=1)
            for hookpoints in self.module_plan
        ]
        local_hooks = self.module_plan[dist.get_rank()]
        shape = next(iter(hidden_dict.values())).shape

        # Allocate one contiguous buffer to minimize memcpys
        buffer = outputs[0].new_empty(
            # The (micro)batch size times the world size
            shape[0] * dist.get_world_size(),
            # The number of layers we expect to receive
            len(local_hooks),
            # All other dimensions
            *shape[1:],
        )

        # Perform the all-to-all scatter
        inputs = buffer.split([len(output) for output in outputs])
        dist.all_to_all([x for x in inputs], outputs)

        # Return a list of results, one for each layer
        return {hook: buffer[:, i] for i, hook in enumerate(local_hooks)}

    def save(self):
        """Save the SAEs to disk."""

        run_path = self.cfg.run_name or "sae-ckpts"
        path = f"{run_path}/step_{self.global_step}"
        rank_zero = not dist.is_initialized() or dist.get_rank() == 0
        if rank_zero:
            os.makedirs(run_path, exist_ok=True)

        if rank_zero:
            if self.cfg.keep_last_n_checkpoints > 0:
                checkpoints = [f"{run_path}/{p}" for p in os.listdir(run_path) if "step" in p]
                checkpoints = sorted(
                    checkpoints, key=lambda x: int(x.split("_")[-1]), reverse=False
                )
                if self.cfg.keep_last_n_checkpoints == 1:
                    to_remove = checkpoints
                elif len(checkpoints) >= self.cfg.keep_last_n_checkpoints:
                    to_remove = checkpoints[: -self.cfg.keep_last_n_checkpoints + 1]
                else:
                    to_remove = []
                for path_to_remove in to_remove:
                    shutil.rmtree(f"{path_to_remove}", ignore_errors=False)

        # Gather SAEs from all ranks
        if rank_zero or self.distribute_modules:
            os.makedirs(path, exist_ok=True)
            if dist.is_initialized():
                all_saes = [None for _ in range(dist.get_world_size())]
                dist.gather_object(self.saes, all_saes if rank_zero else None)
            else:
                all_saes = [self.saes]

            for saes in all_saes:
                if saes is not None:
                    for hook, sae in saes.items():
                        assert isinstance(sae, Sae)
                        sae.save_to_disk(f"{path}/{hook}")

        # We can save the optimizer and scheduler states only from rank 0
        # because they are the same across all ranks
        if rank_zero:
            torch.save(self.lr_scheduler.state_dict(), f"{path}/lr_scheduler.pt")
            torch.save(self.optimizer.state_dict(), f"{path}/optimizer.pt")
            torch.save(
                {
                    "global_step": self.global_step,
                    "num_tokens_since_fired": self.num_tokens_since_fired,
                },
                f"{path}/state.pt",
            )
            if self.l1_scheduler is not None:
                torch.save(self.l1_scheduler.state_dict(), f"{path}/l1_scheduler.pt")
            if self.cfg.normalize_activations:
                torch.save(self.scaling_factors, f"{path}/scaling_factors.pt")

            local_cfg = copy.deepcopy(self.cfg)
            local_cfg.hook = None
            local_cfg.save_json(f"{path}/config.json")

        # Barrier to ensure all ranks have saved before continuing
        if dist.is_initialized():
            dist.barrier()
