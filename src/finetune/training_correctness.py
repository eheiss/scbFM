from __future__ import annotations

import math
from collections.abc import Iterable

import torch
import torch.distributed as dist


def validate_backbone_checkpoint(
    checkpoint: dict,
    checkpoint_path: str,
    *,
    gene_num: int,
    selected_gene_count: int,
    max_seq_len: int,
    bin_num: int | None = None,
    minimum_epoch: int = 5,
) -> None:
    """Reject checkpoints whose sequence architecture does not match the runner."""
    expected = {
        "backbone": "cancerfoundation",
        "gene_num": int(gene_num),
        "selected_gene_count": int(selected_gene_count),
        "max_seq_len": int(max_seq_len),
    }
    if bin_num is not None:
        expected["bin_num"] = int(bin_num)
    mismatches = {
        key: {"expected": value, "observed": checkpoint.get(key)}
        for key, value in expected.items()
        if checkpoint.get(key) != value
    }
    try:
        completed_epoch = int(checkpoint.get("epoch"))
    except (TypeError, ValueError):
        completed_epoch = -1
    if completed_epoch < minimum_epoch:
        mismatches["epoch"] = {
            "expected": f">= {minimum_epoch}",
            "observed": checkpoint.get("epoch"),
        }
    if "model_state_dict" not in checkpoint:
        mismatches["model_state_dict"] = {
            "expected": "present",
            "observed": "missing",
        }
    if mismatches:
        raise ValueError(
            f"Checkpoint {checkpoint_path} is incompatible with the configured "
            f"CancerFoundation backbone: {mismatches}"
        )


def is_accumulation_boundary(
    step_idx: int,
    total_steps: int,
    grad_acc_steps: int,
) -> bool:
    return step_idx % grad_acc_steps == 0 or step_idx == total_steps


def optimizer_parameters(
    optimizer: torch.optim.Optimizer,
) -> list[torch.nn.Parameter]:
    return [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]


def normalize_accumulated_gradients(
    parameters: Iterable[torch.nn.Parameter],
    local_normalizer: float,
    *,
    device: torch.device,
    is_distributed: bool,
    world_size: int,
) -> None:
    normalizer = torch.tensor(local_normalizer, dtype=torch.float64, device=device)
    if is_distributed:
        dist.all_reduce(normalizer, op=dist.ReduceOp.SUM)
        # DDP averages gradients. Dividing by the mean local normalizer gives
        # the gradient of the mean over the complete global update batch.
        normalizer /= world_size

    divisor = float(normalizer.item())
    if not math.isfinite(divisor) or divisor <= 0.0:
        raise ValueError(
            f"Accumulated loss normalizer must be positive and finite, got {divisor}."
        )
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.div_(divisor)


def add_optimizer_parameter_group(
    *,
    optimizer: torch.optim.Optimizer,
    scheduler,
    parameters: Iterable[torch.nn.Parameter],
    max_lr: float,
    name: str,
) -> None:
    """Add a post-burn-in group without discarding existing Adam state."""
    parameters = list(parameters)
    if not parameters:
        raise ValueError(f"Cannot add empty optimizer parameter group {name!r}.")
    if max_lr <= 0:
        raise ValueError(f"max_lr must be positive, got {max_lr}.")

    if not scheduler.base_max_lrs or not scheduler.min_lrs:
        raise ValueError("Scheduler has no existing learning-rate groups.")
    min_lr_ratio = scheduler.min_lrs[0] / scheduler.base_max_lrs[0]
    cycle_decay = scheduler.gamma**scheduler.cycle

    optimizer.add_param_group(
        {
            "params": parameters,
            "lr": max_lr * min_lr_ratio,
            "name": name,
        }
    )
    scheduler.base_max_lrs.append(float(max_lr))
    scheduler.max_lrs.append(float(max_lr) * cycle_decay)
    scheduler.min_lrs.append(float(max_lr) * min_lr_ratio)
    scheduler._set_lrs(scheduler.get_lr())
