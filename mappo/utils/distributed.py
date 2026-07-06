import os

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import get_state_dict
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
)


def init_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
    return local_rank, rank, world_size


def is_distributed():
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def get_local_rank():
    return int(os.environ.get("LOCAL_RANK", 0))


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def get_world_size():
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def is_main_process():
    return get_rank() == 0


def fsdp_wrap(model, device_id, sharding_strategy=ShardingStrategy.NO_SHARD, **kwargs):
    if not is_distributed():
        return model
    mixed_precision = MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float16,
    )
    model = model.to(f"cuda:{device_id}")
    # FSDP flattens all parameters per unit into a single flat buffer,
    # which requires uniform dtype.  Models often contain a mix of
    # float16 and float32 (e.g. T5's LayerNorm params).  Cast to the
    # MixedPrecision param_dtype so flattening succeeds.
    param_dtypes = {p.dtype for p in model.parameters()}
    if len(param_dtypes) > 1:
        model = model.to(dtype=mixed_precision.param_dtype)
    return FSDP(
        model,
        sharding_strategy=sharding_strategy,
        mixed_precision=mixed_precision,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        **kwargs,
    )


def reduce_train_info(info):
    """All-reduce (sum) all scalar values in a dict across GPUs."""
    if not is_distributed():
        return info
    for k, v in info.items():
        if isinstance(v, (int, float, np.integer, np.floating)):
            t = torch.tensor([v], dtype=torch.float64, device="cuda")
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            info[k] = type(v)(t.item())
    return info


def save_fsdp_model(model, path):
    state_dict = get_state_dict(model, optimizers=[])
    if get_local_rank() == 0:
        torch.save(state_dict, path)
    if is_distributed():
        dist.barrier()
