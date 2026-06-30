import os
from contextlib import contextmanager

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)


def init_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
    return local_rank, rank, world_size


def is_distributed():
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def get_local_rank():
    return int(os.environ.get("LOCAL_RANK", 0))


def get_world_size():
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def fsdp_wrap(model, device_id, sharding_strategy=ShardingStrategy.FULL_SHARD, **kwargs):
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


@contextmanager
def full_state_dict(model):
    if is_distributed():
        config = FullStateDictConfig(rank0_only=True, offload_to_cpu=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, config):
            yield
    else:
        yield


def save_fsdp_model(model, path):
    with full_state_dict(model):
        state_dict = model.state_dict()
        if get_local_rank() == 0:
            torch.save(state_dict, path)
    if is_distributed():
        dist.barrier()
