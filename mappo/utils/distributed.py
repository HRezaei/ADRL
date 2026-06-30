import os
from pathlib import Path

import torch
import torch.distributed as dist


def init_distributed_mode(args):
    """Initialize torch.distributed from torchrun environment variables."""
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    requested = getattr(args, "distributed", False)
    args.distributed = requested or env_world_size > 1
    args.rank = int(os.environ.get("RANK", "0"))
    args.world_size = env_world_size
    args.local_rank = int(os.environ.get("LOCAL_RANK", getattr(args, "local_rank", 0)))

    if not args.distributed:
        args.device = "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
        if args.device == "cpu" and torch.backends.mps.is_available():
            args.device = "mps"
        return

    if not torch.cuda.is_available():
        raise RuntimeError("Distributed training currently requires CUDA GPUs.")

    torch.cuda.set_device(args.local_rank)
    args.device = f"cuda:{args.local_rank}"
    if not dist.is_initialized():
        dist.init_process_group(backend=getattr(args, "dist_backend", "nccl"))
    dist.barrier()


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def get_world_size():
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def is_main_process():
    return get_rank() == 0


def get_device(args=None):
    if args is not None and hasattr(args, "device"):
        return args.device
    if is_dist_avail_and_initialized():
        return f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def broadcast_run_dir(run_dir):
    """Share a rank-0-created run directory with all workers."""
    if not is_dist_avail_and_initialized():
        return Path(run_dir)
    obj = [str(run_dir) if is_main_process() else None]
    dist.broadcast_object_list(obj, src=0)
    dist.barrier()
    return Path(obj[0])


def sync_module_parameters(module):
    if not is_dist_avail_and_initialized():
        return
    with torch.no_grad():
        for param in module.parameters():
            dist.broadcast(param.data, src=0)


def average_gradients(parameters):
    if not is_dist_avail_and_initialized():
        return
    world_size = float(get_world_size())
    for param in parameters:
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data.div_(world_size)


def reduce_train_info(train_info):
    if not is_dist_avail_and_initialized():
        return train_info
    reduced = {}
    for key, value in train_info.items():
        if isinstance(value, (int, float)):
            tensor = torch.tensor(float(value), device=get_device())
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            reduced[key] = (tensor / get_world_size()).item()
        else:
            reduced[key] = value
    return reduced


class NullSummaryWriter:
    def add_scalar(self, *args, **kwargs):
        pass

    def add_scalars(self, *args, **kwargs):
        pass

    def export_scalars_to_json(self, *args, **kwargs):
        pass

    def close(self):
        pass
