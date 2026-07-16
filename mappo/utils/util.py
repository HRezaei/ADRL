import os
import json
import numpy as np
import math
import torch

def check(input):
    if type(input) == np.ndarray:
        return torch.from_numpy(input)
        
def get_gard_norm(it):
    sum_grad = 0
    for x in it:
        if x.grad is None:
            continue
        sum_grad += x.grad.norm() ** 2
    return math.sqrt(sum_grad)

def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (e > d).float()
    return a*e**2/2 + b*d*(abs(e)-d/2)

def mse_loss(e):
    return e**2/2

def get_shape_from_obs_space(obs_space):
    if obs_space.__class__.__name__ == 'Box':
        obs_shape = obs_space.shape
    elif obs_space.__class__.__name__ == 'list':
        obs_shape = obs_space
    else:
        raise NotImplementedError
    return obs_shape

def get_shape_from_act_space(act_space):
    if act_space.__class__.__name__ == 'Discrete':
        act_shape = 1
    elif act_space.__class__.__name__ == "MultiDiscrete":
        act_shape = act_space.shape
    elif act_space.__class__.__name__ == "Box":
        act_shape = act_space.shape[0]
    elif act_space.__class__.__name__ == "MultiBinary":
        act_shape = act_space.shape[0]
    else:  # agar
        act_shape = act_space[0].shape[0] + 1  
    return act_shape


def tile_images(img_nhwc):
    """
    Tile N images into one big PxQ image
    (P,Q) are chosen to be as close as possible, and if N
    is square, then P=Q.
    input: img_nhwc, list or array of images, ndim=4 once turned into array
        n = batch index, h = height, w = width, c = channel
    returns:
        bigim_HWc, ndarray with ndim=3
    """
    img_nhwc = np.asarray(img_nhwc)
    N, h, w, c = img_nhwc.shape
    H = int(np.ceil(np.sqrt(N)))
    W = int(np.ceil(float(N)/H))
    img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0]*0 for _ in range(N, H*W)])
    img_HWhwc = img_nhwc.reshape(H, W, h, w, c)
    img_HhWwc = img_HWhwc.transpose(0, 2, 1, 3, 4)
    img_Hh_Ww_c = img_HhWwc.reshape(H*h, W*w, c)
    return img_Hh_Ww_c


def push_to_hub(folder_path, hub_id, episode=None):
    """Upload a local model folder to HuggingFace Hub.

    Requires the HF_TOKEN environment variable to be set.
    Each episode is pushed as a separate revision (commit) so that
    different checkpoints are preserved.
    """
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN environment variable is not set. "
            "Generate a token at https://huggingface.co/settings/tokens "
            "and export it: export HF_TOKEN=hf_..."
        )

    api = HfApi(token=token)
    repo_url = api.create_repo(repo_id=hub_id, repo_type="model", exist_ok=True)
    print(f"{repo_url=}")
    commit_message = f"checkpoint episode {episode}" if episode is not None else "update"
    print(f"[push_to_hub] uploading {folder_path} -> {hub_id} ({commit_message})")
    api.upload_folder(
        folder_path=folder_path,
        repo_id=hub_id,
        commit_message=commit_message,
    )
    print(f"[push_to_hub] done")


def save_checkpoint(save_dir, episode, total_num_steps, trainer, extra=None):
    """Save training checkpoint for resume: optimizer states in .pth, everything else in metadata.json."""
    import random
    from datetime import datetime

    # optimizer state dicts (contain only tensors — safe for torch.load weights_only=True)
    ckpt = {
        'policy_optimizer': trainer.policy_optimizer.state_dict(),
        'critic_optimizer': trainer.critic_optimizer.state_dict(),
    }
    ckpt_path = os.path.join(save_dir, "checkpoint.pth")
    torch.save(ckpt, ckpt_path)

    # all non-tensor state goes to JSON
    np_state = np.random.get_state()
    np_random_state = [np_state[0], np_state[1].tolist(), np_state[2]]

    torch_random_state = torch.random.get_rng_state().tolist()

    cuda_states = []
    for i in range(torch.cuda.device_count()):
        cuda_states.append(torch.cuda.get_rng_state(i).tolist())

    lr_groups = {}
    for name, group in enumerate(trainer.policy_optimizer.param_groups):
        lr_groups[f"policy_group_{name}"] = group.get('lr', None)
    for name, group in enumerate(trainer.critic_optimizer.param_groups):
        lr_groups[f"critic_group_{name}"] = group.get('lr', None)

    metadata = {
        'episode': episode,
        'total_num_steps': total_num_steps,
        'timestamp': datetime.now().isoformat(),
        'optimizer_lr': lr_groups,
        'cuda_devices': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        'numpy_random_state': np_random_state,
        'torch_random_state': torch_random_state,
        'cuda_random_state': cuda_states,
        'python_random_state': random.getstate(),
    }
    if extra is not None:
        metadata['extra'] = {k: v for k, v in extra.items()
                            if not isinstance(v, (torch.Tensor, np.ndarray))}
    with open(os.path.join(save_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    return ckpt_path


def load_checkpoint(ckpt_path, trainer=None, device="cpu"):
    """Load training checkpoint. Returns (ckpt_dict, metadata_dict).

    If trainer is provided, optimizer states are restored into it.
    RNG states are in metadata — the caller should restore them after
    setting up the environment so env seeding happens first.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    meta_path = os.path.join(os.path.dirname(ckpt_path), "metadata.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
    if trainer is not None:
        trainer.policy_optimizer.load_state_dict(ckpt['policy_optimizer'])
        trainer.critic_optimizer.load_state_dict(ckpt['critic_optimizer'])
    return ckpt, meta


def restore_rng_states(meta):
    """Restore RNG states from a metadata dict."""
    import random
    # numpy
    np_name, np_array, np_pos = meta['numpy_random_state']
    np.random.set_state((np_name, np.array(np_array, dtype=np.uint8), np_pos))
    # torch CPU
    torch.random.set_rng_state(torch.tensor(meta['torch_random_state'], dtype=torch.uint8))
    # torch CUDA
    for i, cuda_state in enumerate(meta.get('cuda_random_state', [])):
        if i < torch.cuda.device_count():
            torch.cuda.set_rng_state(torch.tensor(cuda_state, dtype=torch.uint8), i)
    # python
    py_state = meta['python_random_state']
    # state is (int, tuple[int,...], int)
    random.setstate((py_state[0], tuple(py_state[1]), py_state[2]))


def find_latest_checkpoint(run_dir):
    """Find the latest checkpoint.pth under a run directory's models/ folder.

    Returns the path to checkpoint.pth or None if none found.
    """
    models_dir = os.path.join(run_dir, "models")
    if not os.path.isdir(models_dir):
        return None
    ckpt_dirs = []
    for name in os.listdir(models_dir):
        if name.startswith("episode_") and os.path.isdir(os.path.join(models_dir, name)):
            try:
                ep = int(name.split("episode_")[1])
                ckpt_path = os.path.join(models_dir, name, "checkpoint.pth")
                if os.path.exists(ckpt_path):
                    ckpt_dirs.append((ep, ckpt_path))
            except (ValueError, IndexError):
                continue
    if not ckpt_dirs:
        return None
    ckpt_dirs.sort(key=lambda x: x[0])
    return ckpt_dirs[-1][1]