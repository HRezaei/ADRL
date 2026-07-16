#!/usr/bin/env python
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

from mappo.scripts.train_virtualhome import build_run_dir

import torch.distributed as dist
sys.path.append("../../")
from mappo.config import get_config, validate_tppo_config
from mappo.envs.babyai_text.babyai_text_env import BabyAITextEnv
from mappo.runner.shared.babyai_text_runner import BabyAITextRunner as Runner
from mappo.utils.distributed import init_distributed, is_distributed


def parse_args(args, parser):
    parser.add_argument(
        "--env_name",
        type=str,
        default="BabyAI-Text-v0",
        help="Gym id of the BabyAI-text environment.",
    )
    parser.add_argument("--model_name", type=str, default="", help="Base model path/name.")
    parser.add_argument("--max_new_tokens", type=int, default=32, help="max_new_tokens")
    parser.add_argument("--vacab_size", type=int, default=32000)
    parser.add_argument("--gradient_cp_steps", type=int, default=1)
    parser.add_argument(
        "--use_full_scale",
        type=int,
        nargs="?",
        const=1,
        default=0,
        help="Whether to use full-scale model weights.",
    )
    parser.add_argument(
        "--num_past_obs",
        type=int,
        default=3,
        help="Number of past observations to use for prompt generation.",
    )
    parser.add_argument(
        "--use_planner",
        type=int,
        nargs="?",
        const=1,
        default=0,
        help="Use gold path as teacher actions instead of sampling from LLM",
    )
    return parser.parse_known_args(args)[0]


def main(args):
    local_rank, rank, world_size = init_distributed()

    parser = get_config()
    all_args = parse_args(args, parser)
    validate_tppo_config(all_args)

    if is_distributed():
        all_args.n_rollout_threads_original = all_args.n_rollout_threads
        all_args.n_rollout_threads = max(1, all_args.n_rollout_threads // world_size)
    if rank == 0:
        print(all_args)

    if is_distributed():
        if rank == 0:
            run_dir = build_run_dir(all_args)
        else:
            run_dir = None
        run_dir_list = [run_dir]
        dist.broadcast_object_list(run_dir_list, src=0)
        run_dir = run_dir_list[0]
    else:
        run_dir = build_run_dir(all_args)

    resume_checkpoint = None
    resume_run_dir = getattr(all_args, 'resume_run', None)
    if resume_run_dir:
        from mappo.utils.util import find_latest_checkpoint
        import json as _json
        resume_checkpoint = find_latest_checkpoint(resume_run_dir)
        if resume_checkpoint is None:
            raise RuntimeError(f"No checkpoint found in {resume_run_dir}")
        meta_path = os.path.join(os.path.dirname(resume_checkpoint), "metadata.json")
        with open(meta_path) as _f:
            meta = _json.load(_f)
        episodes_completed = meta['episode'] + 1
        all_args.seed += episodes_completed
        run_dir = Path(resume_run_dir)
        print(f"[resume] checkpoint: {resume_checkpoint}, episodes done: {episodes_completed}, new seed: {all_args.seed}")

    random.seed(all_args.seed)
    np.random.seed(all_args.seed)
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    envs = BabyAITextEnv(
        all_args.env_name, all_args.n_rollout_threads, all_args.seed, num_past_obs=all_args.num_past_obs,
        save_gifs=all_args.save_gifs,
        run_dir=run_dir
    )
    eval_envs = BabyAITextEnv(
        all_args.env_name, all_args.n_eval_rollout_threads, all_args.seed * 5
    )

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": envs.num_agents,
        "run_dir": run_dir,
        "local_rank": local_rank,
        "resume_checkpoint": resume_checkpoint,
    }

    runner = Runner(config)
    runner.run()

    if envs is not None:
        envs.close()

    if rank == 0:
        runner.writter.export_scalars_to_json(str(runner.log_dir + "/summary.json"))
        runner.writter.close()

    if is_distributed():
        dist.destroy_process_group()


if __name__ == "__main__":
    main(sys.argv[1:])
