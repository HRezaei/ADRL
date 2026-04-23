#!/usr/bin/env python
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

from mappo.scripts.train_virtualhome import build_run_dir

sys.path.append("../../")
from mappo.config import get_config
from mappo.envs.babyai_text.babyai_text_env import BabyAITextEnv
from mappo.runner.shared.babyai_text_runner import BabyAITextRunner as Runner


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
        action="store_true",
        default=False,
        help="Whether to use full-scale model weights.",
    )
    parser.add_argument(
        "--num_past_obs",
        type=int,
        default=3,
        help="Number of past observations to use for prompt generation.",
    )
    return parser.parse_known_args(args)[0]


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    # Skeleton defaults; tune later for specific BabyAI-text tasks.
    all_args.episode_length = 32
    all_args.n_rollout_threads = 4
    all_args.log_interval = 1

    run_dir = build_run_dir(all_args)

    random.seed(all_args.seed)
    np.random.seed(all_args.seed)
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    envs = BabyAITextEnv(
        all_args.env_name, all_args.n_rollout_threads, all_args.seed, num_past_obs=all_args.num_past_obs
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
    }

    runner = Runner(config)
    runner.run()

    if envs is not None:
        envs.close()

    runner.writter.export_scalars_to_json(str(runner.log_dir + "/summary.json"))
    runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
