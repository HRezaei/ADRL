#!/usr/bin/env python3
"""Roll out a VirtualHome gold path and save screenshots."""
import argparse, os, sys, random, json
from pathlib import Path

_ADRL_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, _ADRL_ROOT)

import gym
import numpy as np
from PIL import Image

src = os.path.join(_ADRL_ROOT, "src/virtual-home/virtual-home/virtual_home")
sys.path.insert(0, src)
sys.path.insert(0, os.path.join(src, "../simulation"))
import virtual_home  # noqa: register envs

from mappo.envs.virtualhome.virtualhome_planner import VirtualHomePlanner
from mappo.envs.virtualhome.virtualhome_render import render_v1, render_v2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", default="VirtualHome-v1", choices=["VirtualHome-v1", "VirtualHome-v2"])
    parser.add_argument("--seed", type=int, default=None, help="random seed (default: random)")
    parser.add_argument("--out", default="/tmp/xxx", help="output directory")
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else random.randint(0, 2**31 - 1)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make(args.env_id, seed=seed, debug=False)
    obs = env.reset()

    render_fn = render_v1 if args.env_id == "VirtualHome-v1" else render_v2
    # derive goal description from the env's own obs2text
    dict_obs, _ = env.get_vector_obs(env.state)
    obs_text, _ = env.obs2text(dict_obs)
    import re
    m = re.search(r'(In order to .+?),?\s*your next step is to', obs_text)
    goal_text = m.group(1) if m else obs_text

    # ---- compute gold path ----
    planner = VirtualHomePlanner(args.env_id, env.action_list)
    gold_path = planner.plan(env.state)
    if gold_path is None:
        print("No gold path found!")
        return
    print(f"Seed: {seed}")
    print(f"Gold path ({len(gold_path)} steps): {gold_path}")
    for i, a in enumerate(gold_path):
        print(f"  step {i}: [{env.action_list[a]}]")

    # ---- roll out along gold path ----
    step = 0
    final_reward = None
    for gold_idx, action_idx in enumerate(gold_path):
        action_str = env.action_list[action_idx]
        graph = env.state

        # render
        img_arr = render_fn(graph, action_text=action_str, goal_text=goal_text, seed=seed)
        Image.fromarray(img_arr).save(str(out_dir / f"step_{step:02d}.png"))

        # text trace
        with open(str(out_dir / f"step_{step:02d}.txt"), "w") as f:
            f.write(f"Seed: {seed}\nStep: {step}\n")
            f.write(f"Gold idx: {gold_idx}/{len(gold_path)}\n")
            f.write(f"Action: {action_str}\n")

        # step env
        obs, reward, done, _ = env.step(action_idx)
        final_reward = reward
        step += 1

        # Verify env reward matches success
        if done:
            print(f"  step {step-1}: reward={reward} done=True")

    # final frame after last action
    graph = env.state
    img_arr = render_fn(graph, action_text="DONE", goal_text=goal_text, seed=seed)
    Image.fromarray(img_arr).save(str(out_dir / f"step_{step:02d}_done.png"))
    with open(str(out_dir / f"step_{step:02d}_done.txt"), "w") as f:
        f.write(f"Seed: {seed}\nStep: {step}\nAction: DONE\nReward: {final_reward}\n")

    print(f"Gold path executed ({len(gold_path)} steps), final_reward={final_reward}")
    print(f"Screenshots saved to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
