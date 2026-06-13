#!/usr/bin/env python3
import os
import sys
import argparse
from PIL import Image, ImageDraw, ImageFont

import gym
import gym_macro_overcooked

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mappo.envs.overcooked.planner import OvercookedPlanner
from mappo.envs.overcooked.overcooked_env import TASKLIST, REWARDLIST
from gym_macro_overcooked.macActEnvWrapper import MacEnvWrapper


def make_env(env_id, task_name):
    env = gym.make(
        env_id,
        grid_dim=[7, 7],
        task=task_name,
        rewardList=REWARDLIST,
        map_type="A",
        n_agent=1,
        obs_radius=2,
        mode="vector",
        debug=True,
    )
    return MacEnvWrapper(env)


def annotate(img, lines):
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", 14)
    except (OSError, IOError):
        font = ImageFont.load_default()
    for i, line in enumerate(lines):
        draw.text((5, 5 + i * 20), line, fill=(255, 255, 255), font=font)


def main():
    parser = argparse.ArgumentParser(
        description="Step through planner plans and save annotated screenshots."
    )
    parser.add_argument(
        "--env-id",
        default="Overcooked-LLMA-v3",
        choices=["Overcooked-LLMA-v3", "Overcooked-LLMA-v4"],
    )
    parser.add_argument("--tasks", nargs="*", default=None,
                        help="Tasks to run (default: all)")
    parser.add_argument("--out", default="results/planner_frames")
    args = parser.parse_args()

    tasks = args.tasks if args.tasks else TASKLIST
    os.makedirs(args.out, exist_ok=True)
    print(f"Output dir: {args.out}")

    for task_name in tasks:
        print(f"\n{'=' * 60}")
        print(f"Task: {task_name}")
        print(f"{'=' * 60}")

        env = make_env(args.env_id, task_name)
        raw = env.env
        planner = OvercookedPlanner(raw)

        env.reset()
        plan = planner.plan()
        plan_names = planner.get_action_names(plan)

        task_dir = os.path.join(args.out, task_name.replace(" ", "_"))
        os.makedirs(task_dir, exist_ok=True)

        if plan is None:
            img = Image.fromarray(raw.render())
            annotate(img, [f"Task: {task_name}", "No plan found"])
            img.save(os.path.join(task_dir, "no_plan.png"))
            print("  No plan found.")
            env.close()
            continue

        print(f"  Plan ({len(plan)} steps):")
        for i, (idx, name) in enumerate(zip(plan, plan_names)):
            print(f"    {i + 1}. [{idx}] {name}")

        for step_idx, (action_idx, action_name) in enumerate(zip(plan, plan_names)):
            frame = raw.render()
            img = Image.fromarray(frame)
            annotate(img, [
                f"Task: {task_name}",
                f"Step {step_idx + 1} / {len(plan)}: {action_name}",
            ])
            out_path = os.path.join(
                task_dir, f"step_{step_idx + 1:03d}_{action_name.replace(' ', '_')}.png"
            )
            img.save(out_path)

            obs, reward, done, info = raw.run(action_idx)

        frame = raw.render()
        img = Image.fromarray(frame)
        annotate(img, [
            f"Task: {task_name}",
            f"Step {len(plan) + 1} / {len(plan)}: done",
        ])
        out_path = os.path.join(task_dir, f"step_{len(plan) + 1:03d}_done.png")
        img.save(out_path)

        print(f"  Frames saved to {task_dir}")
        env.close()

    print("\nAll done.")


if __name__ == "__main__":
    main()
