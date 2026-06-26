"""Verify that BabyAI Text grid layouts can be recreated from seed + reset count.

Key insight: each reset() and __init__ both call gen_mission(), consuming
one RNG state each. To recreate the Nth reset's layout, you need N+1 resets
after reseeding (N resets to replay + 1 to account for init's gen_mission call).
"""

import gym
import babyai_text


def capture(env, label=""):
    return str(env), env.mission


def recreate(seed, num_resets):
    """Recreate the layout that appears after `num_resets` resets from seed."""
    env = gym.make("BabyAI-MixedTrainLocal-v0", seed=seed)
    env.seed(seed)
    for _ in range(num_resets + 1):
        env.reset()
    grid, mission = str(env), env.mission
    env.close()
    return grid, mission


def test():
    passed = 0
    failed = 0

    for seed in [1, 42, 123, 999]:
        for target_resets in [1, 2, 3, 5]:
            original = gym.make("BabyAI-MixedTrainLocal-v0", seed=seed)
            for _ in range(target_resets):
                original.reset()
            orig_grid, orig_mission = capture(original)
            original.close()

            replay_grid, replay_mission = recreate(seed, target_resets)

            match = orig_grid == replay_grid and orig_mission == replay_mission
            status = "PASS" if match else "FAIL"
            if match:
                passed += 1
            else:
                failed += 1
            print(
                f"[{status}] seed={seed}, resets={target_resets}  "
                f"mission={orig_mission!r}  match={match}"
            )

    print(f"\n{passed}/{passed+failed} passed")
    return failed == 0


if __name__ == "__main__":
    test()
