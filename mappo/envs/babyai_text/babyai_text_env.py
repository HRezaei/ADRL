import gym
import numpy as np
import babyai_text
import string
from collections import deque


def make_env(env_id, seed, idx, env_params):
    def thunk():
        env = gym.make(env_id, seed=seed, **env_params)
        if hasattr(env, "seed"):
            env.seed(seed + idx)
        if hasattr(env, "action_space") and hasattr(env.action_space, "seed"):
            env.action_space.seed(seed + idx)
        if hasattr(env, "observation_space") and hasattr(env.observation_space, "seed"):
            env.observation_space.seed(seed + idx)
        return env

    return thunk


class BabyAITextEnv:
    """Skeleton text-environment adapter for BabyAI-text style tasks."""

    def __init__(self, env_id, num_envs, seed, env_params=None, **kwargs) -> None:
        # Ensure BabyAI-text env registration side effects are loaded when installed.
        try:
            import babyai_text  # noqa: F401
        except ImportError:
            # The skeleton can still be imported before dependencies are installed.
            pass

        env_params = env_params or {}
        self.num_envs = num_envs
        self.num_agents = 1
        self.seed = seed
        self.envs = gym.vector.SyncVectorEnv(
            [make_env(env_id, seed, i, env_params) for i in range(num_envs)]
        )
        self.template2action = [{} for _ in range(num_envs)]
        self.available_actions = ["turn left","turn right","go forward","pick up","drop","toggle"]
        self.past_actions_indexing = False
        self.num_past_obs = kwargs.get("num_past_obs", 3)
        self.obs_history = [deque(maxlen=self.num_past_obs) for _ in range(num_envs)]
        self.action_history = [deque(maxlen=self.num_past_obs-1) for _ in range(num_envs)]
        self.prompt_question_mark = kwargs.get("prompt_question_mark", False)

        print("env_id: ", env_id)
        if not isinstance(self.envs.single_action_space, gym.spaces.Discrete):
            raise ValueError("BabyAITextEnv currently expects a discrete action space.")

    def reset(self):
        reset_out = self.envs.reset()
        self.obs_history = [deque(maxlen=self.num_past_obs) for _ in range(self.num_envs)]
        self.action_history = [deque(maxlen=self.num_past_obs-1) for _ in range(self.num_envs)]
        if isinstance(reset_out, tuple):
            raw_obs = reset_out[0] | reset_out[1]
        else:
            raw_obs = reset_out
        obs, ava = self.handle_obs(raw_obs)
        return obs, ava

    def step(self, ori_action):
        action, action_texts = self.handle_action(ori_action)
        for i in range(self.num_envs):
            self.action_history[i].append(action_texts[i])
        step_out = self.envs.step(action)

        if len(step_out) == 5:
            raw_next_obs, reward, terminated, truncated, info = step_out
            done = np.logical_or(terminated, truncated)
            raw_next_obs = raw_next_obs | info
        elif len(step_out) == 4:
            raw_next_obs, reward, done, info = step_out
            raw_next_obs = raw_next_obs | info
        else:
            raise ValueError("Unexpected env step output format.")

        done = np.asarray(done)
        for i in range(self.num_envs):
            if done[i]:
                self.obs_history[i].clear()
                self.action_history[i].clear()

        next_obs, ava = self.handle_obs(raw_next_obs)
        reward = np.repeat(np.asarray(reward)[:, None], self.num_agents, axis=1)
        done = np.repeat(done[:, None], self.num_agents, axis=1)

        return next_obs, reward, done, ava, info

    def handle_action(self, ori_action):
        action = np.zeros((self.num_envs,), dtype=np.int64)
        action_texts = [""] * self.num_envs
        for i in range(self.num_envs):
            raw = ori_action[i][0]
            if isinstance(raw, (int, np.integer)):
                action_idx = int(raw)
                action[i] = action_idx
                if 0 <= action_idx < len(self.available_actions):
                    action_texts[i] = self.available_actions[action_idx]
                else:
                    action_texts[i] = str(action_idx)
                continue

            if raw not in self.template2action[i]:
                raise KeyError(
                    f"Action '{raw}' not in available action template for env {i}. "
                    "Update `extract_available_actions` for your BabyAI-text observation format."
                )
            action[i] = self.template2action[i][raw]
            action_texts[i] = str(raw)
        return action, action_texts

    def handle_obs(self, raw_obs):
        obs = np.empty((self.num_envs, self.num_agents), dtype=np.object_)
        ava = np.empty_like(obs, dtype=np.object_)

        for i in range(self.num_envs):
            goal = self.envs.envs[i].mission
            step_count = self.envs.envs[i].step_count
            sample_obs = raw_obs["descriptions"][i]
            self.obs_history[i].append(sample_obs)
            prompt = self.generate_prompt(
                goal,
                self.available_actions,
                self.obs_history[i],
                self.action_history[i],
                step_count=step_count,
            )

            obs[i, 0] = prompt
            ava[i, 0] = ",".join(self.available_actions)
            self.template2action[i] = {k: idx for idx, k in enumerate(self.available_actions)}

        return obs, ava

    def extract_prompt(self, sample_obs):
        if isinstance(sample_obs, dict):
            for key in ("prompt", "mission", "observation", "obs", "text"):
                if key in sample_obs:
                    return str(sample_obs[key])
        return str(sample_obs)

    def generate_prompt(self, goal, subgoals, deque_obs, deque_actions, step_count=0):
        ldo = len(deque_obs)
        lda = len(deque_actions)

        head_prompt = "Possible action of the agent:"
        for sg in subgoals:
            head_prompt += " {},".format(sg)
        head_prompt = head_prompt[:-1]

        g = " \n Goal of the agent: {}".format(goal)
        obs = ""
        start_index = step_count - lda if (self.past_actions_indexing and step_count > 0) else 0
        for i in range(ldo):
            obs += " \n Observation {}: ".format(i + start_index)
            current_obs = deque_obs[i]
            if isinstance(current_obs, str):
                current_obs = [current_obs]
            for d_obs in current_obs:
                obs += ("{} " if d_obs[-1] in string.punctuation else "{}, ").format(d_obs)
            obs += "\n Action {}: ".format(i + start_index)
            if i < lda:
                obs += "{}".format(deque_actions[i])
        return head_prompt + g + obs + ("?" if self.prompt_question_mark else "")

    def extract_available_actions(self, sample_obs):
        if isinstance(sample_obs, dict):
            for key in ("available_actions", "avaliable_action", "admissible_commands", "actions"):
                if key in sample_obs and sample_obs[key] is not None:
                    return [str(x) for x in sample_obs[key]]
        return []

    def close(self):
        self.envs.close()
