import os
from copy import deepcopy
from hashlib import md5

import gym
import numpy as np
import babyai_text
import string
from collections import deque

from PIL import Image, ImageDraw
from babyai.bot import Bot


def make_env(env_id, seed, idx, env_params):
    def thunk():
        env = gym.make(env_id, seed=seed + idx, **env_params)
        if hasattr(env, "seed"):
            env.seed(seed + idx)
        if hasattr(env, "action_space") and hasattr(env.action_space, "seed"):
            env.action_space.seed(seed + idx)
        if hasattr(env, "observation_space") and hasattr(env.observation_space, "seed"):
            env.observation_space.seed(seed + idx)

        env.metadata = env.metadata | {
            "process_index": idx,
            "run_index": 0
        }
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
        self.save_gifs = kwargs.get("save_gifs", False)
        self.run_dir = kwargs.get("run_dir", "/tmp/adrl")

        print("env_id: ", env_id)
        if not isinstance(self.envs.single_action_space, gym.spaces.Discrete):
            raise ValueError("BabyAITextEnv currently expects a discrete action space.")

    def reset(self):
        reset_out = self.envs.reset()
        for env in self.envs.envs:
            env.metadata['gold_path'] = gold_paths(env)
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
        pre_step_count = [self.envs.envs[i].step_count for i in range(self.num_envs)]
        pre_gold_steps = [len(self.envs.envs[i].metadata.get('gold_path', {}).get('steps', [])) for i in range(self.num_envs)]
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
                self.envs.envs[i].metadata['info_before_reset'] = {
                    'step_count': pre_step_count[i],
                    'gold_steps': pre_gold_steps[i],
                }
                self.obs_history[i].clear()
                self.action_history[i].clear()
                run_index = self.envs.envs[i].metadata.get('index_in_run', 0)
                self.envs.envs[i].metadata['index_in_run'] = run_index + 1
                self.envs.envs[i].metadata['gold_path'] = gold_paths(self.envs.envs[i])

        next_obs, ava = self.handle_obs(raw_next_obs)
        reward = np.repeat(np.asarray(reward)[:, None], self.num_agents, axis=1)
        done = np.repeat(done[:, None], self.num_agents, axis=1)

        if self.save_gifs:
            for i in range(self.num_envs):

                _saved_image_path, _binary_image = save_image(
                    self.envs.envs[i],
                    file_name_prefix='AUTO',
                    action=action[i],
                    return_binary=False,
                    reward=reward[i].item(),
                    screenshots_dir=str(self.run_dir / "screenshots")
                )

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



def save_image(
    env,
    file_name_prefix='',
    action=None,
    return_binary=False,
    save_file=True,
    reward=0,
    partial_reward=0,
    feedback=None,
    screenshots_dir=''
):
    img = env.render(mode='rgb_array')
    # Convert the numpy array to a PIL Image
    img_pil = Image.fromarray(img)
    scale = .8
    width = int(img_pil.size[0] * scale)
    height = int(img_pil.size[1] * scale)
    img_pil = img_pil.resize((width, height), Image.Resampling.LANCZOS)
    # Create a draw object
    draw = ImageDraw.Draw(img_pil)
    white_fill = (255, 255, 255)
    font_size = 12 * scale
    #green_fill = (0, 255, 0)
    # Add the step number to the image using Pillow (smooth anti-aliased text)
    mission_text = f"{env.mission}"  # Display step number
    draw.text((10, 0), mission_text, font_size=font_size, fill=white_fill)

    step_text = f"Step: {env.step_count}"  # Display step number
    draw.text((5, height * 0.9), step_text, font_size=font_size, fill=white_fill)  # Green text
    if env.step_count:
        rewards_text = (f"Reward: {reward:.4f}".rstrip('0').rstrip('.') + "(" +
                        f"{partial_reward:.4f}".rstrip('0').rstrip('.') + ")")  # Display rewards
        draw.text((width * .55, height * 0.9), rewards_text, font_size=font_size, fill=white_fill)  # Green text

    if action is not None:
        action_texts = ["turn_left", "turn_right", "go_forward", "pick_up", "drop", "toggle"]
        action_text = action_texts[action]  # Display step number
        draw.text((width * .25, height * .9), action_text, font_size=font_size, fill=white_fill)  # Green text

    if feedback is not None:
        draw.text((10, 10), feedback, font_size=font_size, fill=white_fill)  # Green text


    binary_output = img_pil if return_binary else None
    image_path = None
    if save_file:
        process_index = env.metadata.get('process_index', '')
        run_index = env.metadata.get('index_in_run', '')
        save_dir = f"{screenshots_dir}/env{process_index}_run{run_index}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if file_name_prefix == 'AUTO':
            seed = env.metadata.get('seed', '')
            file_name_prefix = f"env{process_index}_{env.spec.id}_run{run_index}_seed{seed}"
        image_path = f"{save_dir}/{file_name_prefix}_step{env.step_count}.png"
        img_pil.save(image_path)
    return image_path, binary_output


def check_config(n_rollout_threads, episode_length, cur_num_batch, num_mini_batch, gradient_cp_steps):
    batch_size = n_rollout_threads * episode_length * cur_num_batch
    # num_mini_batch is the number of mini batches to split per single batch into thus should multiply cur_num_batch
    num_mini_batch *= cur_num_batch

    assert batch_size >= num_mini_batch
    mini_batch_size = batch_size // num_mini_batch
    assert mini_batch_size > 0

    cp_batch_size = int(batch_size // gradient_cp_steps)
    assert cp_batch_size > 0
    print("all_ok")


def env_state_hash(env):
    inventory = ""
    if env.env.carrying:
        inventory = env.env.carrying.color + env.env.carrying.type
    return md5((env.mission + str(env.env.env) + inventory).encode()).hexdigest()


def gold_paths(env):
    actions = []
    game_states = []
    done = False
    target_cell = None
    try:
        env_copy = deepcopy(env)
        replay_data = env_copy.metadata.get('replay_data', {})
        steps = replay_data.get('steps', [])
        seed = env_copy.metadata['seed']
        env_copy.seed(seed)
        env_copy.reset(seed=seed)
        #process_index = env_copy.metadata.get('glam_process_index', '')
        #run_index = env_copy.metadata.get('index_in_run', '')
        #saved_image = save_image(env_copy, f"gold_env{process_index}_{env_copy.spec.id}_run{run_index}_seed{seed}")
        #print(f"After reset image: {saved_image=}")
        for step in steps:
            env_copy.step(step)
        bot = Bot(env_copy)
        action = None
        question_actions = []
        if env_copy.spec.id == "BabyAI-MixedTrainLocalGrounding-v0":
            valid_actions = env_copy.valid_actions()
            question_actions = [index for index, text in valid_actions.items() if "What is" in text]
        while not done:
            try:
                if len(question_actions) > 0:
                    action = question_actions.pop()
                else:
                    action = bot.replan(action)
                if str(action) == 'Actions.done':
                    break
                action_int = int(action)
                actions.append(action_int)
                obs, reward, done, truncated, info = env_copy.step(action_int)
                game_states.append(env_state_hash(env_copy))
                if done:
                    target_cell = [int(x) for x in env_copy.env.env.agent_pos]
            except Exception as e:
                print(f"Error in getting gold paths: {e}")
                break
    except Exception as e:
        print(f"Error in getting gold paths: {e}")

    return {
        "steps": actions,
        "states": game_states if done else [],
        "target_cell": target_cell
    }
