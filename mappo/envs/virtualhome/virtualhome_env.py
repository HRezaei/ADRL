import os
import virtual_home
import gym
import numpy as np
from pathlib import Path
from PIL import Image
from mappo.envs.virtualhome.virtualhome_utils import init_env
from mappo.envs.virtualhome.virtualhome_variants import init_variant_env
from mappo.envs.virtualhome.virtualhome_render import render_v1, render_v2

def make_env(env_id, seed, idx, env_params):
    def thunk():

        env = gym.make(env_id, **env_params)
        return env

    return thunk

class VirtualHomeEnv:
    
    def __init__(self, env_id, num_envs, seed, variant=None, save_gifs=False, run_dir=None) -> None:
        env_params = {'seed': seed, 'debug': False}
        self.num_envs = num_envs
        self.num_agents = 1
        self.env_id = env_id
        self.envs = gym.vector.SyncVectorEnv([make_env(env_id, seed + i, i, env_params) for i in range(num_envs)])
        print("env_id: ", env_id)
        
        assert isinstance(self.envs.single_action_space, gym.spaces.Discrete)
        
        if variant is None:
            self.action_template, self.obs2text = init_env(env_id=env_id)
        else:
            self.action_template, self.obs2text = init_variant_env(env_id=env_id, variant=variant)

        self.template2action = {
            k:i for i,k in enumerate(self.action_template)
        }

        if env_id == "VirtualHome-v1":
            self._render_fn = render_v1
        else:
            self._render_fn = render_v2

        self.save_gifs = save_gifs
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self._step_counts = np.zeros(num_envs, dtype=int)
        self._episode_counts = np.zeros(num_envs, dtype=int)
        
    def reset(self):
        outcome = self.envs.reset()
        if isinstance(outcome, tuple):
            ori_obs, empty_infos = outcome
        else:
            ori_obs = outcome
        obs, ava = self.handle_obs(ori_obs)

        return obs, ava
        
    def _get_graph_state(self, env_idx):
        try:
            vh_env = self.envs.envs[env_idx].env
            return vh_env.get_observations()
        except Exception:
            return None
        
    def step(self, ori_action):
        action = self.handle_action(ori_action)
        outcome = self.envs.step(action)
        if len(outcome) == 5:
            ori_next_obs, reward, done, truncated, info = outcome
        else:
            ori_next_obs, reward, done, info = outcome
        next_obs, ava = self.handle_obs(ori_next_obs)
        reward = np.repeat(reward[:, None], self.num_agents, axis=1)
        done = np.repeat(done[:, None], self.num_agents, axis=1)

        if self.save_gifs:
            for i in range(self.num_envs):
                screenshots_dir = str(self.run_dir / "screenshots") if self.run_dir else "/tmp/adrl/screenshots"
                env_dir = f"{screenshots_dir}/env{i}"
                os.makedirs(env_dir, exist_ok=True)
                step_base = f"{env_dir}/ep{self._episode_counts[i]}_step{self._step_counts[i]}"
                # PNG visualization
                graph = self._get_graph_state(i)
                if graph is not None:
                    try:
                        img_arr = self._render_fn(graph, action_text=ori_action[i][0])
                        Image.fromarray(img_arr).save(step_base + ".png")
                    except Exception:
                        pass
                # Text trace for comparison
                try:
                    with open(step_base + ".txt", "w") as f:
                        f.write(f"Step: {self._step_counts[i]}\n")
                        f.write(f"Action: {ori_action[i][0]}\n")
                        f.write(f"Observation: {next_obs[i, 0]}\n")
                        f.write(f"Available: {ava[i, 0]}\n")
                        f.write(f"Reward: {reward[i, 0]}\n")
                except Exception:
                    pass
                if done[i, 0]:
                    self._episode_counts[i] += 1
                    self._step_counts[i] = 0
                else:
                    self._step_counts[i] += 1

        return next_obs, reward, done, ava, info
    
    def handle_action(self, ori_action):
        action = np.zeros((self.num_envs,), dtype=np.int64)
        for i in range(self.num_envs):
            action[i] = self.template2action[ori_action[i][0]]
        return action
    
    def handle_obs(self, ori_obs):
        obs = np.empty((self.num_envs, self.num_agents), dtype=np.object_)
        ava = np.empty_like(obs, dtype=np.object_)
        for i in range(self.num_envs):
            text_obs = self.obs2text(ori_obs[i], self.action_template)
            obs[i, 0] = text_obs["prompt"]
            ava[i, 0] = ",".join(text_obs["avaliable_action"])
        return obs, ava
    
    def close(self):
        self.envs.close()
