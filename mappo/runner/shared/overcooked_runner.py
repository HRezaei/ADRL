import time
import os
import uuid
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from functools import reduce
import torch
import wandb
from tensorboardX import SummaryWriter
from mappo.models.codellama import Llama
from mappo.agents.llama_lora_agent import LlamaLoRAgent
from mappo.utils.language_buffer import LanguageBuffer
from mappo.trainers.llm_trainer_appo import APPOTrainer
from mappo.trainers.llm_trainer_tppo import TPPOTrainer
import pickle
from mappo.envs.datascience.prompts.scikit_prompts import *
import json

def _t2n(x):
    return x.detach().cpu().numpy()

def cal_token_mask(action_tokens_batch, pad_token):
    token_mask = (action_tokens_batch != pad_token).astype(np.int64)
    return token_mask

class OvercookedRunner:
    game_name = "overcooked"
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""
    def __init__(self, config):
        self.num_agents = config['num_agents']
        self.all_args = config['all_args']
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.log_interval = self.all_args.log_interval
        self.algo = self.all_args.algorithm_name
        self.use_planner = getattr(self.all_args, 'use_planner', 0)

        self.run_dir = config["run_dir"]
        self.log_dir = str(self.run_dir / 'logs')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        config_for_wandb = config.copy()
        config_for_wandb["all_args"] = vars(config_for_wandb["all_args"])
        config_for_wandb.pop("envs", None)
        config_for_wandb.pop("eval_envs", None)
        wandb.init(
            project="adrl",
            sync_tensorboard=True,
            settings=wandb.Settings(_service_wait=300, code_dir="./mappo"),
            config=config_for_wandb,
            name=f"{self.game_name}_{uuid.uuid4().hex[:8]}",
        )
        self.writter = SummaryWriter(self.log_dir)
        self.save_dir = str(self.run_dir / 'models/')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.save_gifs = self.all_args.save_gifs
        if self.save_gifs:
            self.gif_dir = str(self.run_dir / 'screenshots')
            os.makedirs(self.gif_dir, exist_ok=True)
        self.agent = LlamaLoRAgent(self.all_args.model_name, self.all_args.max_new_tokens, self.algo)
        self.buffer = LanguageBuffer(self.all_args, self.num_agents, self.agent.tokenizer.pad_token_id)
        
        if self.algo == "TWOSOME":
            self.trainer = APPOTrainer(self.all_args, self.agent, self.num_agents)
        elif self.algo in ["POAD", "NTPO"]:
            self.trainer = TPPOTrainer(self.all_args, self.agent, self.num_agents)
        else:
            raise NotImplementedError
        
        self.trajectories = None
        

    def _save_frame(self, img_array, goal, action, save_path):
        img = Image.fromarray(img_array)
        banner_h = 52
        canvas = Image.new('RGB', (img.width, img.height + banner_h), (30, 30, 30))
        canvas.paste(img, (0, 0))
        draw = ImageDraw.Draw(canvas)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except Exception:
            font = ImageFont.load_default()
        draw.text((5, img.height + 4),  f"Goal:   {goal}",   fill=(255, 220, 50),  font=font)
        draw.text((5, img.height + 26), f"Action: {action}", fill=(255, 255, 255), font=font)
        canvas.save(save_path)

    def run(self):
        
        obs, ava = self.envs.reset()
        self.buffer.obs[self.buffer.cur_batch_index, 0] = obs.copy()
        self.buffer.available_actions[self.buffer.cur_batch_index, 0] = ava.copy()

        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        
        total_num_steps = 0
        for episode in range(episodes):
            finished_rewards = []
            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_tokens, log_probs = self.collect(step)

                # Obser reward and next obs
                obs, rewards, dones, ava, infos = self.envs.step(actions)
                if self.use_planner:
                    self.envs.advance_plan(dones)

                if self.save_gifs:
                    for i in range(self.n_rollout_threads):
                        img_array = self.envs.render(env_idx=i)
                        if img_array is not None:
                            ep_dir = os.path.join(self.gif_dir, f"env{i:02d}_ep{episode:04d}")
                            os.makedirs(ep_dir, exist_ok=True)
                            self._save_frame(
                                img_array,
                                goal=self.envs.task_name,
                                action=str(actions[i][0]),
                                save_path=os.path.join(ep_dir, f"step{step:04d}.png"),
                            )

                for i in range(self.n_rollout_threads):
                    if "episode" in infos[i].keys():
                        global_step = total_num_steps + step * self.n_rollout_threads + i
                        finished_rewards.append(infos[i]["episode"]["r"])
                        print(f"global_step={global_step}, episodic_return={infos[i]['episode']['r']}, episodic_length={infos[i]['episode']['l']}")
                        self.writter.add_scalar("charts/episodic_return", infos[i]["episode"]["r"], global_step)
                        self.writter.add_scalar("charts/episodic_length", infos[i]["episode"]["l"], global_step)
                        break
                
                # insert data into buffer
                data = obs, rewards, dones, ava, values, \
                       actions, action_tokens, log_probs
                self.insert(data)
                
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            # compute return and update network
            self.before_update()
            # self.trainer.prep_training()
            train_infos = self.trainer.train(self.buffer)
            success_per_episode = [1 if r > 0 else 0 for r in finished_rewards] if finished_rewards else [0]
            train_infos["success_rate"] = sum(success_per_episode) / len(success_per_episode)
            self.buffer.after_update()

            # log information
            if episode % self.log_interval == 0:
                print("total_num_steps: ", total_num_steps, ", success_rate: ", train_infos.get("success_rate", "N/A"))
                # print("average_step_rewards: ", np.mean(self.buffer.rewards[self.buffer.pre_batch_index]))
                self.log_train(train_infos, total_num_steps)
        

    @torch.no_grad()
    def collect(self, step):
        # self.trainer.prep_rollout()
        
        obs_concat = np.concatenate(self.buffer.obs[self.buffer.cur_batch_index, step])
        ava_concat = np.concatenate(self.buffer.available_actions[self.buffer.cur_batch_index, step])

        teacher_actions = None
        if self.use_planner:
            planner_out = self.envs.get_planner_actions(
                self.buffer.available_actions[self.buffer.cur_batch_index, step]
            )
            if planner_out is not None:
                teacher_actions = np.concatenate(planner_out)
        
        behaviour_data = self.agent.infer_for_rollout(obs_concat, ava_concat, actions=teacher_actions)
        actions, action_tokens, values, log_probs = behaviour_data
        
        # [self.envs, agents]
        values = np.array(np.split(values, self.n_rollout_threads))
        actions = np.array(np.split(actions, self.n_rollout_threads))
        action_tokens = np.array(np.split(action_tokens, self.n_rollout_threads))
        
        log_probs = np.array(np.split(log_probs, self.n_rollout_threads))
    
            
        return values, actions, action_tokens, log_probs

    def insert(self, data):
        obs, rewards, dones, ava, values, actions, action_tokens, log_probs = data
            
        dones_env = np.all(dones, axis=1)
        masks = np.ones((self.n_rollout_threads, self.num_agents), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents), dtype=np.float32)
        
        if self.algo == "TWOSOME":
            self.buffer.insert_appo(obs, ava, actions, values, rewards, masks, action_tokens, log_probs)
        elif self.algo in ["POAD", "NTPO"]:
            self.buffer.insert_tppo(obs, ava, actions, values, rewards, masks, action_tokens, log_probs)
        else:
            raise NotImplementedError

    @torch.no_grad()
    def before_update(self):
        """Calculate returns for the collected data."""
        if self.algo == "TWOSOME":
            next_values = self.agent.get_next_values(np.concatenate(self.buffer.obs[self.buffer.cur_batch_index, -1]))
            next_values = np.array(np.split(next_values, self.n_rollout_threads))
            self.buffer.batch_process_appo(next_values)
        elif self.algo in ["POAD", "NTPO"]:
            next_values = self.agent.get_next_values(np.concatenate(self.buffer.obs[self.buffer.cur_batch_index, -1]))
            next_values = np.array(np.split(next_values, self.n_rollout_threads))
            self.buffer.batch_process_tppo(next_values)
        else:
            raise NotImplementedError

    def log_train(self, train_infos, total_num_steps):
        train_infos["average_step_rewards"] = np.mean(self.buffer.rewards[self.buffer.cur_batch_index])
        for k, v in train_infos.items():
            # print("k: ", k, ", v: ", v)
            self.writter.add_scalars(k, {k: v}, total_num_steps)
                
    def save(self, episode):
        """Save policy's actor and critic networks."""
        self.agent.save(self.save_dir, episode)



