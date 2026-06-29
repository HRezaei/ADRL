import time
import os
import uuid
from math import nan

import numpy as np
from functools import reduce
import torch
import wandb
from tensorboardX import SummaryWriter
from mappo.models.codellama import Llama
from mappo.agents.llama_lora_agent import LlamaLoRAgent
from mappo.agents.llama_full_agent import LlamaFullAgent
from mappo.agents.causal_full_agent import CausalFullAgent
from mappo.agents.causal_lora_agent import CausalLoRAgent
from mappo.agents.seq2seq_full_agent import Seq2SeqFullAgent
from mappo.agents.seq2seq_lora_agent import Seq2SeqLoRAgent
from mappo.utils.language_buffer import LanguageBuffer
from mappo.trainers.llm_trainer_appo import APPOTrainer
from mappo.trainers.llm_trainer_tppo import TPPOTrainer
from mappo.utils.distributed import is_distributed, get_local_rank, fsdp_wrap, full_state_dict
import torch.distributed as dist
import pickle
from mappo.envs.datascience.prompts.scikit_prompts import *
import json

def _t2n(x):
    return x.detach().cpu().numpy()

def cal_token_mask(action_tokens_batch, pad_token):
    token_mask = (action_tokens_batch != pad_token).astype(np.int64)
    return token_mask

class VirtualHomeRunner:
    game_name = "virtualhome"
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""
    def __init__(self, config):
        self.num_agents = config['num_agents']
        self.all_args = config['all_args']
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.log_interval = self.all_args.log_interval
        self.algo = self.all_args.algorithm_name
        self.use_full_scale = self.all_args.use_full_scale
        self.local_rank = config.get("local_rank", 0)
        self.rank = self.local_rank

        self.run_dir = config["run_dir"]
        self.log_dir = str(self.run_dir / 'logs')
        if self.rank == 0 and not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        if self.rank == 0:
            wandb.tensorboard.patch(root_logdir=self.log_dir)
            config_for_wandb = config.copy()
            config_for_wandb["all_args"] = vars(config_for_wandb["all_args"])
            del config_for_wandb["envs"]
            del config_for_wandb["eval_envs"]
            model_short = os.path.basename(self.all_args.model_name) if self.all_args.model_name else "unknown"
            scale_tag = "full" if self.all_args.use_full_scale else "lora"
            wandb.init(
                project="adrl",
                #sync_tensorboard=True,
                settings=wandb.Settings(_service_wait=300, code_dir="./mappo"),
                config=config_for_wandb,
                name=f"{self.all_args.experiment_name}_{self.game_name}_{model_short}_{scale_tag}_{uuid.uuid4().hex[:8]}",
            )
            self.writter = SummaryWriter(self.log_dir)
        else:
            self.writter = None
        self.save_dir = str(self.run_dir / 'models/')
        if self.rank == 0 and not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.envs = config['envs']
        self.eval_envs = config['eval_envs']

        # Select agent class via config strings (with defaults)
        full_class_name = getattr(self.all_args, "llm_class_full", "LlamaFullAgent")
        lora_class_name = getattr(self.all_args, "llm_class_lora", "LlamaLoRAgent")
        target_class_name = full_class_name if self.use_full_scale else lora_class_name

        agent_cls = globals().get(target_class_name)
        if agent_cls is None:
            raise ValueError(f"Configured LLM agent class not found: {target_class_name}")

        self.agent = agent_cls(self.all_args.model_name, self.all_args.max_new_tokens, self.algo, local_rank=self.local_rank)

        # FSDP-wrap actor for full-scale distributed training
        # Critic is not wrapped (its transformer is frozen with no_grad)
        if is_distributed() and self.use_full_scale:
            # Break tied weights (e.g. T5 shared embedding / lm_head) before FSDP,
            # since FSDP's parameter flattening cannot handle shared tensors.
            if hasattr(self.agent.actor, 'lm_head') and hasattr(self.agent.actor, 'shared'):
                if self.agent.actor.lm_head.weight is self.agent.actor.shared.weight:
                    self.agent.actor.lm_head.weight = torch.nn.Parameter(
                        self.agent.actor.lm_head.weight.clone()
                    )
            self.agent.actor = fsdp_wrap(self.agent.actor, device_id=self.local_rank)
            self.agent.critic = self.agent.critic.to(self.agent.device)
        else:
            self.agent.actor = self.agent.actor.to(self.agent.device)
            self.agent.critic = self.agent.critic.to(self.agent.device)
        if self.rank == 0:
            model = getattr(self.agent, 'actor', self.agent.base_model)
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            wandb.config.update({
                "model/total_params": total_params,
                "model/trainable_params": trainable_params,
                "model/trainable_pct": 100.0 * trainable_params / total_params if total_params > 0 else 0,
            })
        self.buffer = LanguageBuffer(self.all_args, self.num_agents, self.agent.tokenizer.pad_token_id)
        

        if self.algo == "TWOSOME":
            self.trainer = APPOTrainer(self.all_args, self.agent, self.num_agents)
        elif self.algo in ["POAD", "NTPO", "ARCHER"]:
            self.trainer = TPPOTrainer(self.all_args, self.agent, self.num_agents)
        else:
            raise NotImplementedError
        
        self.trajectories = None
        

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

                for i in range(self.n_rollout_threads):
                    if "episode" in infos[i].keys():
                        finished_rewards.append(infos[i]["episode"]["r"])

                for i in range(self.n_rollout_threads):
                    if "episode" in infos[i].keys():
                        global_step = total_num_steps + step * self.n_rollout_threads + i
                        print(f"global_step={global_step}, episodic_return={infos[i]['episode']['r']}, episodic_length={infos[i]['episode']['l']}")
                        if self.writter is not None:
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
            if self.all_args.skip_updating_model:
                train_infos = {"value_loss": 0.0, "value_grad_norm": 0.0, "policy_loss": 0.0, "policy_grad_norm": 0.0}
            else:
                train_infos = self.trainer.train(self.buffer)      
            self.buffer.after_update()

            success_per_episode =  [1 if r > 0 else 0 for r in finished_rewards]
            train_infos["success_rate"] = sum(success_per_episode) / len(success_per_episode) if len(success_per_episode) > 0 else 0

            # save model
            if episode % self.all_args.model_save_interval == 0 or episode == episodes - 1:
                self.save(episode)

            # log information
            if episode % self.log_interval == 0:
                print("total_num_steps: ", total_num_steps)
                self.log_train(train_infos, total_num_steps)
        

    @torch.no_grad()
    def collect(self, step):
        # self.trainer.prep_rollout()
        
        behaviour_data = self.agent.infer_for_rollout(np.concatenate(self.buffer.obs[self.buffer.cur_batch_index, step]),
                                                np.concatenate(self.buffer.available_actions[self.buffer.cur_batch_index, step]))
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
        elif self.algo in ["POAD", "NTPO", "ARCHER"]:
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
        elif self.algo in ["POAD", "NTPO", "ARCHER"]:
            next_values = self.agent.get_next_values(np.concatenate(self.buffer.obs[self.buffer.cur_batch_index, -1]))
            next_values = np.array(np.split(next_values, self.n_rollout_threads))
            self.buffer.batch_process_tppo(next_values)
        else:
            raise NotImplementedError

    def log_train(self, train_infos, total_num_steps):
        train_infos["average_step_rewards"] = np.mean(self.buffer.rewards[self.buffer.cur_batch_index])
        if self.writter is None:
            return
        for k, v in train_infos.items():
            self.writter.add_scalars(k, {k: v}, total_num_steps)
                
    def save(self, episode):
        """Save policy's actor and critic networks."""
        if is_distributed() and self.use_full_scale:
            exp_path = os.path.join(self.save_dir, "episode_{:04d}".format(episode))
            with full_state_dict(self.agent.actor):
                state_dict = self.agent.actor.state_dict()
                if self.rank == 0:
                    os.makedirs(exp_path, exist_ok=True)
                    base_model = getattr(self.agent, 'base_model', None)
                    if base_model is not None and hasattr(base_model, 'config'):
                        base_model.config.save_pretrained(exp_path)
                    torch.save(state_dict, os.path.join(exp_path, "pytorch_model.bin"))
            dist.barrier()
        else:
            self.agent.save(self.save_dir, episode)
