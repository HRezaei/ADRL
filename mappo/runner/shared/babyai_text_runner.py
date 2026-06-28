import json
import time

import numpy as np
import torch
from tqdm import tqdm

from mappo.runner.shared.virtualhome_runner import VirtualHomeRunner


class BabyAITextRunner(VirtualHomeRunner):
    game_name = "babyai"

    def __init__(self, config):
        super().__init__(config)
        self.use_planner = getattr(self.all_args, "use_planner", 0)
        self.save_json_interval = getattr(self.all_args, "save_json_interval", 1)
        self.games_json_path = str(self.run_dir / "games.json")
        self.games_record = {i: [] for i in range(self.n_rollout_threads)}
        self._current_game = [None] * self.n_rollout_threads
        self._env_seeds = [self.all_args.seed + i for i in range(self.n_rollout_threads)]

    def run(self):

        obs, ava = self.envs.reset()
        self.buffer.obs[self.buffer.cur_batch_index, 0] = obs.copy()
        self.buffer.available_actions[self.buffer.cur_batch_index, 0] = ava.copy()
        self.plan_step = np.zeros(self.n_rollout_threads, dtype=np.int64)
        for i in range(self.n_rollout_threads):
            self._start_new_game(i)

        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        total_num_steps = 0
        for episode in tqdm(range(episodes), desc="episodes"):

            update_start_time = time.time()
            collect_logs = self.collect_experiences(episode)

            # compute return and update network
            self.before_update()
            # self.trainer.prep_training()
            if self.all_args.skip_updating_model:
                train_infos = {"value_loss": 0.0, "value_grad_norm": 0.0, "policy_loss": 0.0, "policy_grad_norm": 0.0}
            else:
                train_infos = self.trainer.train(self.buffer)
            self.buffer.after_update()

            update_end_time = time.time()
            collect_logs['fps'] = collect_logs["num_frames"] / (update_end_time - update_start_time)

            train_infos = train_infos | collect_logs
            # save model
            if (episode == episodes - 1):
                self.save(episode)

            # log information
            total_num_steps += collect_logs["num_frames"]
            if episode % self.log_interval == 0:
                print("total_num_steps: ", total_num_steps)
                self.log_train(train_infos, total_num_steps)

            if episode % self.save_json_interval == 0:
                self._flush_games_json()

        self._flush_games_json()

    @torch.no_grad()
    def collect(self, step):
        obs_concat = np.concatenate(self.buffer.obs[self.buffer.cur_batch_index, step])
        ava_concat = np.concatenate(self.buffer.available_actions[self.buffer.cur_batch_index, step])

        teacher_actions = None
        if self.use_planner:
            teacher_actions = np.empty(self.n_rollout_threads, dtype=object)
            for i in range(self.n_rollout_threads):
                gold_path = self.envs.envs.envs[i].metadata.get("gold_path", {})
                steps = gold_path.get("steps", [])
                step_idx = int(self.plan_step[i])
                if step_idx < len(steps):
                    teacher_actions[i] = self.envs.available_actions[int(steps[step_idx])]
                else:
                    teacher_actions = None
                    break

        behaviour_data = self.agent.infer_for_rollout(obs_concat, ava_concat, actions=teacher_actions)
        actions, action_tokens, values, log_probs = behaviour_data

        values = np.array(np.split(values, self.n_rollout_threads))
        actions = np.array(np.split(actions, self.n_rollout_threads))
        action_tokens = np.array(np.split(action_tokens, self.n_rollout_threads))
        log_probs = np.array(np.split(log_probs, self.n_rollout_threads))

        return values, actions, action_tokens, log_probs

    def _gold_path_to_strings(self, i):
        gold_path = self.envs.envs.envs[i].metadata.get("gold_path", {})
        return [int(s) for s in gold_path.get("steps", [])]

    def _start_new_game(self, i):
        self._current_game[i] = {
            "seed": self._env_seeds[i],
            "goal": self.envs.envs.envs[i].mission,
            "index_in_run": self.envs.envs.envs[i].metadata.get("index_in_run", 0),
            "gold_actions": self._gold_path_to_strings(i),
            "actions": [],
        }

    def _finish_game(self, i, won):
        game = self._current_game[i]
        if game is None:
            return
        self.games_record[i].append({
            "seed": game["seed"],
            "goal": game["goal"],
            "index_in_run": game["index_in_run"],
            "won": won,
            "actions": game["actions"],
            "gold_actions": game["gold_actions"],
        })
        self._current_game[i] = None

    def _flush_games_json(self):
        with open(self.games_json_path, "w") as f:
            json.dump(self.games_record, f, indent=2, default=str)

    def collect_experiences(self, episode):
        finished_rewards = []
        log_waste_frames = []
        completed_frames = []
        log_frames_to_win = []
        log_waste_frames_to_win = []
        num_frames = self.episode_length * self.n_rollout_threads
        total_num_steps = (episode + 1) * num_frames
        games_done = 0
        for step in tqdm(range(self.episode_length), desc="steps"):
            # Sample actions
            values, actions, action_tokens, log_probs = self.collect(step)

            for i in range(self.n_rollout_threads):
                if self._current_game[i] is not None:
                    a = actions[i, 0]
                    if isinstance(a, str):
                        a = self.envs.available_actions.index(a)
                    self._current_game[i]["actions"].append(int(a))

            # Obser reward and next obs
            obs, rewards, dones, ava, infos = self.envs.step(actions)

            if self.use_planner:
                for i in range(self.n_rollout_threads):
                    if dones[i] or self.envs.envs.envs[i].steps_remaining == 0:
                        self.plan_step[i] = 0
                    else:
                        self.plan_step[i] += 1

            for i in range(self.n_rollout_threads):
                if dones[i] or self.envs.envs.envs[i].steps_remaining == 0:
                    games_done += 1
                    self._finish_game(i, bool(rewards[i].item() > 0))
                    self._start_new_game(i)
                    finished_reward = rewards[i]
                    finished_rewards.append(finished_reward)
                    info = self.envs.envs.envs[i].metadata.get('info_before_reset', {})
                    gold_steps = info.get('gold_steps', 0)
                    actual_steps = info.get('step_count', 0)
                    completed_frames.append(actual_steps)
                    waste_frames = actual_steps
                    # All played frames are wasted, unless won case, in which gold_steps are subtracted
                    if finished_reward > 0:
                        waste_frames -= gold_steps
                        log_frames_to_win.append(actual_steps)
                        log_waste_frames_to_win.append(waste_frames)
                    log_waste_frames.append(waste_frames)
                    #self.envs.envs.envs[i].max_steps = 40
                    global_step = total_num_steps + step * self.n_rollout_threads + i
                    print(
                        f"global_step={global_step}, episodic_return={rewards[i]}, episodic_length={0}, waste_frames={actual_steps - gold_steps}")
                    self.writter.add_scalar("charts/episodic_return", rewards[i], global_step)
                    self.writter.add_scalar("charts/episodic_length", 0, global_step)
                    # self.writter.add_scalar("charts/waste_frames", actual_steps - gold_steps, global_step)
                    # break Original virtualhome runner has break here! I don't know why?!

            # insert data into buffer
            data = obs, rewards, dones, ava, values, \
                actions, action_tokens, log_probs
            self.insert(data)

        success_per_episode =  [1 if r > 0 else 0 for r in finished_rewards]

        log = {
            "success_rate": sum(success_per_episode) / len(success_per_episode) if len(success_per_episode)>0 else 0,
            "num_frames": num_frames,
            "episodes_done": games_done,
            "mean_waste_frames": np.mean(log_waste_frames) if len(log_waste_frames) > 0 else float('nan'),
            "mean_frames_to_win": np.mean(log_frames_to_win) if len(log_frames_to_win) > 0 else float('nan'),
            "mean_waste_frames_to_win": np.mean(log_waste_frames_to_win) if len(log_waste_frames_to_win) > 0 else float(
                'nan'),
            "completed_frames": np.sum(completed_frames),
            "waste_frames": np.sum(log_waste_frames) if len(log_waste_frames) > 0 else float('nan'),
            "frames_to_win": np.sum(log_frames_to_win) if len(log_frames_to_win) > 0 else float('nan'),
            "waste_frames_to_win": np.sum(log_waste_frames_to_win) if len(log_waste_frames_to_win) > 0 else float(
                'nan'),
        }

        sum_completed_frames = log["completed_frames"]
        log["waste_frames_percentage"] = (log["waste_frames"] / sum_completed_frames) if sum_completed_frames > 0 else float('nan')
        log["waste_frames_to_win_percentage"] = (log["waste_frames_to_win"] / log["frames_to_win"]) if log["frames_to_win"] > 0 else float('nan')

        return log
