from tqdm import tqdm

from mappo.runner.shared.virtualhome_runner import VirtualHomeRunner


class BabyAITextRunner(VirtualHomeRunner):
    game_name = "babyai"
    def run(self):

        obs, ava = self.envs.reset()
        self.buffer.obs[self.buffer.cur_batch_index, 0] = obs.copy()
        self.buffer.available_actions[self.buffer.cur_batch_index, 0] = ava.copy()

        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        total_num_steps = 0
        for episode in tqdm(range(episodes), desc="episodes"):
            finished_rewards = []
            finished_waste = []
            for step in tqdm(range(self.episode_length), desc="steps"):
                # Sample actions
                values, actions, action_tokens, log_probs = self.collect(step)

                # Obser reward and next obs
                obs, rewards, dones, ava, infos = self.envs.step(actions)

                for i in range(self.n_rollout_threads):
                    if dones[i] or self.envs.envs.envs[i].steps_remaining == 0:
                        finished_rewards.append(rewards[i])
                        info = self.envs.envs.envs[i].metadata.get('info_before_reset', {})
                        gold_steps = info.get('gold_steps', 0)
                        actual_steps = info.get('step_count', 0)
                        finished_waste.append(actual_steps - gold_steps)
                        global_step = total_num_steps + step * self.n_rollout_threads + i
                        print(
                            f"global_step={global_step}, episodic_return={rewards[i]}, episodic_length={0}, waste_frames={actual_steps - gold_steps}")
                        self.writter.add_scalar("charts/episodic_return", rewards[i], global_step)
                        self.writter.add_scalar("charts/episodic_length", 0, global_step)
                        #self.writter.add_scalar("charts/waste_frames", actual_steps - gold_steps, global_step)
                        #break Original virtualhome runner has break here! I don't know why?!

                # insert data into buffer
                data = obs, rewards, dones, ava, values, \
                    actions, action_tokens, log_probs
                self.insert(data)

            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            success_per_episode =  [1 if r > 0 else 0 for r in finished_rewards]
            # compute return and update network
            self.before_update()
            # self.trainer.prep_training()
            train_infos = self.trainer.train(self.buffer)
            self.buffer.after_update()

            train_infos["success_rate"] = sum(success_per_episode) / len(success_per_episode)
            if finished_waste:
                train_infos["waste_frames"] = sum(finished_waste)
            # save model
            if (episode == episodes - 1):
                self.save(episode)

            # log information
            if episode % self.log_interval == 0:
                print("total_num_steps: ", total_num_steps)
                self.log_train(train_infos, total_num_steps)
