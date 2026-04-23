from tqdm import tqdm

from mappo.runner.shared.virtualhome_runner import VirtualHomeRunner


class BabyAITextRunner(VirtualHomeRunner):
    def run(self):

        obs, ava = self.envs.reset()
        self.buffer.obs[self.buffer.cur_batch_index, 0] = obs.copy()
        self.buffer.available_actions[self.buffer.cur_batch_index, 0] = ava.copy()

        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        total_num_steps = 0
        for episode in tqdm(range(episodes), desc="episodes"):
            for step in tqdm(range(self.episode_length), desc="steps"):
                # Sample actions
                values, actions, action_tokens, log_probs = self.collect(step)

                # Obser reward and next obs
                obs, rewards, dones, ava, infos = self.envs.step(actions)

                for i in range(self.n_rollout_threads):
                    if dones[i] or self.envs.envs.envs[i].steps_remaining == 0:
                        global_step = total_num_steps + step * self.n_rollout_threads + i
                        print(
                            f"global_step={global_step}, episodic_return={rewards[i]}, episodic_length={0}")
                        self.writter.add_scalar("charts/episodic_return", rewards[i], global_step)
                        self.writter.add_scalar("charts/episodic_length", 0, global_step)
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
            self.buffer.after_update()

            # save model
            if (episode == episodes - 1):
                self.save(episode)

            # log information
            if episode % self.log_interval == 0:
                print("total_num_steps: ", total_num_steps)
                self.log_train(train_infos, total_num_steps)
