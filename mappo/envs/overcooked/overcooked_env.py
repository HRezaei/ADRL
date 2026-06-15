from gym_macro_overcooked.macActEnvWrapper import MacEnvWrapper
import gym
import numpy as np
from mappo.envs.overcooked.planner import OvercookedPlanner

def make_env(env_id, seed, idx, env_params):
    def thunk():

        env = gym.make(env_id, **env_params)
        env = MacEnvWrapper(env)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

TASKLIST = ["tomato salad", "lettuce salad", "onion salad", "lettuce-tomato salad", "onion-tomato salad", "lettuce-onion salad", "lettuce-onion-tomato salad"]
REWARDLIST = {"subtask finished": 0.2, "correct delivery": 1.0, "wrong delivery": -0.1, "step penalty": -0.001}

class OvercookedEnv:

    def __init__(self, env_id, num_envs, seed, debug=False, use_planner=False) -> None:
        if env_id == "Overcooked-LLMA-v4":
            task_start = 0
        elif env_id == "Overcooked-LLMA-v3":
            task_start = 0
        env_params = {
              'grid_dim': [7,7],
              'rewardList': REWARDLIST,
              'map_type': "A",
              'n_agent': 1,
              'obs_radius': 2,
              'mode': "vector",
              'debug': debug,
              }
        self.num_envs = num_envs
        self.num_agents = 1
        self.envs = gym.vector.SyncVectorEnv([make_env(env_id, seed + i, i, {**env_params, 'task': TASKLIST[(task_start + i) % len(TASKLIST)]}) for i in range(num_envs)])
        self._task_offset = task_start
        self._planner = OvercookedPlanner(self.envs.envs[0].env.unwrapped)
        self.use_planner = use_planner
        self.plan = None
        self.plan_names = None
        self.plan_step = np.zeros(self.num_envs, dtype=np.int64)
        self._plans_by_task = {}
        self.env_plans = None
        self.env_plan_names = None
        if env_id == "Overcooked-LLMA-v4":
            self.task = 0
        elif env_id == "Overcooked-LLMA-v3":
            self.task = 3
        print("env_id: ", env_id)
        
        assert isinstance(self.envs.single_action_space, gym.spaces.Discrete)

    def _set_env_task(self, env_idx, task_name):
        raw = self.envs.envs[env_idx].env.unwrapped
        raw.task = task_name
        raw.oneHotTask = [1 if t == task_name else 0 for t in TASKLIST]

    def _refresh_tasks(self):
        for i in range(self.num_envs):
            task_name = TASKLIST[(self._task_offset + i) % len(TASKLIST)]
            self._set_env_task(i, task_name)
        self._task_offset = (self._task_offset + self.num_envs) % len(TASKLIST)

    def _ensure_plans(self, task_names):
        unknown = [t for t in task_names if t not in self._plans_by_task]
        if not unknown:
            return
        saved_task = self.envs.envs[0].env.unwrapped.task
        for task_name in unknown:
            self._set_env_task(0, task_name)
            plan = self._planner.plan()
            plan_names = self._planner.get_action_names(plan)
            self._plans_by_task[task_name] = (plan, plan_names)
        self._set_env_task(0, saved_task)

    @property
    def task_name(self):
        return TASKLIST[self._task_offset % len(TASKLIST)]
        
    @property
    def task_names(self):
        return [self.envs.envs[i].env.unwrapped.task for i in range(self.num_envs)]
        
    def reset(self):
        self._refresh_tasks()
        ori_obs = self.envs.reset()
        obs, ava = self.handle_obs(ori_obs)

        # Per-env plans
        current_tasks = [self.envs.envs[i].env.unwrapped.task for i in range(self.num_envs)]
        self._ensure_plans(current_tasks)
        self.env_plans = [self._plans_by_task[t][0] for t in current_tasks]
        self.env_plan_names = [self._plans_by_task[t][1] for t in current_tasks]
        self.plan = self.env_plans[0]
        self.plan_names = self.env_plan_names[0]
        self.plan_step[:] = 0

        return obs, ava

    def get_planner_actions(self, ava):
        if self.env_plans is None:
            return None
        if any(p is None for p in self.env_plans):
            return None
        actions = np.empty((self.num_envs, self.num_agents), dtype=np.object_)
        for i in range(self.num_envs):
            action_list = ava[i, 0].split(",")
            actions[i, 0] = action_list[self.env_plans[i][self.plan_step[i]]]
        return actions

    def advance_plan(self, dones):
        done_mask = np.squeeze(dones) > 0
        self.plan_step += 1
        self.plan_step[done_mask] = 0
        if self.env_plans is not None:
            for i in range(self.num_envs):
                if self.env_plans[i] is None:
                    self.plan_step[i] = 0
                elif self.plan_step[i] >= len(self.env_plans[i]):
                    self.plan_step[i] = 0
        
    def step(self, ori_action):
        action = self.handle_action(ori_action)
        ori_next_obs, reward, done, info = self.envs.step(action)
        next_obs, ava = self.handle_obs(ori_next_obs)
        reward = np.repeat(reward[:, None], self.num_agents, axis=1)
        done = np.repeat(done[:, None], self.num_agents, axis=1)
        
        # for i in range(done.shape[0]):
        #     if done[i][0] and reward[i][0] == 0:
        #         reward[i][0] = -1
        
        return next_obs, reward, done, ava, info
    
    def handle_action(self, ori_action):
        action = np.zeros((self.num_envs,), dtype=np.int64)
        for i in range(self.num_envs):
            action[i] = self.template2action[i][ori_action[i][0]]
        return action
    
    def handle_obs(self, ori_obs):
        obs = np.empty((self.num_envs, self.num_agents), dtype=np.object_)
        ava = np.empty_like(obs, dtype=np.object_)
        self.template2action = []
        for i in range(self.num_envs):
            text_obs = self.obs2text(ori_obs[i])
            obs[i, 0] = text_obs["prompt"]
            ava[i, 0] = ",".join(text_obs["avaliable_action"])
            self.template2action.append({k:i for i,k in enumerate(text_obs["avaliable_action"])})
        return obs, ava
    
    def render(self, env_idx=0):
        return self.envs.envs[env_idx].render(mode='rgb_array')

    def close(self):
        self.envs.close()
        
    def obs2text(self, obs):
        if self.task == 3:
            obs = obs.tolist()
            action_list = [
                "pick up the tomato", 
                "pick up the lettuce", 
                "pick up the onion", 
                "take the empty bowl",
                "walk to the first cutting board",
                "walk to the second cutting board",
                "serve nothing",
                "chop nothing",
            ]

            ingredient_in_ori_pos = [0, 0, 0, 0]
            ingredient = ["a tomato", "a lettuce", "an onion", "a bowl"]
            raw_ingredient = ["tomato", "lettuce", "onion", "bowl"]
            chopped = [False, False, False]
            ori_pos = [[0, 5], [1, 6], [2, 6], [6, 5]]
            sentences = ["There are two fixed cutting boards in the room."]

            item = []
            item_index = []
            agent_pos = obs[17:19]
            first_cutting_board_pos = [1, 0]
            second_cutting_board_pos = [2, 0]

            item_pos = {"in_agent": agent_pos, "in_first_cutting_board": first_cutting_board_pos, "in_second_cutting_board": second_cutting_board_pos}
            overlay = {"in_agent": [], "in_first_cutting_board": [], "in_second_cutting_board": []}
            

            for i in range(4):
                pos = obs[3 * i: 3 * i + 2]
                if  pos == ori_pos[i]:
                    ingredient_in_ori_pos[i] == 1
                    item.append(ingredient[i])
                    item_index.append(i)
                    
                if i < 3 and obs[3 * i + 2] == 3:
                    chopped[i] = True
                
                for k in overlay.keys():
                    if pos == item_pos[k]:
                        overlay[k].append(i)
                        
                        if len(overlay[k]) > 1:
                            action_list[3] = "take the bowl"

            if len(item) == 1:
                template = "You notice {} on the table."
            elif len(item) == 2:
                template = "You notice {} and {} on the different tables."
            elif len(item) == 3:
                template = "You notice {}, {} and {} on the different tables."
            elif len(item) == 4:
                template = "You notice {}, {}, {} and {} on the different tables."

            if len(item) > 0:
                sentences.append(template.format(*item).capitalize()) 

            cutting_board_index = ["first", "second"]
            cutting_board_name = ["in_first_cutting_board", "in_second_cutting_board"]
            for cindex in range(2):
                if len(overlay[cutting_board_name[cindex]]) == 1:
                    id  = overlay[cutting_board_name[cindex]][0]
                    template = "{} is on the {} cutting board."
                    if id == 3:
                        sentences.append(template.format("a bowl", cutting_board_index[cindex]).capitalize()) 
                    else:
                        if chopped[id]:
                            sentences.append(template.format("a chopped " + raw_ingredient[id], cutting_board_index[cindex]).capitalize()) 
                        else:
                            sentences.append(template.format("an unchopped " + raw_ingredient[id], cutting_board_index[cindex]).capitalize()) 
                        if agent_pos == [cindex + 1, 1]:
                            action_list[-1] = "chop the " + raw_ingredient[id]
                                
                elif len(overlay[cutting_board_name[cindex]]) > 1:
                    in_plate_item = overlay[cutting_board_name[cindex]][:-1]
                    if len(in_plate_item) == 1:
                        full_plate_template = "A bowl containing chopped {} is on the {} cutting board."
                    elif len(in_plate_item) == 2:
                        full_plate_template = "A bowl containing chopped {} and {} is on the {} cutting board."
                    elif len(in_plate_item) == 3:
                        full_plate_template = "A bowl containing chopped {}, {} and {} is on the {} cutting board."
                    sentences.append(full_plate_template.format(*[raw_ingredient[id] for id in in_plate_item], cutting_board_index[cindex]).capitalize()) 

            #in front of cutting board 1
            if agent_pos == [1, 1]:
                cindex = 0
            #in front of cutting board 2
            elif agent_pos == [2, 1]:
                cindex = 1
            else:
                cindex = -1

            action_template = "put the {} on the {} cutting board"
            hold_bowl_action = [
                "put the tomato in the bowl",
                "put the lettuce in the bowl",  
                "put the onion in the bowl",
            ]

            if cindex >= 0:
                if len(overlay["in_agent"]) == 0:
                    template = "Currently you are standing in front of the {} cutting board without anything in hand."
                    sentences.append(template.format(cutting_board_index[cindex]).capitalize()) 

                elif len(overlay["in_agent"]) == 1:
                    action_list[6] = "serve the dish"
                    id  = overlay["in_agent"][0]
                    template = "Currently you are standing in front of the {} cutting board, carrying {} in hand."
                    if id == 3:
                        sentences.append(template.format(cutting_board_index[cindex], "a bowl").capitalize())
                        action_list[:3] = hold_bowl_action
                        action_list[4] = action_template.format(raw_ingredient[id], "first") 
                        action_list[5] = action_template.format(raw_ingredient[id], "second") 
                    else:
                        if chopped[id]:
                            sentences.append(template.format(cutting_board_index[cindex], "a chopped " + raw_ingredient[id], ).capitalize()) 
                        else:
                            sentences.append(template.format(cutting_board_index[cindex], "an unchopped " + raw_ingredient[id]).capitalize()) 
                            action_list[4] = action_template.format(raw_ingredient[id], "first") 
                            action_list[5] = action_template.format(raw_ingredient[id], "second") 
                elif len(overlay["in_agent"]) > 1:
                    action_list[6] = "serve the dish"
                    in_plate_item = overlay["in_agent"][:-1]
                    if len(in_plate_item) == 1:
                        full_plate_template = "Currently you are standing in front of the {} cutting board, carrying a bowl containing chopped {} in hand."
                    elif len(in_plate_item) == 2:
                        full_plate_template = "Currently you are standing in front of the {} cutting board, carrying a bowl containing chopped {} and {} in hand."
                    elif len(in_plate_item) == 3:
                        full_plate_template = "Currently you are standing in front of the {} cutting board, carrying a bowl containing chopped {}, {} and {} in hand."

                    sentences.append(full_plate_template.format(cutting_board_index[cindex], *[raw_ingredient[id] for id in in_plate_item]).capitalize())
                    action_list[:3] = hold_bowl_action
                    action_list[4] = action_template.format("bowl", "first") 
                    action_list[5] = action_template.format("bowl", "second") 
            else:
                if len(overlay["in_agent"]) == 0:
                    template = "Currently you don't have anything in hand."
                    sentences.append(template.format(cutting_board_index[cindex]).capitalize()) 

                elif len(overlay["in_agent"]) == 1:
                    action_list[6] = "serve the dish"
                    id  = overlay["in_agent"][0]
                    template = "Currently you are carrying {} in hand."
                    if id == 3:
                        sentences.append(template.format("a bowl").capitalize())
                        action_list[:3] = hold_bowl_action
                        action_list[4] = action_template.format(raw_ingredient[id], "first") 
                        action_list[5] = action_template.format(raw_ingredient[id], "second") 
                    else:
                        if chopped[id]:
                            sentences.append(template.format("a chopped " + raw_ingredient[id], ).capitalize()) 
                        else:
                            sentences.append(template.format("an unchopped " + raw_ingredient[id]).capitalize()) 
                            action_list[4] = action_template.format(raw_ingredient[id], "first") 
                            action_list[5] = action_template.format(raw_ingredient[id], "second") 
                elif len(overlay["in_agent"]) > 1:
                    action_list[6] = "serve the dish"
                    in_plate_item = overlay["in_agent"][:-1]
                    if len(in_plate_item) == 1:
                        full_plate_template = "Currently you are carrying a bowl containing chopped {}."
                    elif len(in_plate_item) == 2:
                        full_plate_template = "Currently you are carrying a bowl containing chopped {} and {}."
                    elif len(in_plate_item) == 3:
                        full_plate_template = "Currently you are carrying a bowl containing chopped {}, {} and {}."

                    sentences.append(full_plate_template.format(*[raw_ingredient[id] for id in in_plate_item]).capitalize())
                    action_list[:3] = hold_bowl_action
                    action_list[4] = action_template.format("bowl", "first") 
                    action_list[5] = action_template.format("bowl", "second")
            sentences.append("To serve the dish of a bowl only containing chopped tomato and lettuce, you should first")
        elif self.task == 0:
            obs = obs.tolist()

            action_list = [
                "pick up the tomato", 
                "take the bowl",
                "walk to the cutting board",
                "serve nothing",
                "chop nothing",
            ]

            ingredient_in_ori_pos = [0, 0]
            ingredient = ["a tomato", "a bowl"]
            raw_ingredient = ["tomato", "bowl"]
            chopped = [False]
            ori_pos = [[0, 5], [6, 5]]
            sentences = ["There is a fixed cutting board in the room."]
            in_plate = [False, False, False]

            item = []
            item_index = []
            plate_pos = obs[3:5]
            agent_pos = obs[9:11]
            first_cutting_board_pos = [1, 0]

            item_pos = {"in_agent": agent_pos, "in_first_cutting_board": first_cutting_board_pos}
            overlay = {"in_agent": [], "in_first_cutting_board": []}
            
            
            for i in range(2):
                pos = obs[3 * i: 3 * i + 2]
                if  pos == ori_pos[i]:
                    ingredient_in_ori_pos[i] == 1
                    item.append(ingredient[i])
                    item_index.append(i)
                    
                if i < 1 and obs[3 * i + 2] == 3:
                    chopped[i] = True
                
                for k in overlay.keys():
                    if pos == item_pos[k]:
                        overlay[k].append(i)
            if len(item) == 1:
                template = "You notice {} on the table."
            elif len(item) == 2:
                template = "You notice {} and {} on the different tables."

            if len(item) > 0:
                sentences.append(template.format(*item).capitalize()) 

            cutting_board_index = ["first"]
            cutting_board_name = ["in_first_cutting_board"]

            cindex = 0
            if len(overlay[cutting_board_name[cindex]]) == 1:
                id  = overlay[cutting_board_name[cindex]][0]
                template = "{} is on the cutting board."
                if id == 1:
                    sentences.append(template.format("a bowl").capitalize()) 
                else:
                    if chopped[id]:
                        sentences.append(template.format("a chopped " + raw_ingredient[id]).capitalize()) 
                    else:
                        sentences.append(template.format("an unchopped " + raw_ingredient[id]).capitalize()) 
                    if agent_pos == [cindex + 1, 1]:
                        action_list[-1] = "chop the " + raw_ingredient[id]
                        
                            
            elif len(overlay[cutting_board_name[cindex]]) > 1:

                full_plate_template = "a bowl containing a chopped tomato is on the cutting board."
                sentences.append(full_plate_template.capitalize()) 

            #in front of cutting board 1
            if agent_pos == [1, 1]:
                cindex = 0
            #in front of cutting board 2
            elif agent_pos == [2, 1]:
                cindex = 1
            else:
                cindex = -1

            action_template = "put the {} on the cutting board"
            hold_bowl_action = [
                "put the tomato in the bowl",
            ]

            if cindex >= 0:
                if len(overlay["in_agent"]) == 0:
                    template = "Currently you are standing in front of the cutting board without anything in hand."
                    sentences.append(template.format(cutting_board_index[cindex]).capitalize()) 

                elif len(overlay["in_agent"]) == 1:
                    id  = overlay["in_agent"][0]
                    action_list[3] = "serve the dish"
                    template = "Currently you are standing in front of the cutting board, carrying {} in hand."
                    if id == 1:
                        sentences.append(template.format("a bowl").capitalize()) 
                        action_list[0] = hold_bowl_action[0]
                        action_list[2] = action_template.format(raw_ingredient[id]) 
                    else:
                        if chopped[id]:
                            sentences.append(template.format("a chopped " + raw_ingredient[id] ).capitalize()) 
                        else:
                            sentences.append(template.format("an unchopped " + raw_ingredient[id]).capitalize()) 
                            action_list[2] = action_template.format(raw_ingredient[id]) 
                elif len(overlay["in_agent"]) > 1:
                    action_list[3] = "serve the dish"
                    in_plate_item = overlay["in_agent"][:-1]
                    if len(in_plate_item) == 1:
                        full_plate_template = "Currently you are standing in front of the cutting board, carrying a bowl containing chopped {} in hand."
                    sentences.append(full_plate_template.format(*[raw_ingredient[id] for id in in_plate_item]).capitalize()) 
                    action_list[0] = hold_bowl_action[0]
                    action_list[2] = action_template.format("bowl") 
            else:
                if len(overlay["in_agent"]) == 0:
                    template = "Currently you don't have anything in hand."
                    sentences.append(template.format(cutting_board_index[cindex]).capitalize()) 

                elif len(overlay["in_agent"]) == 1:
                    action_list[3] = "serve the dish"
                    id  = overlay["in_agent"][0]
                    template = "Currently you are carrying {} in hand."
                    if id == 1:
                        sentences.append(template.format("a bowl").capitalize()) 
                        action_list[0] = hold_bowl_action[0]
                        action_list[2] = action_template.format(raw_ingredient[id]) 
                    else:
                        if chopped[id]:
                            sentences.append(template.format("a chopped " + raw_ingredient[id], ).capitalize()) 
                        else:
                            sentences.append(template.format("an unchopped " + raw_ingredient[id]).capitalize()) 
                            action_list[2] = action_template.format(raw_ingredient[id]) 
                elif len(overlay["in_agent"]) > 1:
                    action_list[3] = "serve the dish"
                    in_plate_item = overlay["in_agent"][:-1]
                    if len(in_plate_item) == 1:
                        full_plate_template = "Currently you are carrying a bowl containing chopped {}."
                    elif len(in_plate_item) == 2:
                        full_plate_template = "Currently you are carrying a bowl containing chopped {} and {}."
                    elif len(in_plate_item) == 3:
                        full_plate_template = "Currently you are carrying a bowl containing chopped {}, {} and {}."
                    sentences.append(full_plate_template.format(*[raw_ingredient[id] for id in in_plate_item]).capitalize())   
                    action_list[0] = hold_bowl_action[0]
                    action_list[2] = action_template.format("bowl") 

            sentences.append("To serve the dish of a bowl only containing chopped tomato, you should first")

        return {"prompt": " ".join(sentences), "avaliable_action": action_list}
