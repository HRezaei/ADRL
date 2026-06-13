import copy
from collections import deque


class OvercookedPlanner:
    """BFS planner for Overcooked-LLMA environments.

    Given the macro-action env, finds the shortest sequence of macro-actions
    that completes the task (delivers the correct recipe) from the initial state.
    The result is cached by task so it only computes once per env instance.
    """

    def __init__(self, env):
        self._env = env
        self.macro_action_names = env.macroActionName
        self.num_actions = len(self.macro_action_names)
        self._cache = {}

    def _state_key(self, e):
        parts = []
        for ag in e.agent:
            parts += [ag.x, ag.y]
            if ag.holding is None:
                parts.append(-1)
            else:
                h = ag.holding
                parts.append(hash(type(h).__name__))
                if hasattr(h, 'chopped'):
                    parts.append(1 if h.chopped else 0)
                else:
                    parts.append(0)
                if hasattr(h, 'containing'):
                    if h.containing:
                        parts.append(len(h.containing))
                        for ci in h.containing:
                            parts.append(hash(type(ci).__name__))
                            parts.append(1 if ci.chopped else 0)
                    else:
                        parts.append(0)
        for fl in [e.tomato, e.lettuce, e.onion]:
            for f in fl:
                parts += [f.x, f.y, 1 if f.chopped else 0]
        for p in e.plate:
            parts += [p.x, p.y]
            if p.containing:
                parts.append(len(p.containing))
                for ci in p.containing:
                    parts.append(hash(type(ci).__name__))
                    parts.append(1 if ci.chopped else 0)
            else:
                parts.append(0)
        for k in e.knife:
            if k.holding is None:
                parts.append(-1)
            else:
                parts.append(hash(type(k.holding).__name__))
                if hasattr(k.holding, 'chopped'):
                    parts.append(1 if k.holding.chopped else 0)
                else:
                    parts.append(0)
        for d in e.delivery:
            parts.append(1 if d.holding is not None else 0)
        return tuple(parts)

    def _success(self, e):
        return any(d.holding is not None for d in e.delivery)

    def plan(self):
        task_key = str(self._env.task)
        cached = self._cache.get(task_key)
        if cached is not None:
            return cached

        root = copy.deepcopy(self._env)
        root.reset()

        start_key = self._state_key(root)
        q = deque()
        q.append((root, []))
        visited = {start_key}

        while q:
            cur_env, path = q.popleft()

            for a_idx in range(self.num_actions):
                child = copy.deepcopy(cur_env)
                _, reward, done, _ = child.run(a_idx)

                if self._success(child):
                    plan = path + [a_idx]
                    self._cache[task_key] = plan
                    return plan

                if done:
                    continue

                key = self._state_key(child)
                if key not in visited:
                    visited.add(key)
                    q.append((child, path + [a_idx]))

        self._cache[task_key] = None
        return None

    def get_action_names(self, plan):
        if plan is None:
            return None
        return [self.macro_action_names[i] for i in plan]
