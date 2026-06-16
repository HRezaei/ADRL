import copy
from collections import deque

import sys
import os
curr_dir = os.path.dirname(os.path.realpath(__file__))
src_dir = os.path.join(curr_dir, "../../../src/virtual-home/virtual-home/virtual_home")
sys.path.append(src_dir)
sys.path.append(os.path.join(src_dir, "../simulation"))

from simulation.evolving_graph.environment import EnvironmentGraph, EnvironmentState
from simulation.evolving_graph.execution import ScriptExecutor
from simulation.evolving_graph.scripts import read_script_from_string


def _get_target_ids(graph_dict):
    tid = {}
    nodes_by_class = {}
    for n in graph_dict["nodes"]:
        nodes_by_class.setdefault(n["class_name"], []).append(n["id"])
    for cls in ("character", "pancake", "microwave", "chips", "milk", "tv",
                "sofa", "coffeetable", "kitchen", "livingroom", "bathroom", "bedroom"):
        if cls in nodes_by_class:
            if cls == "sofa":
                tid[cls] = nodes_by_class[cls][2] if len(nodes_by_class[cls]) > 2 else nodes_by_class[cls][-1]
            else:
                tid[cls] = nodes_by_class[cls][0]
    return tid


class VirtualHomePlanner:
    """BFS planner for symbolic VirtualHome tasks.

    Finds the shortest sequence of action indices that reaches the goal
    from the initial graph state. Cached by task (env_id) so it is
    computed only once.
    """

    def __init__(self, env_id, action_scripts, name_equivalence=None):
        self.env_id = env_id
        self.action_scripts = action_scripts
        self.name_equivalence = name_equivalence
        self._cache = {}

    def _state_key(self, graph_dict, tid):
        char_id = tid.get("character")
        if char_id is None:
            for n in graph_dict["nodes"]:
                if n["class_name"] == "character":
                    char_id = n["id"]
                    break

        relevant_edges = []
        relevant_node_states = {}

        for n in graph_dict["nodes"]:
            cls = n["class_name"]
            if cls in ("character", "pancake", "microwave", "chips", "milk",
                       "tv", "sofa", "coffeetable", "kitchen", "livingroom",
                       "bathroom", "bedroom"):
                relevant_node_states[n["id"]] = tuple(sorted(n.get("states", [])))

        for e in graph_dict["edges"]:
            if e["from_id"] != char_id and e["to_id"] != char_id:
                continue
            if e["relation_type"] in ("CLOSE", "INSIDE", "FACING", "ON",
                                      "HOLDS_LH", "HOLDS_RH"):
                relevant_edges.append((e["from_id"], e["to_id"], e["relation_type"]))

        if self.env_id == "VirtualHome-v1":
            for e in graph_dict["edges"]:
                if (e["from_id"] == tid.get("pancake")
                        and e["to_id"] == tid.get("microwave")
                        and e["relation_type"] == "INSIDE"):
                    relevant_edges.append((e["from_id"], e["to_id"], e["relation_type"]))

        if self.env_id == "VirtualHome-v2":
            for e in graph_dict["edges"]:
                if e["relation_type"] == "ON":
                    if (e["from_id"] == tid.get("milk")
                            and e["to_id"] == tid.get("coffeetable")):
                        relevant_edges.append((e["from_id"], e["to_id"], e["relation_type"]))
                    if (e["from_id"] == tid.get("chips")
                            and e["to_id"] == tid.get("coffeetable")):
                        relevant_edges.append((e["from_id"], e["to_id"], e["relation_type"]))

        relevant_edges.sort()
        node_states_tuple = sorted(relevant_node_states.items())
        return (tuple(node_states_tuple), tuple(relevant_edges))

    def _is_success_v1(self, graph_dict, tid):
        for e in graph_dict["edges"]:
            if (e["from_id"] == tid.get("pancake")
                    and e["to_id"] == tid.get("microwave")
                    and e["relation_type"] == "INSIDE"):
                return True
        return False

    def _is_success_v2(self, graph_dict, tid):
        nodes = {n["id"]: n for n in graph_dict["nodes"]}
        edges = graph_dict["edges"]

        char_id = tid.get("character")
        if char_id is None:
            return False

        in_livingroom = False
        facing_tv = False
        sitting_sofa = False
        close_to_milk = False
        close_to_chips = False
        milk_on_table = False
        chips_on_table = False

        for e in edges:
            if e["from_id"] != char_id:
                continue
            if e["relation_type"] == "INSIDE" and e["to_id"] == tid.get("livingroom"):
                in_livingroom = True
            if e["relation_type"] == "FACING" and e["to_id"] == tid.get("tv"):
                facing_tv = True
            if e["relation_type"] == "ON" and e["to_id"] == tid.get("sofa"):
                sitting_sofa = True
            if e["relation_type"] == "CLOSE" and e["to_id"] == tid.get("milk"):
                close_to_milk = True
            if e["relation_type"] == "CLOSE" and e["to_id"] == tid.get("chips"):
                close_to_chips = True

        for e in edges:
            if e["relation_type"] == "ON":
                if e["from_id"] == tid.get("milk") and e["to_id"] == tid.get("coffeetable"):
                    milk_on_table = True
                if e["from_id"] == tid.get("chips") and e["to_id"] == tid.get("coffeetable"):
                    chips_on_table = True

        tv_node = nodes.get(tid.get("tv"))
        tv_on = tv_node is not None and "ON" in tv_node.get("states", [])

        return (in_livingroom and facing_tv and tv_on and sitting_sofa
                and (close_to_milk or milk_on_table)
                and (close_to_chips or chips_on_table))

    def _is_success(self, graph_dict, tid):
        if self.env_id == "VirtualHome-v1":
            return self._is_success_v1(graph_dict, tid)
        else:
            return self._is_success_v2(graph_dict, tid)

    def plan(self, init_graph_dict):
        task_key = self.env_id
        cached = self._cache.get(task_key)
        if cached is not None:
            return cached

        tid = _get_target_ids(init_graph_dict)
        name_equiv = self.name_equivalence

        q = deque()
        q.append((init_graph_dict, []))
        visited = {self._state_key(init_graph_dict, tid)}

        while q:
            cur_dict, path = q.popleft()

            for a_idx, a_script in enumerate(self.action_scripts):
                graph = EnvironmentGraph(cur_dict)
                state = EnvironmentState(graph, name_equiv, instance_selection=True)
                executor = ScriptExecutor(graph, name_equiv, char_index=0)
                script = read_script_from_string(a_script)
                success, new_state = executor.execute_one_step(script, state)
                if not success:
                    continue

                new_dict = new_state.to_dict()
                if self._is_success(new_dict, tid):
                    plan = path + [a_idx]
                    self._cache[task_key] = plan
                    return plan

                key = self._state_key(new_dict, tid)
                if key not in visited:
                    visited.add(key)
                    q.append((new_dict, path + [a_idx]))

        self._cache[task_key] = None
        return None
