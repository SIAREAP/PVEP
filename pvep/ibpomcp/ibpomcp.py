from __future__ import annotations

import random
import time
from typing import Any, Dict, Tuple

import numpy as np

from .node import IANode, IONode


class IBPOMCP:
    def __init__(self, max_depth: int, max_it: int, kwargs: Dict[str, Any]):
        # Traditional MCTS parameters
        self.root = None
        self.max_depth = int(max_depth)
        self.max_it = int(max_it)

        discount_factor = kwargs.get("discount_factor")
        self.discount_factor = float(discount_factor) if discount_factor is not None else 0.95

        self.alpha = 0.5

        # POMCP enhancements
        particle_revigoration = kwargs.get("particle_revigoration")
        self.pr = bool(particle_revigoration) if particle_revigoration is not None else True

        k = kwargs.get("k")
        self.k = int(k) if k is not None else 100

        # Further settings
        target = kwargs.get("target")
        if target is not None:
            self.target = str(target)
            self.initial_target = str(target)
        else:
            self.target = "iucb-max"
            self.initial_target = "iucb-max"

        adversary_mode = kwargs.get("adversary")
        self.adversary = bool(adversary_mode) if adversary_mode is not None else False

        stack_size = kwargs.get("state_stack_size")
        self.state_stack_size = int(stack_size) if stack_size is not None else 1

        # Evaluation
        self.rollout_total_time = 0.0
        self.rollout_count = 0.0
        self.simulation_total_time = 0.0
        self.simulation_count = 0.0

    def change_paradigm(self) -> str:
        if self.target == "iucb-max":
            return "iucb-min"
        if self.target == "iucb-min":
            return "iucb-max"
        raise NotImplementedError

    def simulate_action(self, node: Any, action: Any) -> Tuple[IANode, float]:
        # 1. Copying the current state for simulation
        tmp_state = node.state.copy()

        # 2. Acting
        next_state, reward, _done, _info = tmp_state.step(action)
        # Depth semantics: observation/history nodes count environment steps; action nodes do not advance time.
        next_node = IANode(action, next_state, node.depth, node)

        # 3. Returning the next node and the reward
        return next_node, float(reward)

    def rollout_policy(self, state: Any) -> Any:
        if getattr(state, "default_policy", None) is not None:
            return state.default_policy()
        return random.choice(state.get_actions_list())

    def rollout(self, node: Any) -> float:
        # 1. Checking if it is an end state or leaf node
        if self.is_terminal(node) or self.is_leaf(node):
            return 0.0

        self.rollout_count += 1
        start_t = time.time()

        # 2. Choosing an action
        action = self.rollout_policy(node.state)

        # 3. Simulating the action
        next_state, reward, _done, _info = node.state.step(action)
        node.state = next_state
        node.observation = next_state.get_observation()
        # One environment step simulated.
        node.depth += 1

        end_t = time.time()
        self.rollout_total_time += float(end_t - start_t)

        # 4. Rolling out
        r = float(reward) + (self.discount_factor * float(self.rollout(node)))
        return float(r)

    def get_rollout_node(self, node: Any) -> IONode:
        obs = node.state.get_observation()
        tmp_state = node.state.copy()
        depth = node.depth
        return IONode(observation=obs, state=tmp_state, depth=depth, parent=None)

    def is_leaf(self, node: Any) -> bool:
        # Depth semantics: environment steps/actions from the root.
        return bool(node.depth >= self.max_depth)

    def is_terminal(self, node: Any) -> bool:
        return bool(node.state.state_set.is_final_state(node.state))

    def simulate(self, node: Any):
        # 1. Checking the stop condition
        if node.depth == 0:
            node.visits += 1

        if self.is_terminal(node) or self.is_leaf(node):
            return 0.0, [node.state]

        # 2. Checking child nodes
        if node.children == []:
            # a. adding the children
            for action in node.actions:
                next_node, _reward = self.simulate_action(node, action)
                node.children.append(next_node)
            rollout_node = self.get_rollout_node(node)
            return float(self.rollout(rollout_node)), [node.state]

        self.simulation_count += 1
        start_t = time.time()

        # 3. Selecting the best action
        action = node.select_action(coef=self.alpha, mode=self.target)
        self.target = self.change_paradigm() if self.adversary else self.target

        # 4. Simulating the action
        action_node, reward = self.simulate_action(node, action)

        # 5. Adding the action child on the tree
        existing = None
        for child in node.children:
            if getattr(action_node, "action", None) == getattr(child, "action", None):
                existing = child
                break
        if existing is not None:
            existing.state = action_node.state.copy()
            action_node = existing
        else:
            node.children.append(action_node)
        action_node.visits += 1

        # 6. Getting the observation and adding the observation child on the tree
        observation_node = None
        observation = action_node.state.get_observation()

        for child in action_node.children:
            if getattr(child, "observation", None) == observation:
                observation_node = child
                observation_node.state = action_node.state.copy()
                break

        if observation_node is None:
            observation_node = action_node.add_child(observation)
        observation_node.visits += 1

        end_t = time.time()
        self.simulation_total_time += float(end_t - start_t)

        # 7. Calculating the reward, quality and updating the node
        future_reward, observation_states_found = self.simulate(observation_node)
        r = float(reward) + (self.discount_factor * float(future_reward))

        # - node update
        node.add_to_observation_distribution(observation_states_found)
        node.particle_filter.append(node.state)
        node.update(action, r)

        observation_states_found.append(node.state)
        return float(r), observation_states_found

    def search(self, node: Any, agent: Any) -> Any:
        # 1. Performing the Monte-Carlo Tree Search
        it = 0
        while it < self.max_it:
            self.target = self.initial_target  # resetting the optimisation target

            # a. Sampling the belief state for simulation
            if len(node.particle_filter) == 0:
                belief_state = node.state.sample_state(agent)
            else:
                belief_state = random.sample(node.particle_filter, 1)[0]
            node.state = belief_state

            # b. simulating
            self.alpha = node.get_alpha()
            self.simulate(node)
            it += 1

        self.target = self.initial_target
        self.alpha = node.get_alpha()
        return node.get_best_action(self.alpha, self.target)

    def planning(self, state: Any, agent: Any) -> Tuple[Any, Dict[str, Any]]:
        # 1. Getting the current state and previous action-observation pair
        previous_action = getattr(agent, "next_action", None)
        current_observation = state.get_observation()

        # 2. Defining the root of our search tree
        if self.root is None:
            px = 0.0
            self.root = IONode(observation=None, state=state, depth=0, parent=None)
        else:
            self.root, px = find_new_PO_root(
                state, previous_action, current_observation, agent, self.root, adversary=self.adversary
            )

        # 3. Estimating the parameters (optional; not used by nl2pomdp by default)
        if isinstance(getattr(agent, "smart_parameters", None), dict) and "estimation_method" in agent.smart_parameters:
            # Keep this optional & best-effort; estimation is domain-specific.
            try:
                from .estimation_stub import type_parameter_estimation  # type: ignore

                self.root.state, agent.smart_parameters["estimation"] = type_parameter_estimation(
                    self.root.state,
                    agent,
                    agent.smart_parameters["estimation_method"],
                    *agent.smart_parameters.get("estimation_args", ()),
                )
            except Exception:
                pass

        # 4. Performing particle revigoration
        if self.pr:
            particle_revigoration(state, agent, self.root, self.k, float(px))

        # 5. Searching for the best action within the tree
        best_action = self.search(self.root, agent)

        # 6. Returning the best action
        info = {
            "nrollouts": float(self.rollout_count),
            "nsimulations": float(self.simulation_count),
            "px": float(px),
        }
        return best_action, info


def ibpomcp_planning(env: Any, agent: Any, max_depth: int = 20, max_it: int = 250, **kwargs: Any):
    # 1. Setting the environment for simulation
    copy_env = env.copy()
    try:
        copy_env.simulation = True
    except Exception:
        pass

    # 2. POMCP Planning
    ibpomcp = IBPOMCP(max_depth, max_it, kwargs) if "ibpomcp" not in agent.smart_parameters else agent.smart_parameters[
        "ibpomcp"
    ]

    # - planning
    next_action, info = ibpomcp.planning(copy_env, agent)

    # 3. Updating the search tree
    agent.smart_parameters["ibpomcp"] = ibpomcp
    agent.smart_parameters["count"] = info
    return next_action, None


def find_new_PO_root(
    current_state: Any,
    previous_action: Any,
    current_observation: Any,
    agent: Any,
    previous_root: Any,
    adversary: bool = False,
):
    px = 0.0
    if previous_root is None:
        new_root = IONode(observation=None, state=current_state, depth=0, parent=None)
        return new_root, px

    action_node, observation_node = None, None

    for child in previous_root.children:
        if getattr(child, "action", None) == previous_action:
            action_node = child
            break

    if action_node is None:
        new_root = IONode(observation=None, state=current_state, depth=0, parent=None)
        return new_root, px

    for child in action_node.children:
        try:
            ok = child.state.observation_is_equal(current_observation)
        except Exception:
            ok = getattr(child, "observation", None) == current_observation
        if ok:
            observation_node = child
            break

    if observation_node is None:
        new_root = IONode(observation=None, state=current_state, depth=0, parent=None)
        return new_root, px

    if adversary:
        # Not used by nl2pomdp; keep behavior close to original.
        action_node, observation_node = None, None
        for child in previous_root.children:
            if getattr(child, "action", None) == agent.smart_parameters.get("adversary_last_action", None):
                action_node = child
                break
        if action_node is None:
            new_root = IONode(observation=None, state=current_state, depth=0, parent=None)
            return new_root, px
        for child in action_node.children:
            if child.state.observation_is_equal(agent.smart_parameters.get("adversary_last_observation", None)):
                observation_node = child
                break
        if observation_node is None:
            new_root = IONode(observation=None, state=current_state, depth=0, parent=None)
            return new_root, px

    new_root = observation_node
    try:
        px = float(new_root.visits) / float(previous_root.visits) if float(previous_root.visits) > 0 else 0.0
    except Exception:
        px = 0.0
    new_root.parent = None
    new_root.update_depth(0)
    return new_root, float(px)


def particle_revigoration(env: Any, agent: Any, root: Any, k: int, px: float) -> None:
    # 1. Copying the current root particle filter
    current_particle_filter = [p for p in getattr(root, "particle_filter", [])]
    px = float(px) if len(current_particle_filter) > 1 else 0.0

    # 2. Reinvigorating particles for the new particle filter or
    # picking particles from the uniform distribution
    root.particle_filter = []

    particle_counter = 0
    while particle_counter < (px) * float(k):
        particle = random.sample(current_particle_filter, 1)[0]
        root.particle_filter.append(particle)
        particle_counter += 1

    particle_counter = 0
    while particle_counter < (1.0 - px) * float(k):
        particle = env.sample_state(agent)
        root.particle_filter.append(particle)
        particle_counter += 1


__all__ = [
    "IBPOMCP",
    "ibpomcp_planning",
    "find_new_PO_root",
    "particle_revigoration",
]













