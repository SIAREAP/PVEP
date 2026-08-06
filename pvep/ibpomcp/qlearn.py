from __future__ import annotations

import math
import random
from typing import Any, Dict, List

import numpy as np


def create_qtable(actions: List[Any]) -> Dict[str, Dict[str, float | int]]:
    qtable: Dict[str, Dict[str, float | int]] = {}
    for a in actions:
        qtable[str(a)] = {"qvalue": 0.0, "sumvalue": 0.0, "trials": 0}
    return qtable


def ucb_select_action(node: Any, c: float = 0.5, mode: str = "max") -> Any:
    # 1. Initialising the support values
    if mode == "max":
        target_ucb, target_a = -np.inf, None
    elif mode == "min":
        target_ucb, target_a = np.inf, None
    else:
        raise NotImplementedError(f"Invalid mode for UCB: {mode}")

    # 2. Checking the best action via UCT algorithm
    for a in node.actions:
        qvalue = float(node.qtable[str(a)]["qvalue"])
        trials = int(node.qtable[str(a)]["trials"])
        if trials > 0:
            if mode == "max":
                current_ucb = qvalue + float(c) * np.sqrt(np.log(float(node.visits)) / float(trials))
                if current_ucb > target_ucb:
                    target_ucb = current_ucb
                    target_a = a
            else:
                current_ucb = qvalue - float(c) * np.sqrt(np.log(float(node.visits)) / float(trials))
                if current_ucb < target_ucb:
                    target_ucb = current_ucb
                    target_a = a
        else:
            return a

    # 3. Checking if the best action was found
    if target_a is None:
        target_a = random.sample(node.actions, 1)[0]

    # 4. Returning the best action
    return target_a


def create_etable(actions: List[Any]) -> Dict[str, Dict[str, float | int]]:
    etable: Dict[str, Dict[str, float | int]] = {}
    for a in actions:
        etable[str(a)] = {"entropy": 0.0, "cumentropy": 0.0, "trials": 0, "max_entropy": 1.0}
    return etable


def iucb_select_action(node: Any, alpha: float, mode: str = "max") -> Any:
    # 1. Initialising the support values
    if mode == "max":
        target_ucb, target_a = -np.inf, None
    elif mode == "min":
        target_ucb, target_a = np.inf, None 
    else:
        raise NotImplementedError(f"Invalid mode for I-UCB: {mode}")

    # 2. Checking the best action via UCT algorithm
    actions = [a for a in node.actions]
    np.random.shuffle(actions)
    for a in actions:
        qvalue = float(node.qtable[str(a)]["qvalue"])
        trials = int(node.qtable[str(a)]["trials"])
        if trials > 0:
            exploration_value = float(np.sqrt(np.log(float(node.visits)) / float(trials)))
            information_value = float(node.etable[str(a)]["entropy"]) / float(node.etable[str(a)]["max_entropy"])
            current_ucb = qvalue + ((1.0 - float(alpha)) * exploration_value) + (float(alpha) * information_value)

            if mode == "max" and current_ucb > target_ucb:
                target_ucb = current_ucb
                target_a = a
            elif mode == "min" and current_ucb < target_ucb:
                target_ucb = current_ucb
                target_a = a
        else:
            return a

    # 3. Checking if the best action was found
    if target_a is None:
        target_a = random.sample(node.actions, 1)[0]

    # 4. Returning the best action
    return target_a


def entropy(counts: Dict[Any, int | float]) -> float:
    h = 0.0
    norm = float(sum(float(counts[y]) for y in counts))
    if norm <= 0.0:
        return 0.0
    for x in counts:
        px = float(counts[x]) / norm
        if px > 0.0:
            h += px * math.log(px)
    return -h


__all__ = [
    "create_qtable",
    "create_etable",
    "entropy",
    "ucb_select_action",
    "iucb_select_action",
]













