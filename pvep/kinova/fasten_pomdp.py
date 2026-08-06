"""POMDP model for seven-hole screw installation from tri-state starts.

Each hole starts in one hidden state:

  EMPTY     no screw is present; missing detections are treated as EMPTY
  MISALIGN  screw is present but not aligned
  ALIGN     screw is present and aligned, but not fastened

The robot must make every hole fastened. Visual recognition is tri-class:
EMPTY / MISALIGN / ALIGN, plus a discretized confidence bin. Ordinary
recognition uses noisy MLM observations; optimal recognition uses clean MLM
observations from the original image.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple


STATE_EMPTY = "empty"
STATE_MISALIGN = "misalign"
STATE_ALIGN = "align"
STATE_ORDER = (STATE_EMPTY, STATE_MISALIGN, STATE_ALIGN)


class FastenAction(IntEnum):
    FASTEN = 0
    PROBE_FIXED = 1
    PROBE_OPTIMAL = 2
    RECOVER = 3
    INSERT = 4


class VisualLabel(IntEnum):
    EMPTY = 0
    MISALIGN = 1
    ALIGN = 2


class ConfBin(IntEnum):
    LOW = 0     # winning confidence < 0.50
    MED = 1     # 0.50 <= winning confidence < 0.80
    HIGH = 2    # winning confidence >= 0.80


DiscObs = Tuple[VisualLabel, ConfBin]
Belief = Dict[str, float]

STATE_TO_LABEL = {
    STATE_EMPTY: VisualLabel.EMPTY,
    STATE_MISALIGN: VisualLabel.MISALIGN,
    STATE_ALIGN: VisualLabel.ALIGN,
}
LABEL_TO_STATE = {label: state for state, label in STATE_TO_LABEL.items()}
DEFAULT_TERMINAL_ACTION_BELIEF_THRESHOLD = 0.50
DEFAULT_FASTEN_CONFIDENCE_THRESHOLD = DEFAULT_TERMINAL_ACTION_BELIEF_THRESHOLD
EMPIRICAL_LIKELIHOOD_SMOOTHING = 0.02
DEFAULT_FAILURE_PENALTY = 50.0
DEFAULT_TASK_SUCCESS_REWARD = 10.0
ALL_DISC_OBS: Tuple[DiscObs, ...] = tuple(
    (label, conf_bin) for label in VisualLabel for conf_bin in ConfBin
)


# ---------------------------------------------------------------------------
# Empirical observation model
# ---------------------------------------------------------------------------

# Source: train60 proposal model, test40 split, target holes A-F only,
# conf=0.15, noise_strength=0.65, seed=20260707. Continuous prediction
# confidence is rebinned as LOW < 0.50, MED [0.50, 0.80), HIGH >= 0.80.
# MISSING_PRED is mapped to EMPTY/LOW.
_NOISY_OBS_COUNTS: Dict[str, Dict[DiscObs, int]] = {
    STATE_EMPTY: {
        (VisualLabel.EMPTY, ConfBin.HIGH): 44,
        (VisualLabel.EMPTY, ConfBin.MED): 31,
        (VisualLabel.EMPTY, ConfBin.LOW): 12,
        (VisualLabel.MISALIGN, ConfBin.MED): 1,
        (VisualLabel.MISALIGN, ConfBin.LOW): 3,
        (VisualLabel.ALIGN, ConfBin.MED): 1,
    },
    STATE_MISALIGN: {
        (VisualLabel.EMPTY, ConfBin.MED): 2,
        (VisualLabel.EMPTY, ConfBin.LOW): 2,
        (VisualLabel.MISALIGN, ConfBin.HIGH): 5,
        (VisualLabel.MISALIGN, ConfBin.MED): 10,
        (VisualLabel.MISALIGN, ConfBin.LOW): 6,
        (VisualLabel.ALIGN, ConfBin.MED): 3,
        (VisualLabel.ALIGN, ConfBin.LOW): 6,
    },
    STATE_ALIGN: {
        (VisualLabel.MISALIGN, ConfBin.LOW): 2,
        (VisualLabel.ALIGN, ConfBin.HIGH): 67,
        (VisualLabel.ALIGN, ConfBin.MED): 43,
        (VisualLabel.ALIGN, ConfBin.LOW): 2,
    },
}

# Source: train60 proposal model, test40 split, target holes A-F only,
# conf=0.15, noise_strength=0.0. Same confidence binning as above.
_CLEAN_OBS_COUNTS: Dict[str, Dict[DiscObs, int]] = {
    STATE_EMPTY: {
        (VisualLabel.EMPTY, ConfBin.HIGH): 65,
        (VisualLabel.EMPTY, ConfBin.MED): 22,
        (VisualLabel.EMPTY, ConfBin.LOW): 5,
    },
    STATE_MISALIGN: {
        (VisualLabel.EMPTY, ConfBin.LOW): 2,
        (VisualLabel.EMPTY, ConfBin.MED): 1,
        (VisualLabel.MISALIGN, ConfBin.HIGH): 6,
        (VisualLabel.MISALIGN, ConfBin.MED): 9,
        (VisualLabel.MISALIGN, ConfBin.LOW): 8,
        (VisualLabel.ALIGN, ConfBin.MED): 3,
        (VisualLabel.ALIGN, ConfBin.LOW): 5,
    },
    STATE_ALIGN: {
        (VisualLabel.ALIGN, ConfBin.HIGH): 102,
        (VisualLabel.ALIGN, ConfBin.MED): 12,
    },
}

DEFAULT_NOISY_TARGET_ACC = 220 / 240
_NOISY_TARGET_ACC = DEFAULT_NOISY_TARGET_ACC
DEFAULT_NOISY_OUTPUT_CORRUPTION = 0.0
_NOISY_OUTPUT_CORRUPTION = DEFAULT_NOISY_OUTPUT_CORRUPTION
_OPTIMAL_TRUTH_OVERRIDE: Optional[float] = None


def normalize_belief(belief: Belief) -> Belief:
    total = sum(max(0.0, float(belief.get(state, 0.0))) for state in STATE_ORDER)
    if total <= 1e-12:
        return {state: 1.0 / len(STATE_ORDER) for state in STATE_ORDER}
    return {
        state: max(0.0, float(belief.get(state, 0.0))) / total
        for state in STATE_ORDER
    }


def set_observation_truth_probs(*, optimal: Optional[float] = None) -> None:
    """Optionally override empirical clean-MLM observations with P(truth)."""
    global _OPTIMAL_TRUTH_OVERRIDE
    if optimal is None:
        return
    if not 0.0 <= optimal <= 1.0:
        raise ValueError(f"optimal truth probability must be in [0, 1], got {optimal}")
    _OPTIMAL_TRUTH_OVERRIDE = float(optimal)


def _counts_accuracy(counts_by_state: Dict[str, Dict[DiscObs, int]]) -> float:
    total = 0
    correct = 0
    for state, counts in counts_by_state.items():
        truth_label = STATE_TO_LABEL[state]
        total += sum(counts.values())
        correct += sum(count for (label, _conf), count in counts.items() if label == truth_label)
    return correct / total if total else 0.0


def set_yolo_observation_accuracy(target_acc: Optional[float] = None) -> None:
    """Set effective ordinary/noisy MLM accuracy by flipping correct labels."""
    global _NOISY_TARGET_ACC
    if target_acc is None:
        return
    base_acc = _counts_accuracy(_NOISY_OBS_COUNTS)
    if not 0.0 <= target_acc <= base_acc:
        raise ValueError(
            f"target noisy MLM accuracy must be in [0, {base_acc:.6f}], got {target_acc}"
        )
    _NOISY_TARGET_ACC = float(target_acc)


def set_yolo_output_corruption(epsilon: Optional[float] = None) -> None:
    """Flip ordinary/noisy MLM labels to one of the other labels with probability epsilon."""
    global _NOISY_OUTPUT_CORRUPTION
    if epsilon is None:
        return
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError(f"MLM corruption epsilon must be in [0, 1], got {epsilon}")
    _NOISY_OUTPUT_CORRUPTION = float(epsilon)


def _other_labels(label: VisualLabel) -> List[VisualLabel]:
    return [candidate for candidate in VisualLabel if candidate != label]


def _maybe_corrupt_label(label: VisualLabel, truth_label: VisualLabel) -> VisualLabel:
    base_acc = _counts_accuracy(_NOISY_OBS_COUNTS)
    if label == truth_label and _NOISY_TARGET_ACC < base_acc:
        flip_prob = 1.0 - _NOISY_TARGET_ACC / base_acc
        if random.random() < flip_prob:
            label = random.choice(_other_labels(label))
    if random.random() < _NOISY_OUTPUT_CORRUPTION:
        label = random.choice(_other_labels(label))
    return label


def _empirical_distribution(
    counts_by_state: Dict[str, Dict[DiscObs, int]],
    state: str,
    *,
    apply_noisy_adjustments: bool,
) -> Dict[DiscObs, float]:
    counts = counts_by_state[state]
    total = sum(counts.values())
    if total <= 0:
        return {}

    truth_label = STATE_TO_LABEL[state]
    base_acc = _counts_accuracy(counts_by_state)
    flip_prob = (
        1.0 - _NOISY_TARGET_ACC / base_acc
        if apply_noisy_adjustments and _NOISY_TARGET_ACC < base_acc
        else 0.0
    )
    corruption = _NOISY_OUTPUT_CORRUPTION if apply_noisy_adjustments else 0.0

    dist: Dict[DiscObs, float] = {}
    for (label, conf_bin), count in counts.items():
        prob = count / total
        stay_prob = prob
        if label == truth_label and flip_prob > 0.0:
            stay_prob *= 1.0 - flip_prob
            wrong_share = prob * flip_prob / 2.0
            for wrong in _other_labels(label):
                dist[(wrong, conf_bin)] = dist.get((wrong, conf_bin), 0.0) + wrong_share
        if corruption > 0.0:
            keep = stay_prob * (1.0 - corruption)
            wrong_share = stay_prob * corruption / 2.0
            dist[(label, conf_bin)] = dist.get((label, conf_bin), 0.0) + keep
            for wrong in _other_labels(label):
                dist[(wrong, conf_bin)] = dist.get((wrong, conf_bin), 0.0) + wrong_share
        else:
            dist[(label, conf_bin)] = dist.get((label, conf_bin), 0.0) + stay_prob
    return dist


def _synthetic_truth_distribution(state: str, p_truth: float) -> Dict[DiscObs, float]:
    truth_label = STATE_TO_LABEL[state]
    dist: Dict[DiscObs, float] = {}
    true_conf = {ConfBin.HIGH: 0.90, ConfBin.MED: 0.09, ConfBin.LOW: 0.01}
    wrong_conf = {ConfBin.HIGH: 0.05, ConfBin.MED: 0.35, ConfBin.LOW: 0.60}
    for conf_bin, prob in true_conf.items():
        dist[(truth_label, conf_bin)] = dist.get((truth_label, conf_bin), 0.0) + p_truth * prob
    for wrong_label in _other_labels(truth_label):
        for conf_bin, prob in wrong_conf.items():
            dist[(wrong_label, conf_bin)] = (
                dist.get((wrong_label, conf_bin), 0.0)
                + (1.0 - p_truth) * prob / 2.0
            )
    return dist


def obs_distribution(state: str, probe_type: str) -> Dict[DiscObs, float]:
    if probe_type in ("initial", "fixed"):
        return _empirical_distribution(
            _NOISY_OBS_COUNTS,
            state,
            apply_noisy_adjustments=True,
        )
    if probe_type == "optimal":
        if _OPTIMAL_TRUTH_OVERRIDE is not None:
            return _synthetic_truth_distribution(state, _OPTIMAL_TRUTH_OVERRIDE)
        return _empirical_distribution(
            _CLEAN_OBS_COUNTS,
            state,
            apply_noisy_adjustments=False,
        )
    raise ValueError(f"unknown probe_type: {probe_type!r}")


def _distribution_accuracy(
    counts_by_state: Dict[str, Dict[DiscObs, int]],
    *,
    probe_type: str,
) -> float:
    total = sum(sum(counts.values()) for counts in counts_by_state.values())
    if total <= 0:
        return 0.0
    acc = 0.0
    for state, counts in counts_by_state.items():
        weight = sum(counts.values()) / total
        truth_label = STATE_TO_LABEL[state]
        dist = obs_distribution(state, probe_type)
        acc += weight * sum(prob for (label, _conf), prob in dist.items() if label == truth_label)
    return acc


def _counts_summary(
    *,
    source: str,
    counts_by_state: Dict[str, Dict[DiscObs, int]],
    probe_type: str,
) -> Dict[str, Any]:
    total = sum(sum(counts.values()) for counts in counts_by_state.values())
    correct = 0
    for state, counts in counts_by_state.items():
        truth_label = STATE_TO_LABEL[state]
        correct += sum(count for (label, _conf), count in counts.items() if label == truth_label)
    counts = {
        state: {f"{label.name}/{conf.name}": count for (label, conf), count in state_counts.items()}
        for state, state_counts in counts_by_state.items()
    }
    return {
        "source": source,
        "missing_mapping": "EMPTY",
        "correct": correct,
        "total": total,
        "base_accuracy": correct / total if total else 0.0,
        "effective_accuracy": _distribution_accuracy(counts_by_state, probe_type=probe_type),
        "target_accuracy_before_corruption": (
            _NOISY_TARGET_ACC if probe_type in ("initial", "fixed") else None
        ),
        "label_corruption_epsilon": (
            _NOISY_OUTPUT_CORRUPTION if probe_type in ("initial", "fixed") else 0.0
        ),
        "likelihood_smoothing": EMPIRICAL_LIKELIHOOD_SMOOTHING,
        "counts": counts,
    }


def get_observation_model_summary() -> Dict[str, Any]:
    noisy = _counts_summary(
        source="empirical_noisy_mlm_train60_test40_holes_A-F_conf0p15_noise0p65_confbin050080",
        counts_by_state=_NOISY_OBS_COUNTS,
        probe_type="fixed",
    )
    if _OPTIMAL_TRUTH_OVERRIDE is None:
        optimal = _counts_summary(
            source="empirical_clean_mlm_train60_test40_holes_A-F_conf0p15_noise0p0_confbin050080",
            counts_by_state=_CLEAN_OBS_COUNTS,
            probe_type="optimal",
        )
    else:
        optimal = {
            "source": "truth_probability_override",
            "p_truth": _OPTIMAL_TRUTH_OVERRIDE,
        }
    return {"initial": dict(noisy), "fixed": dict(noisy), "optimal": optimal}


def sample_disc_obs(state: str, probe_type: str) -> DiscObs:
    dist = obs_distribution(state, probe_type)
    obs_list = list(dist.keys())
    weights = [dist[obs] for obs in obs_list]
    return random.choices(obs_list, weights=weights, k=1)[0]


def obs_likelihood(state: str, probe_type: str, obs: DiscObs) -> float:
    base = obs_distribution(state, probe_type).get(obs, 0.0)
    if probe_type in ("initial", "fixed", "optimal") and _OPTIMAL_TRUTH_OVERRIDE is None:
        smooth = EMPIRICAL_LIKELIHOOD_SMOOTHING / len(ALL_DISC_OBS)
        return (1.0 - EMPIRICAL_LIKELIHOOD_SMOOTHING) * base + smooth
    return base


def bayesian_update_belief(belief: Belief, obs: DiscObs, probe_type: str) -> Belief:
    belief = normalize_belief(belief)
    updated = {
        state: belief[state] * obs_likelihood(state, probe_type, obs)
        for state in STATE_ORDER
    }
    return normalize_belief(updated)


def obs_distribution_for_belief(belief: Belief, probe_type: str) -> Dict[DiscObs, float]:
    belief = normalize_belief(belief)
    out: Dict[DiscObs, float] = {}
    for state in STATE_ORDER:
        for obs, prob in obs_distribution(state, probe_type).items():
            out[obs] = out.get(obs, 0.0) + belief[state] * prob
    return out


def sample_state_from_belief(belief: Belief) -> str:
    belief = normalize_belief(belief)
    states = list(STATE_ORDER)
    weights = [belief[state] for state in states]
    return random.choices(states, weights=weights, k=1)[0]


def belief_after_insert(belief: Belief) -> Belief:
    belief = normalize_belief(belief)
    return {
        STATE_EMPTY: 0.0,
        STATE_MISALIGN: belief[STATE_MISALIGN],
        STATE_ALIGN: belief[STATE_ALIGN] + belief[STATE_EMPTY],
    }


def belief_after_recover(belief: Belief) -> Belief:
    belief = normalize_belief(belief)
    return {
        STATE_EMPTY: belief[STATE_EMPTY],
        STATE_MISALIGN: 0.0,
        STATE_ALIGN: belief[STATE_ALIGN] + belief[STATE_MISALIGN],
    }


def belief_after_recover_applicable(_belief: Belief) -> Belief:
    return {
        STATE_EMPTY: 0.0,
        STATE_MISALIGN: 0.0,
        STATE_ALIGN: 1.0,
    }


def belief_after_recover_not_applicable(_belief: Belief) -> Belief:
    return {
        STATE_EMPTY: 1.0,
        STATE_MISALIGN: 0.0,
        STATE_ALIGN: 0.0,
    }


def transition_after_insert(state: str) -> str:
    return STATE_ALIGN if state == STATE_EMPTY else state


def transition_after_recover(state: str) -> str:
    return STATE_ALIGN if state == STATE_MISALIGN else state


def terminal_expected_cost(
    belief: Belief,
    *,
    insert_cost: float,
    recover_cost: float,
    failure_penalty: float,
    success_reward: float = DEFAULT_TASK_SUCCESS_REWARD,
    fasten_confidence_threshold: float = DEFAULT_FASTEN_CONFIDENCE_THRESHOLD,
    allow_recover: bool = True,
) -> Tuple[FastenAction, float]:
    """Best non-probe action and approximate expected positive cost."""
    belief = normalize_belief(belief)

    fasten_cost = float("inf")
    if belief[STATE_ALIGN] > fasten_confidence_threshold:
        fasten_cost = (
            (1.0 - belief[STATE_ALIGN]) * failure_penalty
            - belief[STATE_ALIGN] * success_reward
        )

    insert_plan = float("inf")
    if belief[STATE_EMPTY] > fasten_confidence_threshold:
        insert_plan = (
            belief[STATE_EMPTY] * (insert_cost - success_reward)
            + (1.0 - belief[STATE_EMPTY]) * failure_penalty
        )

    recover_plan = float("inf")
    if allow_recover:
        recover_plan = recover_cost + belief[STATE_EMPTY] * insert_cost - success_reward

    choices = [
        (FastenAction.FASTEN, fasten_cost),
        (FastenAction.INSERT, insert_plan),
        (FastenAction.RECOVER, recover_plan),
    ]
    return min(choices, key=lambda item: item[1])


def one_step_probe_expected_cost(
    belief: Belief,
    *,
    probe_type: str,
    probe_cost: float,
    insert_cost: float,
    recover_cost: float,
    failure_penalty: float,
    success_reward: float = DEFAULT_TASK_SUCCESS_REWARD,
    fasten_confidence_threshold: float = DEFAULT_FASTEN_CONFIDENCE_THRESHOLD,
    allow_recover: bool = True,
) -> float:
    expected_cost = probe_cost
    for obs, obs_prob in obs_distribution_for_belief(belief, probe_type).items():
        posterior = bayesian_update_belief(belief, obs, probe_type)
        _, terminal_cost = terminal_expected_cost(
            posterior,
            insert_cost=insert_cost,
            recover_cost=recover_cost,
            failure_penalty=failure_penalty,
            success_reward=success_reward,
            fasten_confidence_threshold=fasten_confidence_threshold,
            allow_recover=allow_recover,
        )
        expected_cost += obs_prob * terminal_cost
    return expected_cost


def sample_visual_scores(state: str, probe_type: str) -> Dict[VisualLabel, float]:
    """Compatibility helper: sample a discrete obs and return label scores."""
    label, conf_bin = sample_disc_obs(state, probe_type)
    if conf_bin == ConfBin.HIGH:
        win = random.uniform(0.80, 0.95)
    elif conf_bin == ConfBin.MED:
        win = random.uniform(0.50, 0.80)
    else:
        win = random.uniform(0.10, 0.50)
    lose_hi = max(0.05, win * 0.75)
    scores = {candidate: random.uniform(0.05, lose_hi) for candidate in VisualLabel}
    scores[label] = win
    return scores


class FastenStateSet:
    def is_final_state(self, state: "FastenEnv") -> bool:
        return state.phase in ("done", "failed")


@dataclass
class FastenEnv:
    """POMCP-compatible environment/state for one hole."""

    hole_state: str = STATE_EMPTY
    belief: Belief = field(default_factory=lambda: {state: 1.0 / 3.0 for state in STATE_ORDER})
    last_obs: Any = None
    phase: str = "active"
    step_count: int = 0

    visual_cost: float = 3.0
    probe_cost: float = 3.0
    probe_cost_fixed: float = 3.0
    probe_cost_optimal: float = 4.0
    insert_cost: float = 6.0
    recover_cost: float = 10.0
    failure_penalty: float = DEFAULT_FAILURE_PENALTY
    success_reward: float = DEFAULT_TASK_SUCCESS_REWARD
    fasten_confidence_threshold: float = DEFAULT_FASTEN_CONFIDENCE_THRESHOLD
    probe_type: str = "fixed"
    allow_recover: bool = True
    max_steps: int = 12

    state_set: FastenStateSet = field(default_factory=FastenStateSet)

    @classmethod
    def create(
        cls,
        *,
        belief: Optional[Belief] = None,
        probe_type: str = "fixed",
        visual_cost: float = 3.0,
        probe_cost: float = 3.0,
        probe_cost_fixed: float = 3.0,
        probe_cost_optimal: float = 4.0,
        insert_cost: float = 6.0,
        recover_cost: float = 10.0,
        failure_penalty: float = DEFAULT_FAILURE_PENALTY,
        success_reward: float = DEFAULT_TASK_SUCCESS_REWARD,
        fasten_confidence_threshold: float = DEFAULT_FASTEN_CONFIDENCE_THRESHOLD,
        allow_recover: bool = True,
        max_steps: int = 12,
        seed: Optional[int] = None,
    ) -> "FastenEnv":
        if seed is not None:
            random.seed(seed)
        b = normalize_belief(belief or {state: 1.0 / 3.0 for state in STATE_ORDER})
        return cls(
            hole_state=sample_state_from_belief(b),
            belief=b,
            last_obs=None,
            phase="active",
            step_count=0,
            visual_cost=visual_cost,
            probe_cost=probe_cost,
            probe_cost_fixed=probe_cost_fixed,
            probe_cost_optimal=probe_cost_optimal,
            insert_cost=insert_cost,
            recover_cost=recover_cost,
            failure_penalty=failure_penalty,
            success_reward=success_reward,
            fasten_confidence_threshold=fasten_confidence_threshold,
            probe_type=probe_type,
            allow_recover=bool(allow_recover),
            max_steps=max_steps,
            state_set=FastenStateSet(),
        )

    def sample_state(self, _agent: Any) -> "FastenEnv":
        sampled = self.copy()
        sampled.hole_state = sample_state_from_belief(self.belief)
        return sampled

    def copy(self) -> "FastenEnv":
        return copy.deepcopy(self)

    def get_observation(self) -> Any:
        return self.last_obs

    def observation_is_equal(self, other: Any) -> bool:
        return self.last_obs == other

    def get_actions_list(self) -> List[FastenAction]:
        if self.phase != "active":
            return []
        belief = normalize_belief(self.belief)
        actions = []
        if belief[STATE_ALIGN] > self.fasten_confidence_threshold:
            actions.append(FastenAction.FASTEN)
        if belief[STATE_EMPTY] > self.fasten_confidence_threshold:
            actions.append(FastenAction.INSERT)
        if self.allow_recover:
            actions.append(FastenAction.RECOVER)
        actions.append(FastenAction.PROBE_FIXED)
        if self.probe_type != "fixed":
            actions.append(FastenAction.PROBE_OPTIMAL)
        return actions

    def _cost_for_probe_type(self, probe_type: str) -> float:
        if probe_type == "fixed":
            return self.probe_cost_fixed
        if probe_type == "optimal":
            return self.probe_cost_optimal
        raise ValueError(f"unknown probe_type: {probe_type!r}")

    def default_policy(self) -> FastenAction:
        terminal_action, terminal_cost = terminal_expected_cost(
            self.belief,
            insert_cost=self.insert_cost,
            recover_cost=self.recover_cost,
            failure_penalty=self.failure_penalty,
            success_reward=self.success_reward,
            fasten_confidence_threshold=self.fasten_confidence_threshold,
            allow_recover=self.allow_recover,
        )
        if self.step_count >= self.max_steps - 1 and math.isfinite(terminal_cost):
            return terminal_action

        candidates: List[Tuple[FastenAction, float]] = [
            (
                FastenAction.PROBE_FIXED,
                one_step_probe_expected_cost(
                    self.belief,
                    probe_type="fixed",
                    probe_cost=self.probe_cost_fixed,
                    insert_cost=self.insert_cost,
                    recover_cost=self.recover_cost,
                    failure_penalty=self.failure_penalty,
                    success_reward=self.success_reward,
                    fasten_confidence_threshold=self.fasten_confidence_threshold,
                    allow_recover=self.allow_recover,
                ),
            )
        ]
        if self.probe_type != "fixed":
            candidates.append(
                (
                    FastenAction.PROBE_OPTIMAL,
                    one_step_probe_expected_cost(
                        self.belief,
                        probe_type="optimal",
                        probe_cost=self.probe_cost_optimal,
                        insert_cost=self.insert_cost,
                        recover_cost=self.recover_cost,
                        failure_penalty=self.failure_penalty,
                        success_reward=self.success_reward,
                        fasten_confidence_threshold=self.fasten_confidence_threshold,
                        allow_recover=self.allow_recover,
                    ),
                )
            )
        probe_action, probe_cost = min(candidates, key=lambda item: item[1])
        if not math.isfinite(terminal_cost):
            return probe_action
        if probe_cost + 1e-9 < terminal_cost:
            return probe_action
        return terminal_action

    def step(self, action: int) -> Tuple["FastenEnv", float, bool, Dict[str, Any]]:
        self.step_count += 1
        action = FastenAction(int(action))
        if action == FastenAction.RECOVER and not self.allow_recover:
            raise ValueError("RECOVER is disabled by the action-scope configuration")
        if action == FastenAction.FASTEN:
            reward, done, info = self._step_fasten()
        elif action == FastenAction.INSERT:
            reward, done, info = self._step_insert()
        elif action == FastenAction.RECOVER:
            reward, done, info = self._step_recover()
        elif action == FastenAction.PROBE_FIXED:
            reward, done, info = self._step_probe("fixed")
        elif action == FastenAction.PROBE_OPTIMAL:
            reward, done, info = self._step_probe("optimal")
        else:
            raise ValueError(f"unknown action: {action}")

        if self.step_count >= self.max_steps and not done:
            self.phase = "failed"
            done = True
            info["timeout"] = True
        return self, reward, done, info

    def _step_fasten(self) -> Tuple[float, bool, Dict[str, Any]]:
        if self.hole_state == STATE_ALIGN:
            self.phase = "done"
            reward = self.success_reward
            info = {"result": "success", "state": self.hole_state}
        else:
            self.phase = "failed"
            reward = -self.failure_penalty
            info = {"result": "failure", "state": self.hole_state}
        self.last_obs = None
        return reward, True, info

    def _step_insert(self) -> Tuple[float, bool, Dict[str, Any]]:
        before = self.hole_state
        if before != STATE_EMPTY:
            self.phase = "failed"
            self.last_obs = None
            return -self.failure_penalty, True, {
                "result": "insert_failure",
                "state_before": before,
                "state_after": self.hole_state,
            }

        self.hole_state = transition_after_insert(self.hole_state)
        self.belief = {
            STATE_EMPTY: 0.0,
            STATE_MISALIGN: 0.0,
            STATE_ALIGN: 1.0,
        }
        self.last_obs = ("insert", "success")
        return -self.insert_cost, False, {
            "result": "insert",
            "state_before": before,
            "state_after": self.hole_state,
        }

    def _step_recover(self) -> Tuple[float, bool, Dict[str, Any]]:
        before = self.hole_state
        if before == STATE_EMPTY:
            self.belief = belief_after_recover_not_applicable(self.belief)
            self.last_obs = ("recover", "not_applicable")
            return -self.recover_cost, False, {
                "result": "recover_not_applicable",
                "state_before": before,
                "state_after": self.hole_state,
            }
        self.hole_state = transition_after_recover(self.hole_state)
        self.belief = belief_after_recover_applicable(self.belief)
        self.last_obs = ("recover", "applicable")
        return -self.recover_cost, False, {
            "result": "recover",
            "state_before": before,
            "state_after": self.hole_state,
        }

    def _step_probe(self, probe_type: str) -> Tuple[float, bool, Dict[str, Any]]:
        obs = sample_disc_obs(self.hole_state, probe_type)
        before = dict(self.belief)
        self.belief = bayesian_update_belief(self.belief, obs, probe_type)
        self.last_obs = obs
        return -self._cost_for_probe_type(probe_type), False, {
            "result": f"probe_{probe_type}",
            "obs": obs,
            "state": self.hole_state,
            "belief_before": before,
            "belief_after": dict(self.belief),
        }


class FastenAgent:
    def __init__(self) -> None:
        self.next_action: Optional[int] = None
        self.smart_parameters: Dict[str, Any] = {}
