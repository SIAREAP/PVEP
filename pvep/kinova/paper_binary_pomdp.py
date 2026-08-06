"""Paper-faithful binary POMDP for post-insertion screw fastening.

This module implements Supplement S2.2 of ``中文版.pdf``.  It intentionally
does not model empty holes or screw insertion: an episode starts *after*
``insert_screw`` and the only hidden variable is placement quality
``q in {good, poor}``.

The image model must provide an observation likelihood, not an already-updated
belief.  For a categorical visual signal ``o``, ``Evidence.s_align`` means
``P(o | q=good)`` and ``Evidence.s_misalign`` means ``P(o | q=poor)``.  Equation
(S8) is then an ordinary Bayesian update and can safely be applied repeatedly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import Dict, Iterable, Mapping, Tuple


class Quality(str, Enum):
    GOOD = "good"
    POOR = "poor"


class Action(str, Enum):
    FASTEN = "FASTEN"
    PROBE_FIXED = "PROBE_FIXED"
    PROBE_OPTIMAL = "PROBE_OPTIMAL"
    RECOVER = "RECOVER"


class Signal(str, Enum):
    ALIGN_LOW = "ALIGN_LOW"
    ALIGN_MED = "ALIGN_MED"
    ALIGN_HIGH = "ALIGN_HIGH"
    MISALIGN_LOW = "MISALIGN_LOW"
    MISALIGN_MED = "MISALIGN_MED"
    MISALIGN_HIGH = "MISALIGN_HIGH"
    UNCERTAIN = "UNCERTAIN"


PROBE_ACTIONS = (Action.PROBE_FIXED, Action.PROBE_OPTIMAL)


@dataclass(frozen=True)
class Evidence:
    """Likelihood pair used by Eq. (S8), not a pair of class posteriors."""

    s_align: float
    s_misalign: float

    def __post_init__(self) -> None:
        for name, value in (
            ("s_align", self.s_align),
            ("s_misalign", self.s_misalign),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative, got {value}")
        if self.s_align + self.s_misalign <= 0.0:
            raise ValueError("at least one observation likelihood must be positive")


def bayes_update(prior_good: float, evidence: Evidence) -> float:
    """Apply Supplement Eq. (S8) and return ``P(q=good | observation)``."""

    if not 0.0 <= prior_good <= 1.0:
        raise ValueError(f"prior_good must be in [0, 1], got {prior_good}")
    numerator = evidence.s_align * prior_good
    denominator = numerator + evidence.s_misalign * (1.0 - prior_good)
    if denominator <= 1e-15:
        return prior_good
    return min(1.0, max(0.0, numerator / denominator))


def belief_to_text(
    belief_good: float,
    *,
    tau_low: float = 0.15,
    tau_high: float = 0.90,
) -> str:
    """Small Belief-to-Text bridge matching Supplement S1.1."""

    if belief_good >= tau_high:
        return f"placement quality is GOOD with high confidence (P={belief_good:.3f})"
    if belief_good <= tau_low:
        return f"placement quality is POOR with high confidence (P={1-belief_good:.3f})"
    return f"placement quality is uncertain (P(good)={belief_good:.3f})"


@dataclass(frozen=True)
class PaperConfig:
    prior_good: float = 0.80
    tau_goal: float = 0.90
    tau_low: float = 0.15
    tau_high: float = 0.90
    initial_visual_cost: float = 3.0
    probe_fixed_cost: float = 3.0
    probe_optimal_cost: float = 4.0
    recover_cost: float = 10.0
    fasten_failure_cost: float = 80.0
    gamma: float = 0.95
    max_steps: int = 10

    def __post_init__(self) -> None:
        for name in ("prior_good", "tau_goal", "tau_low", "tau_high", "gamma"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")

    def action_cost(self, action: Action) -> float:
        if action == Action.PROBE_FIXED:
            return self.probe_fixed_cost
        if action == Action.PROBE_OPTIMAL:
            return self.probe_optimal_cost
        if action == Action.RECOVER:
            return self.recover_cost
        raise ValueError(f"{action} has no state-independent action cost")


class ObservationModel:
    """Categorical ``O(o | q, a)`` for the two probing actions.

    ``tables[action][quality][signal]`` stores a conditional probability.  A
    small Laplace prior should be used when constructing a table from images so
    that a finite dataset never creates impossible evidence by accident.
    """

    def __init__(
        self,
        tables: Mapping[Action, Mapping[Quality, Mapping[Signal, float]]],
    ) -> None:
        normalized: Dict[Action, Dict[Quality, Dict[Signal, float]]] = {}
        for action in PROBE_ACTIONS:
            if action not in tables:
                raise ValueError(f"missing observation table for {action.value}")
            normalized[action] = {}
            for quality in Quality:
                raw = {signal: float(tables[action][quality].get(signal, 0.0)) for signal in Signal}
                if any((not math.isfinite(v) or v < 0.0) for v in raw.values()):
                    raise ValueError(f"invalid probabilities for {action.value}/{quality.value}")
                total = sum(raw.values())
                if total <= 0.0:
                    raise ValueError(f"empty distribution for {action.value}/{quality.value}")
                normalized[action][quality] = {
                    signal: raw[signal] / total for signal in Signal
                }
        self._tables = normalized

    @classmethod
    def from_counts(
        cls,
        counts: Mapping[Action, Mapping[Quality, Mapping[Signal, int]]],
        *,
        laplace: float = 1.0,
    ) -> "ObservationModel":
        if laplace < 0.0:
            raise ValueError("laplace must be non-negative")
        tables: Dict[Action, Dict[Quality, Dict[Signal, float]]] = {}
        for action in PROBE_ACTIONS:
            tables[action] = {}
            for quality in Quality:
                tables[action][quality] = {
                    signal: float(counts[action][quality].get(signal, 0)) + laplace
                    for signal in Signal
                }
        return cls(tables)

    def probability(self, action: Action, quality: Quality, signal: Signal) -> float:
        return self._tables[action][quality][signal]

    def evidence(self, action: Action, signal: Signal) -> Evidence:
        return Evidence(
            s_align=self.probability(action, Quality.GOOD, signal),
            s_misalign=self.probability(action, Quality.POOR, signal),
        )

    def signal_probability(self, action: Action, belief_good: float, signal: Signal) -> float:
        return (
            belief_good * self.probability(action, Quality.GOOD, signal)
            + (1.0 - belief_good) * self.probability(action, Quality.POOR, signal)
        )

    def posterior(self, action: Action, belief_good: float, signal: Signal) -> float:
        return bayes_update(belief_good, self.evidence(action, signal))

    def as_dict(self) -> dict:
        return {
            action.value: {
                quality.value: {
                    signal.value: self.probability(action, quality, signal)
                    for signal in Signal
                }
                for quality in Quality
            }
            for action in PROBE_ACTIONS
        }


@dataclass(frozen=True)
class Decision:
    action: Action
    expected_cost: float
    q_costs: Mapping[Action, float]
    semantic_gradient: str | None


class BinaryPOMDPSolver:
    """Finite-horizon belief-state solver for the paper's two-state POMDP.

    The hidden state is binary and the calibrated observation alphabet is
    small, so exact belief recursion is deterministic and easier to audit than
    Monte Carlo noise.  It solves the same finite-horizon decision problem that
    POMCP approximates for this special case.
    """

    def __init__(self, observation_model: ObservationModel, config: PaperConfig | None = None):
        self.observation_model = observation_model
        self.config = config or PaperConfig()

    def decide(
        self,
        belief_good: float,
        *,
        steps_remaining: int | None = None,
        fixed_left: int | None = None,
        optimal_left: int | None = None,
    ) -> Decision:
        if not 0.0 <= belief_good <= 1.0:
            raise ValueError("belief_good must be in [0, 1]")
        steps = self.config.max_steps if steps_remaining is None else int(steps_remaining)
        # ``None`` means the action may be selected on every remaining step.
        fixed = steps if fixed_left is None else max(0, int(fixed_left))
        optimal = steps if optimal_left is None else max(0, int(optimal_left))
        action, cost, q_costs = self._solve(round(belief_good, 12), steps, fixed, optimal)
        gradient = None
        if action in PROBE_ACTIONS and belief_good < self.config.tau_goal:
            gradient = (
                "∇ent: FASTEN rejected because placement-quality belief is below "
                f"tau_goal ({belief_good:.3f} < {self.config.tau_goal:.2f}); "
                f"request {action.value}."
            )
        elif action == Action.RECOVER and belief_good < self.config.tau_goal:
            gradient = (
                "∇inf: available visual evidence cannot justify safe FASTEN within "
                "the remaining budget; execute RECOVER."
            )
        return Decision(action, cost, q_costs, gradient)

    @lru_cache(maxsize=200_000)
    def _solve(
        self,
        belief_good: float,
        steps_remaining: int,
        fixed_left: int,
        optimal_left: int,
    ) -> Tuple[Action, float, Mapping[Action, float]]:
        cfg = self.config
        q_costs: Dict[Action, float] = {Action.RECOVER: cfg.recover_cost}

        # Verifier-enforced terminal gate from S1.2/S2.2.
        if belief_good >= cfg.tau_goal:
            q_costs[Action.FASTEN] = (1.0 - belief_good) * cfg.fasten_failure_cost

        if steps_remaining > 1:
            for action, left in (
                (Action.PROBE_FIXED, fixed_left),
                (Action.PROBE_OPTIMAL, optimal_left),
            ):
                if left <= 0:
                    continue
                future = 0.0
                next_fixed = fixed_left - int(action == Action.PROBE_FIXED)
                next_optimal = optimal_left - int(action == Action.PROBE_OPTIMAL)
                for signal in Signal:
                    obs_prob = self.observation_model.signal_probability(
                        action, belief_good, signal
                    )
                    if obs_prob <= 1e-15:
                        continue
                    posterior = self.observation_model.posterior(action, belief_good, signal)
                    _, next_cost, _ = self._solve(
                        round(posterior, 12),
                        steps_remaining - 1,
                        next_fixed,
                        next_optimal,
                    )
                    future += obs_prob * next_cost
                q_costs[action] = cfg.action_cost(action) + cfg.gamma * future

        priority = {
            Action.FASTEN: 0,
            Action.PROBE_OPTIMAL: 1,
            Action.PROBE_FIXED: 2,
            Action.RECOVER: 3,
        }
        action, cost = min(q_costs.items(), key=lambda item: (item[1], priority[item[0]]))
        return action, cost, dict(q_costs)


def quality_from_truth_label(label: str) -> Quality | None:
    """Map existing image annotations into the paper state space.

    Empty holes return ``None`` because they are outside the post-insertion
    problem rather than a third hidden state.
    """

    normalized = label.strip().upper()
    if normalized == "ALIGN":
        return Quality.GOOD
    if normalized == "MISALIGN":
        return Quality.POOR
    return None


def signal_from_prediction(label: str | None, confidence: float | None = None) -> Signal:
    """Convert a detector result into a calibrated categorical observation.

    Confidence is retained as part of the observation (low < .50, medium <
    .80, high otherwise), but is never interpreted directly as a posterior.
    Its likelihood is learned conditionally from calibration images.
    """

    normalized = (label or "").strip().upper()
    if confidence is None or not math.isfinite(float(confidence)):
        return Signal.UNCERTAIN
    confidence = float(confidence)
    suffix = "LOW" if confidence < 0.50 else "MED" if confidence < 0.80 else "HIGH"
    if normalized == "ALIGN":
        return Signal[f"ALIGN_{suffix}"]
    if normalized == "MISALIGN":
        return Signal[f"MISALIGN_{suffix}"]
    return Signal.UNCERTAIN


def count_observations(
    rows: Iterable[Tuple[Quality, Signal]],
) -> Dict[Quality, Dict[Signal, int]]:
    counts = {quality: {signal: 0 for signal in Signal} for quality in Quality}
    for quality, signal in rows:
        counts[quality][signal] += 1
    return counts
