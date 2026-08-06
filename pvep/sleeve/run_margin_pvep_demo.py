from __future__ import annotations

import csv
import copy
import math
import random
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from heat import theoretical_min_heating_temperature_c


class Action(str, Enum):
    HEAT_5 = "HEAT_5"
    INSPECT = "INSPECT"
    ASSEMBLE = "ASSEMBLE"
    SAFE_ABORT = "SAFE_ABORT"


class Observation(str, Enum):
    NONE = "NONE"
    TOO_SMALL = "TOO_SMALL"
    BORDERLINE = "BORDERLINE"
    IN_SPEC = "IN_SPEC"


@dataclass(frozen=True)
class HiddenState:
    delta_needed_c: int


@dataclass(frozen=True)
class MarginDemo:
    target_temp_c: int
    start_temp_c: int
    true_delta_needed_c: int
    assembly_cooling_c: int
    delta_bins_c: Tuple[int, ...]
    risk_threshold: float
    material: str = "generic"
    task_id: str = ""
    belief_scope: str = "broad"
    belief_window_c: int = 15


@dataclass(frozen=True)
class PolicyConfig:
    max_temp_c: int = 450
    max_steps: int = 80
    inspect_uncertainty_threshold: float = 0.22
    min_temp_between_inspects_c: int = 20


@dataclass(frozen=True)
class BeliefSummary:
    expected_delta_c: float
    delta_marginal: Dict[int, float]


@dataclass(frozen=True)
class TraceStep:
    step: int
    action: Action
    temp_before_c: int
    temp_after_c: int
    observation: Observation
    true_margin_now_c: int
    true_margin_at_assembly_c: int
    p_now_in_spec: float
    p_violate: float
    belief_summary: BeliefSummary
    top_belief: Tuple[Tuple[int, float], ...]
    violation: bool
    note: str


@dataclass(frozen=True)
class InspectEvent:
    task_id: str
    step: int
    material: str
    current_temp_c: int
    current_heat_c: int
    true_delta_needed_c: int
    margin_c: int
    obs: Observation
    belief_entropy_before: float
    belief_entropy_after: float
    inspect_model_type: str


@dataclass(frozen=True)
class EpisodeResult:
    success: bool
    aborted: bool
    rvr: int
    final_temp_c: int
    trace: Tuple[TraceStep, ...]
    inspect_events: Tuple[InspectEvent, ...] = ()


@dataclass(frozen=True)
class MethodResult:
    method: str
    description: str
    episode: EpisodeResult
    n_heat: int
    n_inspect: int
    n_assemble: int
    final_temp_c: int
    extra_heat_c: int
    total_cost: float
    policy_driven: bool
    planner: str
    belief_source: str = "none"
    initial_belief_entropy: float = 0.0
    initial_expected_delta_c: float = 0.0
    inspect_model_type: str = "none"
    prompt_history: bool = False


@dataclass(frozen=True)
class POMCPRewardConfig:
    success_reward: float = 100.0
    heat_cost_per_c: float = 1.0
    inspect_cost: float = 10.0
    assemble_cost: float = 50.0
    violation_penalty: float = 100.0
    safe_abort_cost: float = 500.0
    overheat_cost_per_c: float = 0.5


@dataclass(frozen=True)
class AggregateResult:
    method: str
    n_cases: int
    success_rate: float
    rvr_rate: float
    avg_heat_c: float
    avg_inspect: float
    avg_cost: float


@dataclass(frozen=True)
class MarginTaskRun:
    task_id: str
    material: str
    t_needed_c: float
    proposal_temp_c: int
    demo: MarginDemo
    method_result: MethodResult


@dataclass(frozen=True)
class MarginTaskAggregate:
    method: str
    n_tasks: int
    success_rate: float
    rvr_rate: float
    avg_final_temp_c: float
    avg_heat_c: float
    avg_inspect: float
    avg_cost: float


@dataclass(frozen=True)
class FixedReflowSweepResult:
    heat_c: int
    inspect: bool
    n_tasks: int
    success_rate: float
    rvr_rate: float
    avg_final_temp_c: float
    avg_heat_c: float
    avg_inspect: float
    avg_cost: float


Belief = Dict[HiddenState, float]
BELIEF_SCOPES: Tuple[str, ...] = ("broad", "narrow")
REFLOW_PROMPT_METHODS: Tuple[str, ...] = (
    "llm+reflow_no_pomcp",
    "llm+pomdp+reflow",
)
DEFAULT_REFLOW_HISTORY_C: Tuple[int, ...] = (20, 30, 30, 40, 40, 40, 50, 50)
DEFAULT_REFLOW_HISTORY_ROWS: Tuple[Mapping[str, Any], ...] = (
    {"material": "碳素钢", "delta_needed_c": 20},
    {"material": "碳素钢", "delta_needed_c": 30},
    {"material": "碳素钢", "delta_needed_c": 40},
    {"material": "铜合金", "delta_needed_c": 30},
    {"material": "铜合金", "delta_needed_c": 40},
    {"material": "铜合金", "delta_needed_c": 50},
    {"material": "铝合金", "delta_needed_c": 20},
    {"material": "铝合金", "delta_needed_c": 30},
)
HUMAN_FORMULA_MARGIN_C_BY_MATERIAL: Dict[str, float] = {
    "碳素钢": 30.0,
    "铜合金": 32.0,
    "铝合金": 28.0,
}
HUMAN_FORMULA_LINEAR_COEFFICIENT = 1.025
HUMAN_FORMULA_SCALE_BY_MATERIAL: Dict[str, float] = {
    "碳素钢": 0.995,
    "铜合金": 1.010,
    "铝合金": 0.995,
}


def _ceil_to_step_c(value_c: float, *, step_c: int = 5) -> int:
    return int(math.ceil(float(value_c) / float(step_c)) * int(step_c))


def human_formula_proposal_temp_c(
    row: Mapping[str, Any],
    *,
    room_temp_c: int = 16,
    step_c: int = 5,
    min_temp_c: int = 250,
    max_temp_c: int = 450,
) -> int:
    """Engineer baseline with a calibrated linear model and material margin."""
    material = str(row.get("material", ""))
    t_theory = theoretical_min_heating_temperature_c(
        initial_inner_diameter_mm=float(row["initial_inner_mm"]),
        target_shaft_diameter_mm=float(row["target_shaft_mm"]),
        material=str(material),
        room_temperature_c=float(room_temp_c),
    )
    margin = float(HUMAN_FORMULA_MARGIN_C_BY_MATERIAL.get(str(material), 30.0))
    material_scale = float(HUMAN_FORMULA_SCALE_BY_MATERIAL.get(str(material), 1.0))
    modeled_rise_c = (
        float(HUMAN_FORMULA_LINEAR_COEFFICIENT)
        * float(material_scale)
        * (float(t_theory) - float(room_temp_c))
    )
    temp_c = _ceil_to_step_c(
        float(room_temp_c) + float(modeled_rise_c) + float(margin),
        step_c=int(step_c),
    )
    return int(max(int(min_temp_c), min(int(max_temp_c), int(temp_c))))


def build_default_demo() -> MarginDemo:
    """A deliberately hard corrupted proposal: target=300C, true requirement=340C."""
    return build_demo_for_delta(40)


def build_demo_for_delta(delta_needed_c: int) -> MarginDemo:
    return MarginDemo(
        target_temp_c=300,
        start_temp_c=300,
        true_delta_needed_c=int(delta_needed_c),
        assembly_cooling_c=10,
        delta_bins_c=tuple(range(-50, 55, 5)),
        risk_threshold=0.0,
        material="generic",
        task_id="demo",
    )


def delta_needed_from_threshold(
    *,
    threshold_c: float,
    proposal_temp_c: int,
    assembly_cooling_c: int,
    step_c: int = 5,
    cap_c: int = 50,
) -> int:
    _ = assembly_cooling_c
    raw_delta = float(threshold_c) - float(proposal_temp_c)
    binned = int(math.ceil(float(raw_delta) / float(step_c)) * int(step_c))
    return max(-int(cap_c), min(int(cap_c), int(binned)))


def build_demo_from_task_row(
    row: Mapping[str, Any],
    *,
    proposal_temp_c: int = 300,
    assembly_cooling_c: int = 10,
    delta_step_c: int = 5,
    required_temp_min_c: int = 250,
    required_temp_max_c: int = 440,
    risk_threshold: float = 0.3,
    belief_scope: str = "broad",
    belief_window_c: int = 15,
) -> MarginDemo:
    step_c = int(delta_step_c)
    if step_c <= 0:
        raise ValueError("delta_step_c must be positive")
    min_required_c = _ceil_to_step_c(float(required_temp_min_c), step_c=step_c)
    max_required_c = int(math.floor(float(required_temp_max_c) / step_c) * step_c)
    if min_required_c > max_required_c:
        raise ValueError("required temperature prior must contain at least one bin")

    broad_required_temp_bins_c = tuple(
        range(int(min_required_c), int(max_required_c) + step_c, step_c)
    )
    scope = str(belief_scope).strip().lower()
    if scope not in set(BELIEF_SCOPES):
        raise ValueError(
            f"belief_scope must be one of {BELIEF_SCOPES}, got {belief_scope!r}"
        )
    window_c = int(belief_window_c)
    if window_c < 0:
        raise ValueError("belief_window_c must be non-negative")

    if scope == "broad":
        required_temp_bins_c = broad_required_temp_bins_c
    else:
        # Keep the original absolute 5 C grid and only truncate its support.
        # For example, proposal=351 and window=15 retains 340,...,365 C.
        lower_c = int(proposal_temp_c) - int(window_c)
        upper_c = int(proposal_temp_c) + int(window_c)
        required_temp_bins_c = tuple(
            int(required_temp_c)
            for required_temp_c in broad_required_temp_bins_c
            if int(lower_c) <= int(required_temp_c) <= int(upper_c)
        )
        if not required_temp_bins_c:
            raise ValueError(
                "narrow belief support contains no global temperature-grid bin; "
                "increase belief_window_c or adjust the proposal/prior bounds"
            )

    true_required_temp_c = _ceil_to_step_c(float(row["t_needed_c"]), step_c=step_c)
    delta = int(true_required_temp_c) - int(proposal_temp_c)
    return MarginDemo(
        target_temp_c=int(proposal_temp_c),
        start_temp_c=int(proposal_temp_c),
        true_delta_needed_c=int(delta),
        assembly_cooling_c=int(assembly_cooling_c),
        delta_bins_c=tuple(
            int(required_temp_c) - int(proposal_temp_c)
            for required_temp_c in required_temp_bins_c
        ),
        risk_threshold=float(risk_threshold),
        material=str(row.get("material", "generic")),
        task_id=str(row.get("task_id", "")),
        belief_scope=str(scope),
        belief_window_c=int(window_c),
    )


def default_actions() -> Tuple[Action, ...]:
    return (
        Action.INSPECT,
        Action.HEAT_5,
        Action.HEAT_5,
        Action.HEAT_5,
        Action.HEAT_5,
        Action.INSPECT,
        Action.HEAT_5,
        Action.HEAT_5,
        Action.HEAT_5,
        Action.HEAT_5,
        Action.INSPECT,
        Action.HEAT_5,
        Action.HEAT_5,
        Action.INSPECT,
        Action.ASSEMBLE,
    )


def initial_belief(demo: MarginDemo) -> Belief:
    states = [HiddenState(delta_needed_c=int(delta)) for delta in demo.delta_bins_c]
    p = 1.0 / float(len(states))
    return {s: p for s in states}


def _normalize(weights: Mapping[HiddenState, float]) -> Belief:
    total = float(sum(max(0.0, float(v)) for v in weights.values()))
    if total <= 0.0:
        states = list(weights)
        if not states:
            return {}
        p = 1.0 / float(len(states))
        return {s: p for s in states}
    return {s: max(0.0, float(v)) / total for s, v in weights.items()}


def inspect_observation_from_margin(margin_c: int) -> Observation:
    if int(margin_c) < 0:
        return Observation.TOO_SMALL
    if int(margin_c) < 10:
        return Observation.BORDERLINE
    return Observation.IN_SPEC


def inspect_likelihood(observation: Observation, margin_c: int) -> float:
    """Noisy three-bin verifier model used only for belief updates."""
    if int(margin_c) < 0:
        probs = {
            Observation.TOO_SMALL: 0.85,
            Observation.BORDERLINE: 0.15,
            Observation.IN_SPEC: 0.0,
        }
    elif int(margin_c) < 10:
        probs = {
            Observation.TOO_SMALL: 0.15,
            Observation.BORDERLINE: 0.70,
            Observation.IN_SPEC: 0.15,
        }
    else:
        probs = {
            Observation.TOO_SMALL: 0.0,
            Observation.BORDERLINE: 0.10,
            Observation.IN_SPEC: 0.90,
        }
    return float(probs.get(observation, 0.0))


@dataclass(frozen=True)
class MaterialInspectModel:
    model_type: str = "constant"

    def _correct_probability(self, *, margin_c: int, material: str, temp_c: int) -> float:
        if self.model_type == "constant":
            if int(margin_c) < 0:
                return 0.85
            if int(margin_c) < 10:
                return 0.70
            return 0.90

        material_factor = {
            "generic": 0.94,
            "碳素钢": 1.00,
            "铜合金": 0.92,
            "铝合金": 0.86,
        }.get(str(material), 0.90)
        if int(temp_c) < 320:
            temp_factor = 1.00
        elif int(temp_c) < 360:
            temp_factor = 0.93
        else:
            temp_factor = 0.86
        if int(margin_c) < 0:
            base = 0.88
        elif int(margin_c) < 10:
            base = 0.72
        else:
            base = 0.90
        return max(0.45, min(0.95, float(base) * float(material_factor) * float(temp_factor)))

    def likelihood(
        self,
        observation: Observation,
        *,
        margin_c: int,
        material: str,
        temp_c: int,
    ) -> float:
        if self.model_type == "constant":
            return inspect_likelihood(observation, int(margin_c))

        correct = self._correct_probability(margin_c=int(margin_c), material=str(material), temp_c=int(temp_c))
        if int(margin_c) < 0:
            ideal = Observation.TOO_SMALL
            spill = {
                Observation.BORDERLINE: 0.85,
                Observation.IN_SPEC: 0.15,
            }
        elif int(margin_c) < 10:
            ideal = Observation.BORDERLINE
            spill = {
                Observation.TOO_SMALL: 0.50,
                Observation.IN_SPEC: 0.50,
            }
        else:
            ideal = Observation.IN_SPEC
            spill = {
                Observation.BORDERLINE: 0.85,
                Observation.TOO_SMALL: 0.15,
            }
        if observation == ideal:
            return float(correct)
        return float(1.0 - correct) * float(spill.get(observation, 0.0))

    def sample(
        self,
        *,
        margin_c: int,
        material: str,
        temp_c: int,
        rng: random.Random | None = None,
    ) -> Observation:
        r = (rng.random() if rng is not None else random.random())
        cdf = 0.0
        fallback = Observation.IN_SPEC
        for obs in (Observation.TOO_SMALL, Observation.BORDERLINE, Observation.IN_SPEC):
            fallback = obs
            cdf += self.likelihood(obs, margin_c=int(margin_c), material=str(material), temp_c=int(temp_c))
            if r <= cdf:
                return obs
        return fallback


def update_belief_after_inspect(
    belief: Belief,
    *,
    temp_c: int,
    target_temp_c: int,
    assembly_cooling_c: int = 0,
    observation: Observation,
    material: str = "generic",
    inspect_model: MaterialInspectModel | None = None,
) -> Belief:
    model = inspect_model or MaterialInspectModel(model_type="constant")
    weights: Dict[HiddenState, float] = {}
    for state, prob in belief.items():
        margin_now = (
            int(temp_c)
            - int(assembly_cooling_c)
            - int(target_temp_c)
            - int(state.delta_needed_c)
        )
        weights[state] = float(prob) * model.likelihood(
            observation,
            margin_c=int(margin_now),
            material=str(material),
            temp_c=int(temp_c),
        )
    return _normalize(weights)


def belief_entropy(belief: Belief) -> float:
    h = 0.0
    for prob in belief.values():
        p = float(prob)
        if p > 0.0:
            h -= p * math.log(p, 2)
    return float(h)


def summarize_belief(belief: Belief) -> BeliefSummary:
    delta_marginal: Dict[int, float] = {}
    for state, prob in belief.items():
        delta = int(state.delta_needed_c)
        delta_marginal[delta] = float(delta_marginal.get(delta, 0.0)) + float(prob)

    expected_delta = sum(float(k) * float(v) for k, v in delta_marginal.items())
    return BeliefSummary(
        expected_delta_c=float(expected_delta),
        delta_marginal={int(k): float(v) for k, v in sorted(delta_marginal.items())},
    )


def probability_now_in_spec(
    belief: Belief,
    *,
    temp_c: int,
    target_temp_c: int,
) -> float:
    return float(
        sum(
            prob
            for state, prob in belief.items()
            if int(temp_c) - int(target_temp_c) - int(state.delta_needed_c) >= 0
        )
    )


def probability_violate_on_assemble(
    belief: Belief,
    *,
    temp_c: int,
    target_temp_c: int,
    assembly_cooling_c: int,
) -> float:
    return float(
        sum(
            prob
            for state, prob in belief.items()
            if int(temp_c)
            - int(assembly_cooling_c)
            - int(target_temp_c)
            - int(state.delta_needed_c)
            < 0
        )
    )


def predicted_observation_distribution(
    belief: Belief,
    *,
    temp_c: int,
    target_temp_c: int,
    assembly_cooling_c: int = 0,
) -> Dict[Observation, float]:
    out = {
        Observation.TOO_SMALL: 0.0,
        Observation.BORDERLINE: 0.0,
        Observation.IN_SPEC: 0.0,
    }
    for state, prob in belief.items():
        margin_now = (
            int(temp_c)
            - int(assembly_cooling_c)
            - int(target_temp_c)
            - int(state.delta_needed_c)
        )
        for obs in out:
            out[obs] += float(prob) * inspect_likelihood(obs, margin_now)
    return {obs: float(p) for obs, p in out.items()}


def inspect_uncertainty(
    belief: Belief,
    *,
    temp_c: int,
    target_temp_c: int,
    assembly_cooling_c: int = 0,
) -> float:
    dist = predicted_observation_distribution(
        belief,
        temp_c=int(temp_c),
        target_temp_c=int(target_temp_c),
        assembly_cooling_c=int(assembly_cooling_c),
    )
    return float(1.0 - max(float(p) for p in dist.values()))


def choose_policy_action(
    demo: MarginDemo,
    belief: Belief,
    *,
    temp_c: int,
    inspected_temps_c: Sequence[int],
    config: PolicyConfig = PolicyConfig(),
) -> Action:
    p_violate = probability_violate_on_assemble(
        belief,
        temp_c=int(temp_c),
        target_temp_c=int(demo.target_temp_c),
        assembly_cooling_c=int(demo.assembly_cooling_c),
    )
    if float(p_violate) <= float(demo.risk_threshold):
        return Action.ASSEMBLE
    if int(temp_c) >= int(config.max_temp_c):
        return Action.SAFE_ABORT

    last_inspect = max((int(t) for t in inspected_temps_c), default=-10_000)
    can_inspect_here = int(temp_c) not in set(int(t) for t in inspected_temps_c)
    enough_temp_gap = int(temp_c) - int(last_inspect) >= int(config.min_temp_between_inspects_c)
    uncertainty = inspect_uncertainty(
        belief,
        temp_c=int(temp_c),
        target_temp_c=int(demo.target_temp_c),
        assembly_cooling_c=int(demo.assembly_cooling_c),
    )
    p_now_ok = probability_now_in_spec(
        belief,
        temp_c=int(temp_c),
        target_temp_c=int(demo.target_temp_c),
    )
    informative_now = float(uncertainty) >= float(config.inspect_uncertainty_threshold)
    in_decision_band = bool(p_now_ok > 0.0) or not inspected_temps_c
    if bool(can_inspect_here and enough_temp_gap and informative_now and in_decision_band):
        return Action.INSPECT
    return Action.HEAT_5


def top_belief_states(belief: Belief, k: int = 3) -> Tuple[Tuple[int, float], ...]:
    ranked = sorted(
        belief.items(),
        key=lambda item: (-float(item[1]), int(item[0].delta_needed_c)),
    )
    return tuple(
        (int(state.delta_needed_c), float(prob))
        for state, prob in ranked[: int(k)]
    )


def _sample_delta_from_belief(belief: Belief) -> int:
    r = random.random()
    cdf = 0.0
    fallback = 0
    for state, prob in sorted(belief.items(), key=lambda item: int(item[0].delta_needed_c)):
        fallback = int(state.delta_needed_c)
        cdf += float(prob)
        if r <= cdf:
            return int(state.delta_needed_c)
    return int(fallback)


class MarginStateSet:
    def is_final_state(self, state: "MarginPOMCPEnv") -> bool:
        return bool(state.done)


class MarginPOMCPAgent:
    def __init__(self) -> None:
        self.next_action: Optional[Action] = None
        self.smart_parameters: Dict[str, Any] = {}


@dataclass
class MarginPOMCPEnv:
    demo: MarginDemo
    belief: Belief
    delta_needed_c: int
    temp_c: int
    config: PolicyConfig = field(default_factory=PolicyConfig)
    reward_config: POMCPRewardConfig = field(default_factory=POMCPRewardConfig)
    inspect_model: MaterialInspectModel = field(default_factory=lambda: MaterialInspectModel(model_type="calibrated"))
    last_obs: Observation = Observation.NONE
    step_count: int = 0
    done: bool = False
    success: bool = False
    aborted: bool = False
    rvr: int = 0
    simulation: bool = False
    inspected_temps_c: Tuple[int, ...] = ()
    state_set: MarginStateSet = field(default_factory=MarginStateSet)
    real_observation_rng: random.Random | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def copy(self) -> "MarginPOMCPEnv":
        return copy.deepcopy(self)

    def sample_state(self, _agent: Any = None) -> "MarginPOMCPEnv":
        state = self.copy()
        state.delta_needed_c = _sample_delta_from_belief(state.belief)
        return state

    def get_observation(self) -> Tuple[int, str, int]:
        return (int(self.temp_c), self.last_obs.value, int(self.done))

    def observation_is_equal(self, other: Any, _agent: Any = None) -> bool:
        return self.get_observation() == other

    def hash_observation(self, _agent: Any = None) -> int:
        return hash(self.get_observation())

    def _p_violate(self) -> float:
        return probability_violate_on_assemble(
            self.belief,
            temp_c=int(self.temp_c),
            target_temp_c=int(self.demo.target_temp_c),
            assembly_cooling_c=int(self.demo.assembly_cooling_c),
        )

    def _can_inspect(self) -> bool:
        return int(self.temp_c) not in set(int(t) for t in self.inspected_temps_c)

    def get_actions_list(self) -> List[Action]:
        if bool(self.done) or int(self.step_count) >= int(self.config.max_steps):
            return []

        actions: List[Action] = []
        assemble_allowed = float(self._p_violate()) <= float(self.demo.risk_threshold)
        if int(self.temp_c) >= int(self.config.max_temp_c):
            if self._can_inspect():
                actions.append(Action.INSPECT)
            if bool(assemble_allowed):
                actions.append(Action.ASSEMBLE)
            else:
                actions.append(Action.SAFE_ABORT)
            return actions

        actions = [Action.HEAT_5]
        if self._can_inspect():
            actions.append(Action.INSPECT)
        if bool(assemble_allowed):
            actions.append(Action.ASSEMBLE)
        return actions

    def _expected_assemble_reward(self) -> float:
        expected = 0.0
        for state, prob in self.belief.items():
            delta = int(state.delta_needed_c)
            margin_at_assembly = (
                int(self.temp_c)
                - int(self.demo.assembly_cooling_c)
                - int(self.demo.target_temp_c)
                - int(delta)
            )
            true_threshold = (
                int(self.demo.target_temp_c)
                + int(delta)
                + int(self.demo.assembly_cooling_c)
            )
            if int(margin_at_assembly) >= 0:
                overheat_c = max(0, int(self.temp_c) - int(true_threshold))
                reward = (
                    float(self.reward_config.success_reward)
                    - float(self.reward_config.assemble_cost)
                    - float(overheat_c) * float(self.reward_config.overheat_cost_per_c)
                )
            else:
                reward = -(
                    float(self.reward_config.assemble_cost)
                    + float(self.reward_config.violation_penalty)
                )
            expected += float(prob) * float(reward)
        return float(expected)

    def default_policy(self) -> Action:
        actions = self.get_actions_list()
        if not actions:
            return Action.SAFE_ABORT
        expected_assemble_reward = (
            self._expected_assemble_reward()
            if Action.ASSEMBLE in actions
            else float("-inf")
        )
        abort_reward = (
            -float(self.reward_config.safe_abort_cost)
            if Action.SAFE_ABORT in actions
            else float("-inf")
        )
        if Action.SAFE_ABORT in actions and int(self.temp_c) >= int(self.config.max_temp_c):
            if float(expected_assemble_reward) >= float(abort_reward):
                return Action.ASSEMBLE
            return Action.SAFE_ABORT
        uncertainty = inspect_uncertainty(
            self.belief,
            temp_c=int(self.temp_c),
            target_temp_c=int(self.demo.target_temp_c),
            assembly_cooling_c=int(self.demo.assembly_cooling_c),
        )
        p_now_ok = probability_now_in_spec(
            self.belief,
            temp_c=int(self.temp_c),
            target_temp_c=int(self.demo.target_temp_c),
        )
        if (
            Action.INSPECT in actions
            and float(uncertainty) >= float(self.config.inspect_uncertainty_threshold)
            and (float(p_now_ok) > 0.0 or not self.inspected_temps_c)
        ):
            return Action.INSPECT
        if Action.HEAT_5 in actions:
            return Action.HEAT_5
        if (
            Action.ASSEMBLE in actions
            and float(expected_assemble_reward) >= float(abort_reward)
        ):
            return Action.ASSEMBLE
        if Action.SAFE_ABORT in actions:
            return Action.SAFE_ABORT
        return actions[0]

    def step(self, action: Action) -> Tuple["MarginPOMCPEnv", float, bool, Dict[str, Any]]:
        action = Action(action)
        self.step_count += 1
        reward = 0.0
        info: Dict[str, Any] = {"action": action.value}

        if action == Action.HEAT_5:
            self.temp_c = int(self.temp_c) + 5
            self.last_obs = Observation.NONE
            reward = -5.0 * float(self.reward_config.heat_cost_per_c)
            info["result"] = "heated"

        elif action == Action.INSPECT:
            margin_now = (
                int(self.temp_c)
                - int(self.demo.assembly_cooling_c)
                - int(self.demo.target_temp_c)
                - int(self.delta_needed_c)
            )
            self.last_obs = self.inspect_model.sample(
                margin_c=int(margin_now),
                material=str(self.demo.material),
                temp_c=int(self.temp_c),
                # POMCP simulations keep using the planner's global RNG. Only
                # real-world observations use the independent external stream,
                # so search rollouts cannot advance the paired observation noise.
                rng=(None if bool(self.simulation) else self.real_observation_rng),
            )
            self.inspected_temps_c = tuple(self.inspected_temps_c) + (int(self.temp_c),)
            self.belief = update_belief_after_inspect(
                self.belief,
                temp_c=int(self.temp_c),
                target_temp_c=int(self.demo.target_temp_c),
                assembly_cooling_c=int(self.demo.assembly_cooling_c),
                observation=self.last_obs,
                material=str(self.demo.material),
                inspect_model=self.inspect_model,
            )
            reward = -float(self.reward_config.inspect_cost)
            info.update({"result": "inspected", "obs": self.last_obs.value})

        elif action == Action.ASSEMBLE:
            margin_at_assembly = (
                int(self.temp_c)
                - int(self.demo.assembly_cooling_c)
                - int(self.demo.target_temp_c)
                - int(self.delta_needed_c)
            )
            self.done = True
            self.success = bool(margin_at_assembly >= 0)
            self.rvr = 0 if bool(self.success) else 1
            self.last_obs = Observation.NONE
            true_threshold = (
                int(self.demo.target_temp_c)
                + int(self.delta_needed_c)
                + int(self.demo.assembly_cooling_c)
            )
            overheat_c = max(0, int(self.temp_c) - int(true_threshold))
            if self.success:
                reward = (
                    float(self.reward_config.success_reward)
                    - float(self.reward_config.assemble_cost)
                    - float(overheat_c) * float(self.reward_config.overheat_cost_per_c)
                )
                info["result"] = "success"
            else:
                reward = -(
                    float(self.reward_config.assemble_cost)
                    + float(self.reward_config.violation_penalty)
                )
                info["result"] = "violation"

        elif action == Action.SAFE_ABORT:
            self.done = True
            self.success = False
            self.aborted = True
            self.rvr = 0
            self.last_obs = Observation.NONE
            reward = -float(self.reward_config.safe_abort_cost)
            info["result"] = "safe_abort"

        else:
            raise ValueError(f"Unsupported action: {action!r}")

        if int(self.step_count) >= int(self.config.max_steps) and not bool(self.done):
            self.done = True
            self.aborted = True
            self.rvr = 0
            reward -= float(self.reward_config.safe_abort_cost)
            info["timeout"] = True

        return self, float(reward), bool(self.done), info


def run_episode(
    demo: MarginDemo,
    actions: Sequence[Action] | None = None,
    *,
    belief0: Belief | None = None,
) -> EpisodeResult:
    temp = int(demo.start_temp_c)
    belief = dict(belief0) if belief0 is not None else initial_belief(demo)
    trace: List[TraceStep] = []
    inspect_events: List[InspectEvent] = []
    success = False
    aborted = False
    rvr = 0
    done = False

    for idx, action in enumerate(tuple(actions or default_actions()), start=1):
        if done:
            break

        action = Action(action)
        temp_before = int(temp)
        observation = Observation.NONE
        violation = False
        note = ""

        if action == Action.HEAT_5:
            temp += 5
            note = "repair action: add 5C margin"
        elif action == Action.INSPECT:
            true_margin_now = (
                int(temp)
                - int(demo.assembly_cooling_c)
                - int(demo.target_temp_c)
                - int(demo.true_delta_needed_c)
            )
            observation = inspect_observation_from_margin(true_margin_now)
            entropy_before = belief_entropy(belief)
            belief = update_belief_after_inspect(
                belief,
                temp_c=int(temp),
                target_temp_c=int(demo.target_temp_c),
                assembly_cooling_c=int(demo.assembly_cooling_c),
                observation=observation,
                material=str(demo.material),
                inspect_model=MaterialInspectModel(model_type="constant"),
            )
            inspect_events.append(
                InspectEvent(
                    task_id=str(demo.task_id),
                    step=int(idx),
                    material=str(demo.material),
                    current_temp_c=int(temp),
                    current_heat_c=int(temp) - int(demo.start_temp_c),
                    true_delta_needed_c=int(demo.true_delta_needed_c),
                    margin_c=int(true_margin_now),
                    obs=observation,
                    belief_entropy_before=float(entropy_before),
                    belief_entropy_after=belief_entropy(belief),
                    inspect_model_type="constant",
                )
            )
            note = "verifier action: updates P(delta); no RVR recorded"
        elif action == Action.ASSEMBLE:
            true_margin_at_assembly = (
                int(temp)
                - int(demo.assembly_cooling_c)
                - int(demo.target_temp_c)
                - int(demo.true_delta_needed_c)
            )
            success = bool(true_margin_at_assembly >= 0)
            violation = not bool(success)
            rvr = 0 if bool(success) else 1
            done = True
            note = "terminal action: RVR is evaluated here"
        elif action == Action.SAFE_ABORT:
            aborted = True
            success = False
            rvr = 0
            done = True
            note = "terminal safety exit: no success, no RVR"
        else:
            raise ValueError(f"Unsupported action: {action!r}")

        true_margin_now_after = (
            int(temp) - int(demo.target_temp_c) - int(demo.true_delta_needed_c)
        )
        true_margin_at_assembly_after = (
            int(temp)
            - int(demo.assembly_cooling_c)
            - int(demo.target_temp_c)
            - int(demo.true_delta_needed_c)
        )
        p_violate = probability_violate_on_assemble(
            belief,
            temp_c=int(temp),
            target_temp_c=int(demo.target_temp_c),
            assembly_cooling_c=int(demo.assembly_cooling_c),
        )
        trace.append(
            TraceStep(
                step=int(idx),
                action=action,
                temp_before_c=int(temp_before),
                temp_after_c=int(temp),
                observation=observation,
                true_margin_now_c=int(true_margin_now_after),
                true_margin_at_assembly_c=int(true_margin_at_assembly_after),
                p_now_in_spec=probability_now_in_spec(
                    belief,
                    temp_c=int(temp),
                    target_temp_c=int(demo.target_temp_c),
                ),
                p_violate=float(p_violate),
                belief_summary=summarize_belief(belief),
                top_belief=top_belief_states(belief),
                violation=bool(violation),
                note=str(note),
            )
        )

    return EpisodeResult(
        success=bool(success),
        aborted=bool(aborted),
        rvr=int(rvr),
        final_temp_c=int(temp),
        trace=tuple(trace),
        inspect_events=tuple(inspect_events),
    )


def run_policy_episode(
    demo: MarginDemo,
    *,
    belief0: Belief | None = None,
    config: PolicyConfig = PolicyConfig(),
) -> EpisodeResult:
    temp = int(demo.start_temp_c)
    belief = dict(belief0) if belief0 is not None else initial_belief(demo)
    trace: List[TraceStep] = []
    inspected_temps: List[int] = []
    success = False
    aborted = False
    rvr = 0
    done = False

    for idx in range(1, int(config.max_steps) + 1):
        if done:
            break
        action = choose_policy_action(
            demo,
            belief,
            temp_c=int(temp),
            inspected_temps_c=tuple(inspected_temps),
            config=config,
        )
        temp_before = int(temp)
        observation = Observation.NONE
        violation = False
        note = "policy-selected action"

        if action == Action.HEAT_5:
            temp += 5
            note = "policy repair: P_violate too high, add 5C"
        elif action == Action.INSPECT:
            inspected_temps.append(int(temp))
            true_margin_now = (
                int(temp)
                - int(demo.assembly_cooling_c)
                - int(demo.target_temp_c)
                - int(demo.true_delta_needed_c)
            )
            observation = inspect_observation_from_margin(true_margin_now)
            belief = update_belief_after_inspect(
                belief,
                temp_c=int(temp),
                target_temp_c=int(demo.target_temp_c),
                assembly_cooling_c=int(demo.assembly_cooling_c),
                observation=observation,
            )
            note = "policy verifier: update P(delta), no RVR"
        elif action == Action.ASSEMBLE:
            true_margin_at_assembly = (
                int(temp)
                - int(demo.assembly_cooling_c)
                - int(demo.target_temp_c)
                - int(demo.true_delta_needed_c)
            )
            success = bool(true_margin_at_assembly >= 0)
            violation = not bool(success)
            rvr = 0 if bool(success) else 1
            done = True
            note = "policy execute: tail-risk gate passed"
        elif action == Action.SAFE_ABORT:
            aborted = True
            success = False
            rvr = 0
            done = True
            note = "policy safety exit: cannot satisfy risk gate"

        true_margin_now_after = (
            int(temp) - int(demo.target_temp_c) - int(demo.true_delta_needed_c)
        )
        true_margin_at_assembly_after = (
            int(temp)
            - int(demo.assembly_cooling_c)
            - int(demo.target_temp_c)
            - int(demo.true_delta_needed_c)
        )
        p_violate = probability_violate_on_assemble(
            belief,
            temp_c=int(temp),
            target_temp_c=int(demo.target_temp_c),
            assembly_cooling_c=int(demo.assembly_cooling_c),
        )
        trace.append(
            TraceStep(
                step=int(idx),
                action=action,
                temp_before_c=int(temp_before),
                temp_after_c=int(temp),
                observation=observation,
                true_margin_now_c=int(true_margin_now_after),
                true_margin_at_assembly_c=int(true_margin_at_assembly_after),
                p_now_in_spec=probability_now_in_spec(
                    belief,
                    temp_c=int(temp),
                    target_temp_c=int(demo.target_temp_c),
                ),
                p_violate=float(p_violate),
                belief_summary=summarize_belief(belief),
                top_belief=top_belief_states(belief),
                violation=bool(violation),
                note=str(note),
            )
        )

    if not bool(done):
        trace.append(
            TraceStep(
                step=len(trace) + 1,
                action=Action.SAFE_ABORT,
                temp_before_c=int(temp),
                temp_after_c=int(temp),
                observation=Observation.NONE,
                true_margin_now_c=int(temp) - int(demo.target_temp_c) - int(demo.true_delta_needed_c),
                true_margin_at_assembly_c=(
                    int(temp)
                    - int(demo.assembly_cooling_c)
                    - int(demo.target_temp_c)
                    - int(demo.true_delta_needed_c)
                ),
                p_now_in_spec=probability_now_in_spec(
                    belief,
                    temp_c=int(temp),
                    target_temp_c=int(demo.target_temp_c),
                ),
                p_violate=probability_violate_on_assemble(
                    belief,
                    temp_c=int(temp),
                    target_temp_c=int(demo.target_temp_c),
                    assembly_cooling_c=int(demo.assembly_cooling_c),
                ),
                belief_summary=summarize_belief(belief),
                top_belief=top_belief_states(belief),
                violation=False,
                note="max steps reached; safe abort",
            )
        )
        aborted = True

    return EpisodeResult(
        success=bool(success),
        aborted=bool(aborted),
        rvr=int(rvr),
        final_temp_c=int(temp),
        trace=tuple(trace),
    )


def _load_pomcp_planning() -> Any:
    try:
        from ibpomcp.pomcp import pomcp_planning

        return pomcp_planning
    except ModuleNotFoundError as exc:
        if exc.name != "ibpomcp":
            raise
        import sys
        from pathlib import Path

        repo_root = str(Path(__file__).resolve().parents[1])
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        from ibpomcp.pomcp import pomcp_planning

        return pomcp_planning


def run_pomcp_episode(
    demo: MarginDemo,
    *,
    belief0: Belief | None = None,
    config: PolicyConfig = PolicyConfig(),
    reward_config: POMCPRewardConfig = POMCPRewardConfig(),
    inspect_model: MaterialInspectModel = MaterialInspectModel(model_type="calibrated"),
    seed: Optional[int] = 0,
    observation_seed: Optional[int] = None,
    pomcp_max_depth: int = 20,
    pomcp_max_it: int = 200,
    pomcp_particles: int = 200,
    pomcp_discount: float = 0.95,
) -> EpisodeResult:
    pomcp_planning = _load_pomcp_planning()

    if seed is not None:
        random.seed(int(seed))
    external_observation_seed = (
        int(observation_seed)
        if observation_seed is not None
        else (int(seed) if seed is not None else None)
    )
    real_observation_rng = (
        random.Random(int(external_observation_seed))
        if external_observation_seed is not None
        else None
    )

    belief = dict(belief0) if belief0 is not None else initial_belief(demo)
    env = MarginPOMCPEnv(
        demo=demo,
        belief=belief,
        delta_needed_c=int(demo.true_delta_needed_c),
        temp_c=int(demo.start_temp_c),
        config=config,
        reward_config=reward_config,
        inspect_model=inspect_model,
        real_observation_rng=real_observation_rng,
    )
    agent = MarginPOMCPAgent()
    trace: List[TraceStep] = []
    inspect_events: List[InspectEvent] = []

    for idx in range(1, int(config.max_steps) + 1):
        if bool(env.done):
            break

        available_actions = env.get_actions_list()
        if not available_actions:
            action = Action.SAFE_ABORT
        else:
            action, plan_info = pomcp_planning(
                env,
                agent,
                max_depth=int(pomcp_max_depth),
                max_it=int(pomcp_max_it),
                discount_factor=float(pomcp_discount),
                k=int(pomcp_particles),
                particle_revigoration=True,
            )
            action = Action(action)
            if action not in available_actions:
                action = env.default_policy()
        agent.next_action = action

        temp_before = int(env.temp_c)
        belief_before = dict(env.belief)
        entropy_before = belief_entropy(belief_before)
        _next_state, reward, _done, step_info = env.step(action)
        if action == Action.INSPECT:
            inspect_events.append(
                InspectEvent(
                    task_id=str(demo.task_id),
                    step=int(idx),
                    material=str(demo.material),
                    current_temp_c=int(env.temp_c),
                    current_heat_c=int(env.temp_c) - int(demo.start_temp_c),
                    true_delta_needed_c=int(demo.true_delta_needed_c),
                    margin_c=(
                        int(env.temp_c)
                        - int(demo.assembly_cooling_c)
                        - int(demo.target_temp_c)
                        - int(demo.true_delta_needed_c)
                    ),
                    obs=env.last_obs,
                    belief_entropy_before=float(entropy_before),
                    belief_entropy_after=belief_entropy(env.belief),
                    inspect_model_type=str(inspect_model.model_type),
                )
            )
        violation = bool(action == Action.ASSEMBLE and int(env.rvr) == 1)
        p_violate = probability_violate_on_assemble(
            env.belief,
            temp_c=int(env.temp_c),
            target_temp_c=int(demo.target_temp_c),
            assembly_cooling_c=int(demo.assembly_cooling_c),
        )
        note = f"pomcp selected action; reward={float(reward):.1f}"
        if action == Action.INSPECT:
            note = "pomcp information action: update P(delta), no RVR"
        elif action == Action.ASSEMBLE:
            note = "pomcp terminal action: RVR is evaluated here"
        elif action == Action.SAFE_ABORT:
            note = "pomcp terminal safety exit: no success, no RVR"

        trace.append(
            TraceStep(
                step=int(idx),
                action=action,
                temp_before_c=int(temp_before),
                temp_after_c=int(env.temp_c),
                observation=env.last_obs,
                true_margin_now_c=(
                    int(env.temp_c)
                    - int(demo.target_temp_c)
                    - int(demo.true_delta_needed_c)
                ),
                true_margin_at_assembly_c=(
                    int(env.temp_c)
                    - int(demo.assembly_cooling_c)
                    - int(demo.target_temp_c)
                    - int(demo.true_delta_needed_c)
                ),
                p_now_in_spec=probability_now_in_spec(
                    env.belief,
                    temp_c=int(env.temp_c),
                    target_temp_c=int(demo.target_temp_c),
                ),
                p_violate=float(p_violate),
                belief_summary=summarize_belief(env.belief),
                top_belief=top_belief_states(env.belief),
                violation=bool(violation),
                note=str(note),
            )
        )

    if not bool(env.done):
        temp_before = int(env.temp_c)
        _next_state, _reward, _done, _step_info = env.step(Action.SAFE_ABORT)
        trace.append(
            TraceStep(
                step=len(trace) + 1,
                action=Action.SAFE_ABORT,
                temp_before_c=int(temp_before),
                temp_after_c=int(env.temp_c),
                observation=Observation.NONE,
                true_margin_now_c=(
                    int(env.temp_c)
                    - int(demo.target_temp_c)
                    - int(demo.true_delta_needed_c)
                ),
                true_margin_at_assembly_c=(
                    int(env.temp_c)
                    - int(demo.assembly_cooling_c)
                    - int(demo.target_temp_c)
                    - int(demo.true_delta_needed_c)
                ),
                p_now_in_spec=probability_now_in_spec(
                    env.belief,
                    temp_c=int(env.temp_c),
                    target_temp_c=int(demo.target_temp_c),
                ),
                p_violate=probability_violate_on_assemble(
                    env.belief,
                    temp_c=int(env.temp_c),
                    target_temp_c=int(demo.target_temp_c),
                    assembly_cooling_c=int(demo.assembly_cooling_c),
                ),
                belief_summary=summarize_belief(env.belief),
                top_belief=top_belief_states(env.belief),
                violation=False,
                note="pomcp max steps reached; safe abort",
            )
        )

    return EpisodeResult(
        success=bool(env.success),
        aborted=bool(env.aborted),
        rvr=int(env.rvr),
        final_temp_c=int(env.temp_c),
        trace=tuple(trace),
        inspect_events=tuple(inspect_events),
    )


def _belief_text(items: Iterable[Tuple[int, float]]) -> str:
    return "; ".join(f"resid={delta:+03d},p={prob:.2f}" for delta, prob in items)


def _heat5(n: int) -> Tuple[Action, ...]:
    return tuple(Action.HEAT_5 for _ in range(int(n)))


def _method_actions(method: str) -> Tuple[Action, ...]:
    if method == "human":
        # The human baseline is a formula-generated setpoint; no POMDP repair.
        return (Action.ASSEMBLE,)
    if method == "llm":
        # Directly trust the corrupted proposal.
        return (Action.ASSEMBLE,)
    if method == "llm+reflow_no_pomcp":
        # Reflow prompt already changed the LLM proposal; no POMCP/belief repair here.
        return (Action.ASSEMBLE,)
    raise ValueError(f"Unknown method: {method!r}")


def _method_description(method: str) -> str:
    descriptions = {
        "human": "thermal-expansion formula + fixed material margin; no LLM/POMDP",
        "llm": "execute no-reflow LLM proposal directly",
        "llm+reflow_no_pomcp": "execute reflow-prompt LLM proposal directly; no POMCP/belief",
        "llm+pomdp+no_reflow": "no-reflow LLM proposal + POMCP uniform prior + risk-penalized repair",
        "llm+pomdp+reflow": "reflow-prompt LLM proposal + uniform-prior POMCP repair",
    }
    return descriptions[str(method)]


def method_cost_components(episode: EpisodeResult, demo: MarginDemo) -> Dict[str, float]:
    n_inspect = sum(1 for step in episode.trace if step.action == Action.INSPECT)
    n_assemble = sum(1 for step in episode.trace if step.action == Action.ASSEMBLE)
    extra_heat_c = max(0, int(episode.final_temp_c) - int(demo.start_temp_c))
    true_threshold = (
        int(demo.target_temp_c)
        + int(demo.true_delta_needed_c)
        + int(demo.assembly_cooling_c)
    )
    overheat_c = max(0, int(episode.final_temp_c) - int(true_threshold))
    heat_action_cost = float(extra_heat_c)
    inspect_action_cost = float(POMCPRewardConfig().inspect_cost) * float(n_inspect)
    assemble_action_cost = 50.0 * float(n_assemble)
    overheat_action_cost = 0.5 * float(overheat_c)
    abort_action_cost = 500.0 if bool(episode.aborted) else 0.0
    total_cost = (
        float(heat_action_cost)
        + float(inspect_action_cost)
        + float(assemble_action_cost)
        + float(overheat_action_cost)
        + float(abort_action_cost)
    )
    return {
        "heat_action_cost": float(heat_action_cost),
        "inspect_action_cost": float(inspect_action_cost),
        "assemble_action_cost": float(assemble_action_cost),
        "overheat_action_cost": float(overheat_action_cost),
        "abort_action_cost": float(abort_action_cost),
        "total_cost": float(total_cost),
    }


def _method_result_from_episode(
    *,
    method: str,
    description: str,
    episode: EpisodeResult,
    demo: MarginDemo,
    policy_driven: bool,
    planner: str,
    belief_source: str = "none",
    initial_belief_entropy: float = 0.0,
    initial_expected_delta_c: float = 0.0,
    inspect_model_type: str = "none",
    prompt_history: bool = False,
) -> MethodResult:
    n_heat = sum(1 for step in episode.trace if step.action == Action.HEAT_5)
    n_inspect = sum(1 for step in episode.trace if step.action == Action.INSPECT)
    n_assemble = sum(1 for step in episode.trace if step.action == Action.ASSEMBLE)
    extra_heat_c = max(0, int(episode.final_temp_c) - int(demo.start_temp_c))
    costs = method_cost_components(episode, demo)
    return MethodResult(
        method=str(method),
        description=str(description),
        episode=episode,
        n_heat=int(n_heat),
        n_inspect=int(n_inspect),
        n_assemble=int(n_assemble),
        final_temp_c=int(episode.final_temp_c),
        extra_heat_c=int(extra_heat_c),
        total_cost=float(costs["total_cost"]),
        policy_driven=bool(policy_driven),
        planner=str(planner),
        belief_source=str(belief_source),
        initial_belief_entropy=float(initial_belief_entropy),
        initial_expected_delta_c=float(initial_expected_delta_c),
        inspect_model_type=str(inspect_model_type),
        prompt_history=bool(prompt_history),
    )


def run_fixed_reflow_method(
    demo: MarginDemo,
    *,
    heat_c: int,
    inspect: bool,
) -> MethodResult:
    if int(heat_c) < 0 or int(heat_c) % 5 != 0:
        raise ValueError(f"heat_c must be a non-negative multiple of 5, got {heat_c!r}")
    actions = _heat5(int(heat_c) // 5)
    if bool(inspect):
        actions = actions + (Action.INSPECT,)
    actions = actions + (Action.ASSEMBLE,)
    episode = run_episode(demo, actions=actions)
    inspect_label = "with_inspect" if bool(inspect) else "no_inspect"
    return _method_result_from_episode(
        method=f"fixed+{int(heat_c)}_{inspect_label}",
        description=f"fixed reflow +{int(heat_c)}C {inspect_label}",
        episode=episode,
        demo=demo,
        policy_driven=False,
        planner="scripted",
    )


def run_method(
    method: str,
    demo: MarginDemo,
    *,
    prompt_history: bool = False,
    pomcp_max_depth: int = 20,
    pomcp_max_it: int = 200,
    pomcp_particles: int = 200,
    random_seed: int | None = None,
    observation_seed: int | None = None,
) -> MethodResult:
    policy_driven = str(method) in {"llm+pomdp+no_reflow", "llm+pomdp+reflow"}
    if str(method) == "llm+pomdp+no_reflow":
        belief0 = initial_belief(demo)
        inspect_model = MaterialInspectModel(model_type="constant")
        episode = run_pomcp_episode(
            demo,
            belief0=belief0,
            inspect_model=inspect_model,
            pomcp_max_depth=int(pomcp_max_depth),
            pomcp_max_it=int(pomcp_max_it),
            pomcp_particles=int(pomcp_particles),
            seed=(
                int(random_seed)
                if random_seed is not None
                else 17 + int(demo.true_delta_needed_c) * 13
            ),
            observation_seed=(
                int(observation_seed) if observation_seed is not None else random_seed
            ),
        )
        planner = "pomcp"
        belief_source = "uniform"
        initial_summary = summarize_belief(belief0)
        initial_entropy = belief_entropy(belief0)
        inspect_model_type = inspect_model.model_type
    elif str(method) == "llm+pomdp+reflow":
        belief0 = initial_belief(demo)
        belief_source = "uniform"
        inspect_model = MaterialInspectModel(model_type="constant")
        episode = run_pomcp_episode(
            demo,
            belief0=belief0,
            inspect_model=inspect_model,
            pomcp_max_depth=int(pomcp_max_depth),
            pomcp_max_it=int(pomcp_max_it),
            pomcp_particles=int(pomcp_particles),
            seed=(
                int(random_seed)
                if random_seed is not None
                else 18 + int(demo.true_delta_needed_c) * 13
            ),
            observation_seed=(
                int(observation_seed) if observation_seed is not None else random_seed
            ),
        )
        planner = "pomcp"
        initial_summary = summarize_belief(belief0)
        initial_entropy = belief_entropy(belief0)
        inspect_model_type = inspect_model.model_type
    else:
        episode = run_episode(demo, actions=_method_actions(str(method)))
        planner = "scripted"
        belief_source = "none"
        initial_summary = BeliefSummary(expected_delta_c=0.0, delta_marginal={})
        initial_entropy = 0.0
        inspect_model_type = "none"
    return _method_result_from_episode(
        method=str(method),
        description=_method_description(str(method)),
        episode=episode,
        policy_driven=bool(policy_driven),
        planner=str(planner),
        demo=demo,
        belief_source=str(belief_source),
        initial_belief_entropy=float(initial_entropy),
        initial_expected_delta_c=float(initial_summary.expected_delta_c),
        inspect_model_type=str(inspect_model_type),
        prompt_history=bool(prompt_history),
    )


def run_five_methods(
    demo: MarginDemo,
) -> Tuple[MethodResult, ...]:
    methods = (
        "human",
        "llm",
        "llm+reflow_no_pomcp",
        "llm+pomdp+no_reflow",
        "llm+pomdp+reflow",
    )
    return tuple(run_method(method, demo) for method in methods)


def evaluate_methods_on_cases(
    delta_values: Sequence[int] = (0, 10, 20, 30, 40, 50, 60),
) -> Tuple[AggregateResult, ...]:
    by_method: Dict[str, List[MethodResult]] = {}
    for delta in delta_values:
        demo = build_demo_for_delta(int(delta))
        for result in run_five_methods(demo):
            by_method.setdefault(str(result.method), []).append(result)

    rows: List[AggregateResult] = []
    for method in (
        "human",
        "llm",
        "llm+reflow_no_pomcp",
        "llm+pomdp+no_reflow",
        "llm+pomdp+reflow",
    ):
        xs = list(by_method[str(method)])
        n = len(xs)
        rows.append(
            AggregateResult(
                method=str(method),
                n_cases=int(n),
                success_rate=sum(1.0 if r.episode.success else 0.0 for r in xs) / max(1, n),
                rvr_rate=sum(float(r.episode.rvr) for r in xs) / max(1, n),
                avg_heat_c=sum(float(r.extra_heat_c) for r in xs) / max(1, n),
                avg_inspect=sum(float(r.n_inspect) for r in xs) / max(1, n),
                avg_cost=sum(float(r.total_cost) for r in xs) / max(1, n),
            )
        )
    return tuple(rows)


def run_margin_tasks_csv(
    tasks_csv: str | Path,
    *,
    methods: Sequence[str] = (
        "human",
        "llm",
        "llm+reflow_no_pomcp",
        "llm+pomdp+no_reflow",
        "llm+pomdp+reflow",
    ),
    proposal_temp_c: int = 300,
    proposal_temps_c: Mapping[str, int] | None = None,
    reflow_proposal_temps_c: Mapping[str, int] | None = None,
    assembly_cooling_c: int = 10,
    room_temp_c: int = 16,
    risk_threshold: float = 0.3,
    belief_scope: str = "broad",
    belief_window_c: int = 15,
    use_world_seed: bool = False,
) -> Tuple[MarginTaskRun, ...]:
    out: List[MarginTaskRun] = []
    method_set = set(str(method) for method in methods)
    if method_set.intersection(set(REFLOW_PROMPT_METHODS)) and reflow_proposal_temps_c is None:
        raise ValueError(
            "methods with reflow prompt requires reflow proposal temps; "
            "pass reflow_proposal_temps_c."
        )
    proposal_map = {str(k): int(v) for k, v in dict(proposal_temps_c or {}).items()}
    reflow_proposal_map = {
        str(k): int(v) for k, v in dict(reflow_proposal_temps_c or {}).items()
    }
    with Path(tasks_csv).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_id = str(row.get("task_id", ""))
            for method in methods:
                uses_reflow_prompt = str(method) in set(REFLOW_PROMPT_METHODS)
                if str(method) == "human":
                    task_proposal_temp_c = human_formula_proposal_temp_c(
                        row,
                        room_temp_c=int(room_temp_c),
                    )
                else:
                    selected_map = reflow_proposal_map if uses_reflow_prompt else proposal_map
                    task_proposal_temp_c = int(selected_map.get(task_id, int(proposal_temp_c)))
                demo = build_demo_from_task_row(
                    row,
                    proposal_temp_c=int(task_proposal_temp_c),
                    assembly_cooling_c=int(assembly_cooling_c),
                    risk_threshold=float(risk_threshold),
                    belief_scope=str(belief_scope),
                    belief_window_c=int(belief_window_c),
                )
                result = run_method(
                    str(method),
                    demo,
                    prompt_history=bool(uses_reflow_prompt),
                    random_seed=(
                        int(float(row["world_seed"]))
                        if bool(use_world_seed) and str(row.get("world_seed", "")).strip()
                        else None
                    ),
                )
                out.append(
                    MarginTaskRun(
                        task_id=str(task_id),
                        material=str(row.get("material", "")),
                        t_needed_c=float(row["t_needed_c"]),
                        proposal_temp_c=int(task_proposal_temp_c),
                        demo=demo,
                        method_result=result,
                    )
                )
    return tuple(out)


def apply_proposal_shift(
    proposal_temps_c: Mapping[str, int],
    *,
    shift_c: int,
    min_temp_c: int = 250,
    max_temp_c: int = 450,
) -> Dict[str, int]:
    shifted: Dict[str, int] = {}
    for task_id, temp in proposal_temps_c.items():
        t = int(temp) + int(shift_c)
        t = max(int(min_temp_c), min(int(max_temp_c), int(t)))
        shifted[str(task_id)] = int(t)
    return shifted


def evaluate_fixed_reflow_sweep(
    tasks_csv: str | Path,
    *,
    proposal_temps_c: Mapping[str, int] | None = None,
    proposal_temp_c: int = 300,
    assembly_cooling_c: int = 10,
    heat_values_c: Sequence[int] = tuple(range(0, 55, 5)),
    inspect_options: Sequence[bool] = (False, True),
) -> Tuple[FixedReflowSweepResult, ...]:
    task_rows: List[Mapping[str, Any]] = []
    with Path(tasks_csv).open("r", encoding="utf-8") as f:
        task_rows.extend(list(csv.DictReader(f)))

    proposal_map = {str(k): int(v) for k, v in dict(proposal_temps_c or {}).items()}
    out: List[FixedReflowSweepResult] = []
    for inspect in inspect_options:
        for heat_c in heat_values_c:
            results: List[MethodResult] = []
            for row in task_rows:
                task_id = str(row.get("task_id", ""))
                task_proposal = int(proposal_map.get(task_id, int(proposal_temp_c)))
                demo = build_demo_from_task_row(
                    row,
                    proposal_temp_c=int(task_proposal),
                    assembly_cooling_c=int(assembly_cooling_c),
                )
                results.append(run_fixed_reflow_method(demo, heat_c=int(heat_c), inspect=bool(inspect)))
            n = len(results)
            out.append(
                FixedReflowSweepResult(
                    heat_c=int(heat_c),
                    inspect=bool(inspect),
                    n_tasks=int(n),
                    success_rate=sum(1.0 if r.episode.success else 0.0 for r in results) / max(1, n),
                    rvr_rate=sum(float(r.episode.rvr) for r in results) / max(1, n),
                    avg_final_temp_c=sum(float(r.final_temp_c) for r in results) / max(1, n),
                    avg_heat_c=sum(float(r.extra_heat_c) for r in results) / max(1, n),
                    avg_inspect=sum(float(r.n_inspect) for r in results) / max(1, n),
                    avg_cost=sum(float(r.total_cost) for r in results) / max(1, n),
                )
            )
    return tuple(out)


def aggregate_task_results(rows: Sequence[MarginTaskRun]) -> Tuple[MarginTaskAggregate, ...]:
    by_method: Dict[str, List[MarginTaskRun]] = {}
    order: List[str] = []
    for row in rows:
        method = str(row.method_result.method)
        if method not in by_method:
            order.append(method)
        by_method.setdefault(method, []).append(row)

    aggregates: List[MarginTaskAggregate] = []
    for method in order:
        xs = by_method[str(method)]
        n = len(xs)
        aggregates.append(
            MarginTaskAggregate(
                method=str(method),
                n_tasks=int(n),
                success_rate=sum(1.0 if x.method_result.episode.success else 0.0 for x in xs) / max(1, n),
                rvr_rate=sum(float(x.method_result.episode.rvr) for x in xs) / max(1, n),
                avg_final_temp_c=sum(float(x.method_result.final_temp_c) for x in xs) / max(1, n),
                avg_heat_c=sum(float(x.method_result.extra_heat_c) for x in xs) / max(1, n),
                avg_inspect=sum(float(x.method_result.n_inspect) for x in xs) / max(1, n),
                avg_cost=sum(float(x.method_result.total_cost) for x in xs) / max(1, n),
            )
        )
    return tuple(aggregates)


def format_methods_summary(results: Sequence[MethodResult], demo: MarginDemo) -> str:
    lines = [
        "=== Five-method margin-bin comparison ===",
        (
            f"proposal target={demo.target_temp_c}C, true residual={demo.true_delta_needed_c}C, "
            f"assembly cooling={demo.assembly_cooling_c}C, true threshold="
            f"{demo.target_temp_c + demo.true_delta_needed_c + demo.assembly_cooling_c}C"
        ),
        "",
        "method                         | success | RVR | final_T | heat | inspect | cost | description",
        "-------------------------------+---------+-----+---------+------+---------+------+------------------------------------------",
    ]
    for result in results:
        lines.append(
            f"{result.method:<30} | "
            f"{int(result.episode.success):>7} | "
            f"{int(result.episode.rvr):>3} | "
            f"{int(result.final_temp_c):>7} | "
            f"{int(result.extra_heat_c):>4} | "
            f"{int(result.n_inspect):>7} | "
            f"{float(result.total_cost):>4.0f} | "
            f"{result.description}"
        )
    return "\n".join(lines)


def format_task_aggregate(rows: Sequence[MarginTaskAggregate]) -> str:
    lines = [
        "=== Margin-bin CSV aggregate ===",
        "method                         | n  | success_rate | RVR_rate | avg_final_T | avg_heat | avg_inspect | avg_cost",
        "-------------------------------+----+--------------+----------+-------------+----------+-------------+---------",
    ]
    for row in rows:
        lines.append(
            f"{row.method:<30} | "
            f"{int(row.n_tasks):>2} | "
            f"{float(row.success_rate):>12.2f} | "
            f"{float(row.rvr_rate):>8.2f} | "
            f"{float(row.avg_final_temp_c):>11.1f} | "
            f"{float(row.avg_heat_c):>8.1f} | "
            f"{float(row.avg_inspect):>11.1f} | "
            f"{float(row.avg_cost):>7.1f}"
        )
    return "\n".join(lines)


def format_fixed_reflow_sweep(rows: Sequence[FixedReflowSweepResult]) -> str:
    lines = [
        "=== Fixed reflow sweep ===",
        "baseline                 | n  | success_rate | RVR_rate | avg_final_T | avg_heat | avg_inspect | avg_cost",
        "-------------------------+----+--------------+----------+-------------+----------+-------------+---------",
    ]
    for row in rows:
        inspect_label = "with_inspect" if bool(row.inspect) else "no_inspect"
        label = f"fixed +{int(row.heat_c)} {inspect_label}"
        lines.append(
            f"{label:<24} | "
            f"{int(row.n_tasks):>2} | "
            f"{float(row.success_rate):>12.2f} | "
            f"{float(row.rvr_rate):>8.2f} | "
            f"{float(row.avg_final_temp_c):>11.1f} | "
            f"{float(row.avg_heat_c):>8.1f} | "
            f"{float(row.avg_inspect):>11.1f} | "
            f"{float(row.avg_cost):>7.1f}"
        )
    return "\n".join(lines)


def format_aggregate_summary(rows: Sequence[AggregateResult]) -> str:
    lines = [
        "=== Multi-delta aggregate ===",
        "method                         | n | success_rate | RVR_rate | avg_heat | avg_inspect | avg_cost",
        "-------------------------------+---+--------------+----------+----------+-------------+---------",
    ]
    for row in rows:
        lines.append(
            f"{row.method:<30} | "
            f"{int(row.n_cases):>1} | "
            f"{float(row.success_rate):>12.2f} | "
            f"{float(row.rvr_rate):>8.2f} | "
            f"{float(row.avg_heat_c):>8.1f} | "
            f"{float(row.avg_inspect):>11.1f} | "
            f"{float(row.avg_cost):>7.1f}"
        )
    return "\n".join(lines)


def _marginal_text(marginal: Mapping[int, float], *, min_prob: float = 0.04) -> str:
    parts = [
        f"{int(value)}:{float(prob):.2f}"
        for value, prob in sorted(marginal.items())
        if float(prob) >= float(min_prob)
    ]
    return " ".join(parts)


def format_trace(result: EpisodeResult, demo: MarginDemo) -> str:
    lines = [
        "=== Margin-bin PVEP demo ===",
        (
            f"proposal target={demo.target_temp_c}C, true residual={demo.true_delta_needed_c}C, "
            f"known assembly cooling={demo.assembly_cooling_c}C, "
            f"true assembly threshold={demo.target_temp_c + demo.true_delta_needed_c + demo.assembly_cooling_c}C, "
            f"risk_threshold={demo.risk_threshold:.2f}"
        ),
        "",
        "step | action     | temp      | obs        | margin_now | margin_at_asm | E_resid | P(now ok) | P_violate | violation",
        "-----+------------+-----------+------------+------------+---------------+---------+-----------+-----------+----------",
    ]
    for row in result.trace:
        gate = " <= gate" if float(row.p_violate) <= float(demo.risk_threshold) else ""
        lines.append(
            f"{row.step:>4} | "
            f"{row.action.value:<10} | "
            f"{row.temp_before_c:>3}->{row.temp_after_c:<3}C | "
            f"{row.observation.value:<10} | "
            f"{row.true_margin_now_c:>+10} | "
            f"{row.true_margin_at_assembly_c:>+13} | "
            f"{row.belief_summary.expected_delta_c:>7.1f} | "
            f"{row.p_now_in_spec:>9.2f} | "
            f"{row.p_violate:>9.2f}{gate:<8} | "
            f"{int(row.violation):>9}"
        )
        if row.action == Action.INSPECT:
            lines.append(
                f"     P(resid): {_marginal_text(row.belief_summary.delta_marginal)}"
            )
            lines.append(f"     top belief: {_belief_text(row.top_belief)}")
    lines.extend(
        [
            "",
            f"success={int(result.success)} aborted={int(result.aborted)} final_temp={result.final_temp_c}C RVR={result.rvr}",
            "Interpretation: INSPECT is a verifier/belief-update action. RVR is assigned only at ASSEMBLE.",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    demo = build_default_demo()
    results = run_five_methods(demo)
    print(format_methods_summary(results, demo))
    print()
    ours = next(r for r in results if r.method == "llm+pomdp+reflow")
    print(format_trace(ours.episode, demo))
    print()
    print(format_aggregate_summary(evaluate_methods_on_cases()))


if __name__ == "__main__":
    main()
