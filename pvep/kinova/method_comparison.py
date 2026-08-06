"""Compare decision methods for seven-hole tri-state screw installation.

Task
----
Each episode starts with seven holes. Every hole is one of:

  EMPTY     no screw is present; MISSING is treated as EMPTY
  MISALIGN  screw is present but not aligned
  ALIGN     screw is present and aligned, not yet fastened

The goal is to finish all seven holes. ALIGN still requires FASTEN.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fasten_pomdp import (
    STATE_ALIGN,
    STATE_EMPTY,
    STATE_MISALIGN,
    STATE_ORDER,
    Belief,
    ConfBin,
    FastenAction,
    FastenAgent,
    FastenEnv,
    VisualLabel,
    bayesian_update_belief,
    get_observation_model_summary,
    normalize_belief,
    sample_disc_obs,
    set_observation_truth_probs,
    set_yolo_observation_accuracy,
    set_yolo_output_corruption,
)
from evaluate_stage_c_screw_status import truth_by_hole
from prepare_stage_c_yolo_crops import load_template_groups


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

VISUAL_COST = 3.0
PROBE_COST_FIXED = 2.0
PROBE_COST_OPTIMAL = 3.0
INSERT_COST = 6.0
RECOVER_COST = 10.0
FAILURE_PENALTY = 50.0
SUCCESS_REWARD = 10.0
N_HOLES = 7
PVEP_INITIAL_OUTPUT_CORRUPTION = 0.0

TARGET_HOLES = tuple(f"hole_{letter}" for letter in "ABCDEF")
N_HOLES = len(TARGET_HOLES)
DEFAULT_TEMPLATE = _HERE / "螺丝装配" / "yolo训练" / "scew" / "template.json"
DEFAULT_SPLIT_CSV = _HERE / "螺丝装配" / "proposal_splits_40test" / "splits" / "test40.csv"

# Empirical test40 initial-state prior over target holes A-F:
# EMPTY=92, MISALIGN=34, ALIGN=114.
INITIAL_STATE_PRIOR: Belief = normalize_belief(
    {
        STATE_EMPTY: 92 / 240,
        STATE_MISALIGN: 34 / 240,
        STATE_ALIGN: 114 / 240,
    }
)

_EPISODE_STATE_CACHE: Optional[List[Tuple[str, ...]]] = None
_TRUTH_TO_STATE = {
    "EMPTY": STATE_EMPTY,
    "MISALIGN": STATE_MISALIGN,
    "ALIGN": STATE_ALIGN,
}


def set_cost_model(
    *,
    probe_cost_fixed: Optional[float] = None,
    probe_cost_optimal: Optional[float] = None,
    insert_cost: Optional[float] = None,
    recover_cost: Optional[float] = None,
    failure_penalty: Optional[float] = None,
    success_reward: Optional[float] = None,
) -> None:
    """Configure the positive-cost model used by all methods."""
    global PROBE_COST_FIXED, PROBE_COST_OPTIMAL, INSERT_COST, RECOVER_COST
    global FAILURE_PENALTY, SUCCESS_REWARD

    updates = {
        "probe_cost_fixed": probe_cost_fixed,
        "probe_cost_optimal": probe_cost_optimal,
        "insert_cost": insert_cost,
        "recover_cost": recover_cost,
        "failure_penalty": failure_penalty,
        "success_reward": success_reward,
    }
    for name, value in updates.items():
        if value is not None and value < 0.0:
            raise ValueError(f"{name} must be non-negative, got {value}")

    if probe_cost_fixed is not None:
        PROBE_COST_FIXED = float(probe_cost_fixed)
    if probe_cost_optimal is not None:
        PROBE_COST_OPTIMAL = float(probe_cost_optimal)
    if insert_cost is not None:
        INSERT_COST = float(insert_cost)
    if recover_cost is not None:
        RECOVER_COST = float(recover_cost)
    if failure_penalty is not None:
        FAILURE_PENALTY = float(failure_penalty)
    if success_reward is not None:
        SUCCESS_REWARD = float(success_reward)


def get_cost_model_summary() -> Dict[str, float]:
    return {
        "visual_cost": VISUAL_COST,
        "probe_cost_fixed": PROBE_COST_FIXED,
        "probe_cost_optimal": PROBE_COST_OPTIMAL,
        "insert_cost": INSERT_COST,
        "recover_cost": RECOVER_COST,
        "failure_penalty": FAILURE_PENALTY,
        "success_reward": SUCCESS_REWARD,
    }


def set_pvep_initial_output_corruption(epsilon: Optional[float] = None) -> None:
    """Set label corruption applied only to PVEP's initial MLM output sample."""
    global PVEP_INITIAL_OUTPUT_CORRUPTION
    if epsilon is None:
        PVEP_INITIAL_OUTPUT_CORRUPTION = 0.0
        return
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError(f"PVEP initial corruption epsilon must be in [0, 1], got {epsilon}")
    PVEP_INITIAL_OUTPUT_CORRUPTION = float(epsilon)


def get_pvep_initial_output_corruption() -> float:
    return PVEP_INITIAL_OUTPUT_CORRUPTION


class _HoleResult:
    __slots__ = (
        "inspect_cost",
        "insert_cost",
        "recover_cost",
        "failure_cost",
        "success",
        "n_inspections",
        "n_inserts",
        "n_recovers",
        "action_trace",
        "success_reward",
    )

    def __init__(
        self,
        inspect_cost: float,
        insert_cost: float,
        recover_cost: float,
        failure_cost: float,
        success: bool,
        n_inspections: int,
        n_inserts: int,
        n_recovers: int,
        action_trace: Optional[List[str]] = None,
        success_reward: float = 0.0,
    ) -> None:
        self.inspect_cost = inspect_cost
        self.insert_cost = insert_cost
        self.recover_cost = recover_cost
        self.failure_cost = failure_cost
        self.success = success
        self.n_inspections = n_inspections
        self.n_inserts = n_inserts
        self.n_recovers = n_recovers
        self.action_trace = list(action_trace or [])
        self.success_reward = success_reward

    @property
    def total_cost(self) -> float:
        return (
            self.inspect_cost
            + self.insert_cost
            + self.recover_cost
            + self.failure_cost
            - self.success_reward
        )


def _sample_initial_state(prior: Belief = INITIAL_STATE_PRIOR) -> str:
    prior = normalize_belief(prior)
    states = list(STATE_ORDER)
    weights = [prior[state] for state in states]
    return random.choices(states, weights=weights, k=1)[0]


def _truth_status_to_state(status: str) -> str:
    return _TRUTH_TO_STATE.get(status, STATE_EMPTY)


def _other_visual_labels(label: VisualLabel) -> List[VisualLabel]:
    return [candidate for candidate in VisualLabel if candidate != label]


def _corrupt_disc_obs_label(obs: Tuple[VisualLabel, ConfBin], epsilon: float) -> Tuple[VisualLabel, ConfBin]:
    label, conf_bin = obs
    if epsilon > 0.0 and random.random() < epsilon:
        label = random.choice(_other_visual_labels(label))
        conf_bin = ConfBin.LOW
    return label, conf_bin


def _sample_initial_obs(
    true_state: str,
    *,
    apply_pvep_initial_corruption: bool,
) -> Tuple[VisualLabel, ConfBin]:
    obs = sample_disc_obs(true_state, "initial")
    if apply_pvep_initial_corruption:
        return _corrupt_disc_obs_label(obs, PVEP_INITIAL_OUTPUT_CORRUPTION)
    return obs


def _sample_episode_initial_obs(states: Tuple[str, ...]) -> List[Tuple[VisualLabel, ConfBin]]:
    """Sample one image-level initial output and independently corrupt each hole."""
    obs_by_hole = [sample_disc_obs(state, "initial") for state in states]
    if PVEP_INITIAL_OUTPUT_CORRUPTION > 0.0:
        obs_by_hole = [
            _corrupt_disc_obs_label(obs, 1.0)
            if random.random() < PVEP_INITIAL_OUTPUT_CORRUPTION
            else obs
            for obs in obs_by_hole
        ]
    return obs_by_hole


def _load_target_hole_episodes() -> List[Tuple[str, ...]]:
    """Load real test40 A-F hole states; one tuple corresponds to one image."""
    global _EPISODE_STATE_CACHE
    if _EPISODE_STATE_CACHE is not None:
        return _EPISODE_STATE_CACHE

    groups = load_template_groups(DEFAULT_TEMPLATE)
    episodes: List[Tuple[str, ...]] = []
    with DEFAULT_SPLIT_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            truth = truth_by_hole(Path(row["source_json"]), groups)
            episodes.append(
                tuple(_truth_status_to_state(truth.get(hole, "EMPTY")) for hole in TARGET_HOLES)
            )

    if not episodes:
        raise RuntimeError(f"no target-hole episodes loaded from {DEFAULT_SPLIT_CSV}")
    _EPISODE_STATE_CACHE = episodes
    return episodes


def _episode_states_for_index(index: int) -> Tuple[str, ...]:
    """Return one real image-level A-F initial state vector by deterministic cycling."""
    episodes = _load_target_hole_episodes()
    return episodes[index % len(episodes)]


def _state_after_insert(state: str) -> str:
    return STATE_ALIGN if state == STATE_EMPTY else state


def _state_after_recover(state: str) -> str:
    return STATE_ALIGN if state == STATE_MISALIGN else state


def _execute_label_policy(
    true_state: str,
    label: VisualLabel,
    *,
    inspect_cost: float,
    n_insp: int,
) -> _HoleResult:
    """Execute the rule implied by a tri-class visual label, then fasten."""
    insert_cost = 0.0
    recover_cost = 0.0
    failure_cost = 0.0
    success_reward = 0.0
    n_inserts = 0
    n_recovers = 0
    action_trace: List[str] = []
    state = true_state

    if label == VisualLabel.EMPTY:
        n_inserts += 1
        action_trace.append("INSERT")
        if state != STATE_EMPTY:
            failure_cost += FAILURE_PENALTY
            return _HoleResult(
                inspect_cost,
                insert_cost,
                recover_cost,
                failure_cost,
                False,
                n_insp,
                n_inserts,
                n_recovers,
                action_trace,
                success_reward,
            )
        insert_cost += INSERT_COST
        state = _state_after_insert(state)
    elif label == VisualLabel.MISALIGN:
        recover_cost += RECOVER_COST
        n_recovers += 1
        action_trace.append("RECOVER")
        state = _state_after_recover(state)
    elif label == VisualLabel.ALIGN:
        pass
    else:
        raise ValueError(f"unknown label: {label}")

    success = state == STATE_ALIGN
    action_trace.append("FASTEN")
    if success:
        success_reward += SUCCESS_REWARD
    else:
        failure_cost += FAILURE_PENALTY

    return _HoleResult(
        inspect_cost,
        insert_cost,
        recover_cost,
        failure_cost,
        success,
        n_insp,
        n_inserts,
        n_recovers,
        action_trace,
        success_reward,
    )


def _run_one_hole_MLM(true_state: str) -> _HoleResult:
    obs = sample_disc_obs(true_state, "initial")
    label, _conf_bin = obs
    return _execute_label_policy(
        true_state,
        label,
        inspect_cost=VISUAL_COST,
        n_insp=1,
    )


def run_MLM(
    n_episodes: int = 300,
    n_holes: int = N_HOLES,
    p_good: float = 0.0,
    seed: Optional[int] = None,
) -> Dict:
    del p_good
    if seed is not None:
        random.seed(seed)
    results = []
    for episode_idx in range(n_episodes):
        states = _episode_states_for_index(episode_idx)
        holes = [_run_one_hole_MLM(state) for state in states]
        results.append((all(h.success for h in holes), holes))
    return _summary("MLM", results)


def _run_one_hole_human(true_state: str) -> _HoleResult:
    n_insp = 1
    inspect_cost = VISUAL_COST
    label, conf_bin = sample_disc_obs(true_state, "initial")
    if conf_bin == ConfBin.LOW:
        label, _conf_bin = sample_disc_obs(true_state, "fixed")
        inspect_cost += PROBE_COST_FIXED
        n_insp += 1
    return _execute_label_policy(
        true_state,
        label,
        inspect_cost=inspect_cost,
        n_insp=n_insp,
    )


def _run_one_hole_no_pomdp(true_state: str) -> _HoleResult:
    n_insp = 1
    inspect_cost = VISUAL_COST
    label, conf_bin = sample_disc_obs(true_state, "initial")
    if conf_bin == ConfBin.LOW:
        label, _conf_bin = sample_disc_obs(true_state, "optimal")
        inspect_cost += PROBE_COST_OPTIMAL
        n_insp += 1
    elif conf_bin == ConfBin.MED:
        label, _conf_bin = sample_disc_obs(true_state, "fixed")
        inspect_cost += PROBE_COST_FIXED
        n_insp += 1
    return _execute_label_policy(
        true_state,
        label,
        inspect_cost=inspect_cost,
        n_insp=n_insp,
    )


def run_Human(
    n_episodes: int = 300,
    n_holes: int = N_HOLES,
    p_good: float = 0.0,
    seed: Optional[int] = None,
) -> Dict:
    del p_good
    if seed is not None:
        random.seed(seed)
    results = []
    for episode_idx in range(n_episodes):
        states = _episode_states_for_index(episode_idx)
        holes = [_run_one_hole_human(state) for state in states]
        results.append((all(h.success for h in holes), holes))
    return _summary("Human", results)


def run_pvep_no_pomdp(
    n_episodes: int = 300,
    n_holes: int = N_HOLES,
    p_good: float = 0.0,
    seed: Optional[int] = None,
) -> Dict:
    del p_good
    if seed is not None:
        random.seed(seed)
    results = []
    for episode_idx in range(n_episodes):
        states = _episode_states_for_index(episode_idx)
        holes = [_run_one_hole_no_pomdp(state) for state in states]
        results.append((all(h.success for h in holes), holes))
    return _summary("PVEP_no_POMDP", results)


def _run_one_hole_pomcp(
    true_state: str,
    probe_type: str,
    *,
    pomcp_max_it: int = 500,
    pomcp_max_depth: int = 10,
    pomcp_particles: int = 200,
    pomcp_discount: float = 0.95,
    apply_pvep_initial_corruption: bool = False,
    initial_obs: Optional[Tuple[VisualLabel, ConfBin]] = None,
    allow_recover: bool = True,
) -> _HoleResult:
    from ibpomcp.pomcp import pomcp_planning

    if initial_obs is None:
        obs = _sample_initial_obs(
            true_state,
            apply_pvep_initial_corruption=apply_pvep_initial_corruption,
        )
    else:
        obs = initial_obs
    belief = bayesian_update_belief(INITIAL_STATE_PRIOR, obs, "initial")
    n_insp = 1
    n_inserts = 0
    n_recovers = 0
    inspect_cost = VISUAL_COST
    insert_cost = 0.0
    recover_cost = 0.0
    failure_cost = 0.0
    success_reward = 0.0
    action_trace: List[str] = []

    env = FastenEnv.create(
        belief=belief,
        probe_type=probe_type,
        visual_cost=VISUAL_COST,
        probe_cost=PROBE_COST_FIXED if probe_type == "fixed" else PROBE_COST_OPTIMAL,
        probe_cost_fixed=PROBE_COST_FIXED,
        probe_cost_optimal=PROBE_COST_OPTIMAL,
        insert_cost=INSERT_COST,
        recover_cost=RECOVER_COST,
        failure_penalty=FAILURE_PENALTY,
        success_reward=SUCCESS_REWARD,
        allow_recover=allow_recover,
    )
    env.hole_state = true_state

    agent = FastenAgent()
    for _ in range(env.max_steps):
        if env.state_set.is_final_state(env):
            break

        action, _ = pomcp_planning(
            env,
            agent,
            max_depth=pomcp_max_depth,
            max_it=pomcp_max_it,
            discount_factor=pomcp_discount,
            k=pomcp_particles,
        )
        agent.next_action = int(action)
        fa = FastenAction(int(action))
        action_trace.append(fa.name)

        if fa == FastenAction.PROBE_FIXED:
            n_insp += 1
            inspect_cost += PROBE_COST_FIXED
        elif fa == FastenAction.PROBE_OPTIMAL:
            n_insp += 1
            inspect_cost += PROBE_COST_OPTIMAL
        elif fa == FastenAction.INSERT:
            n_inserts += 1
            if env.hole_state == STATE_EMPTY:
                insert_cost += INSERT_COST
        elif fa == FastenAction.RECOVER:
            n_recovers += 1
            recover_cost += RECOVER_COST

        if fa == FastenAction.FASTEN and env.hole_state != STATE_ALIGN:
            failure_cost += FAILURE_PENALTY
        elif fa == FastenAction.INSERT and env.hole_state != STATE_EMPTY:
            failure_cost += FAILURE_PENALTY
        elif fa == FastenAction.FASTEN and env.hole_state == STATE_ALIGN:
            success_reward += SUCCESS_REWARD
        _, _reward, done, _info = env.step(int(action))
        if done:
            break

    success = env.phase == "done"
    if not success and failure_cost <= 0.0:
        failure_cost += FAILURE_PENALTY

    return _HoleResult(
        inspect_cost,
        insert_cost,
        recover_cost,
        failure_cost,
        success,
        n_insp,
        n_inserts,
        n_recovers,
        action_trace,
        success_reward,
    )


def run_pvep_no_prompt(
    n_episodes: int = 300,
    n_holes: int = N_HOLES,
    p_good: float = 0.0,
    seed: Optional[int] = None,
    pomcp_max_it: int = 500,
    pomcp_max_depth: int = 10,
) -> Dict:
    del p_good
    if seed is not None:
        random.seed(seed)
    results = []
    for episode_idx in range(n_episodes):
        states = _episode_states_for_index(episode_idx)
        holes = [
            _run_one_hole_pomcp(
                state,
                "fixed",
                pomcp_max_it=pomcp_max_it,
                pomcp_max_depth=pomcp_max_depth,
            )
            for state in states
        ]
        results.append((all(h.success for h in holes), holes))
    return _summary("PVEP_no_prompt", results)


def run_pvep(
    n_episodes: int = 300,
    n_holes: int = N_HOLES,
    p_good: float = 0.0,
    seed: Optional[int] = None,
    pomcp_max_it: int = 500,
    pomcp_max_depth: int = 10,
) -> Dict:
    del p_good
    if seed is not None:
        random.seed(seed)
    results = []
    for episode_idx in range(n_episodes):
        states = _episode_states_for_index(episode_idx)
        initial_obs_by_hole = _sample_episode_initial_obs(states)
        holes = [
            _run_one_hole_pomcp(
                state,
                "optimal",
                pomcp_max_it=pomcp_max_it,
                pomcp_max_depth=pomcp_max_depth,
                initial_obs=initial_obs,
            )
            for state, initial_obs in zip(states, initial_obs_by_hole)
        ]
        results.append((all(h.success for h in holes), holes))
    return _summary("PVEP", results)


def _summary(name: str, results: List[Tuple[bool, List[_HoleResult]]]) -> Dict:
    n = len(results)
    pass_list = [ep_ok for ep_ok, _holes in results]
    total_costs = [sum(h.total_cost for h in holes) for _ok, holes in results]
    inspect_costs = [sum(h.inspect_cost for h in holes) for _ok, holes in results]
    insert_costs = [sum(h.insert_cost for h in holes) for _ok, holes in results]
    recover_costs = [sum(h.recover_cost for h in holes) for _ok, holes in results]
    failure_costs = [sum(h.failure_cost for h in holes) for _ok, holes in results]
    success_rewards = [sum(h.success_reward for h in holes) for _ok, holes in results]
    fail_counts = [sum(1 for h in holes if not h.success) for _ok, holes in results]
    insp_counts = [sum(h.n_inspections for h in holes) for _ok, holes in results]
    insert_counts = [sum(h.n_inserts for h in holes) for _ok, holes in results]
    recover_counts = [sum(h.n_recovers for h in holes) for _ok, holes in results]

    return {
        "method": name,
        "n_episodes": n,
        "n_holes": N_HOLES,
        "target_holes": list(TARGET_HOLES),
        "pass_rate": sum(pass_list) / n,
        "avg_total_cost": sum(total_costs) / n,
        "avg_inspect_cost": sum(inspect_costs) / n,
        "avg_insert_cost": sum(insert_costs) / n,
        "avg_recover_cost": sum(recover_costs) / n,
        "avg_failure_cost": sum(failure_costs) / n,
        "avg_success_reward": sum(success_rewards) / n,
        "avg_fail_holes": sum(fail_counts) / n,
        "avg_inspections": sum(insp_counts) / n,
        "avg_inserts": sum(insert_counts) / n,
        "avg_recovers": sum(recover_counts) / n,
    }


def run_all(
    n_episodes: int = 300,
    p_good: float = 0.0,
    seed: int = 42,
    pomcp_max_it: int = 500,
    pomcp_max_depth: int = 10,
    yolo_acc: Optional[float] = None,
    yolo_corruption: Optional[float] = None,
    pvep_initial_corruption: Optional[float] = None,
    obs_optimal: Optional[float] = None,
    probe_cost_fixed: Optional[float] = None,
    probe_cost_optimal: Optional[float] = None,
    insert_cost: Optional[float] = None,
    recover_cost: Optional[float] = None,
    failure_penalty: Optional[float] = None,
    success_reward: Optional[float] = None,
    verbose: bool = True,
) -> List[Dict]:
    del p_good
    set_cost_model(
        probe_cost_fixed=probe_cost_fixed,
        probe_cost_optimal=probe_cost_optimal,
        insert_cost=insert_cost,
        recover_cost=recover_cost,
        failure_penalty=failure_penalty,
        success_reward=success_reward,
    )
    set_yolo_observation_accuracy(yolo_acc)
    set_yolo_output_corruption(yolo_corruption)
    set_pvep_initial_output_corruption(pvep_initial_corruption)
    set_observation_truth_probs(optimal=obs_optimal)
    obs_model = get_observation_model_summary()
    cost_model = get_cost_model_summary()
    results: List[Dict] = []

    if verbose:
        optimal = obs_model["optimal"]
        if optimal.get("source") == "truth_probability_override":
            optimal_text = f"optimal P(truth)={optimal['p_truth']:.3f}"
        else:
            optimal_text = f"optimal=clean MLM empirical acc {optimal['effective_accuracy']:.3f}"
        print(f"\nRunning comparison: {n_episodes} episodes x {N_HOLES} target holes")
        print(f"  Target holes = {', '.join(TARGET_HOLES)}")
        print(f"  Initial prior = {INITIAL_STATE_PRIOR}, seed = {seed}\n")
        print(f"  POMCP: max_it={pomcp_max_it}, max_depth={pomcp_max_depth}\n")
        print(
            "  Observation model: "
            f"initial=noisy MLM acc {obs_model['initial']['effective_accuracy']:.3f}, "
            f"fixed=noisy MLM acc {obs_model['fixed']['effective_accuracy']:.3f}, "
            f"corruption epsilon={obs_model['initial']['label_corruption_epsilon']:.3f}, "
            f"{optimal_text}\n"
        )
        print(
            "  PVEP-only initial output corruption: "
            f"epsilon={get_pvep_initial_output_corruption():.3f}\n"
        )
        print(
            "  Cost model: "
            f"visual={cost_model['visual_cost']:.1f}, "
            f"fixed_probe={cost_model['probe_cost_fixed']:.1f}, "
            f"optimal_probe={cost_model['probe_cost_optimal']:.1f}, "
            f"insert={cost_model['insert_cost']:.1f}, "
            f"recover={cost_model['recover_cost']:.1f}, "
            f"failure={cost_model['failure_penalty']:.1f}, "
            f"success_reward={cost_model['success_reward']:.1f}\n"
        )

    for method_name, fn, extra in [
        ("MLM", run_MLM, {}),
        ("Human", run_Human, {}),
        ("PVEP_no_POMDP", run_pvep_no_pomdp, {}),
        (
            "PVEP_no_prompt",
            run_pvep_no_prompt,
            {"pomcp_max_it": pomcp_max_it, "pomcp_max_depth": pomcp_max_depth},
        ),
        (
            "PVEP",
            run_pvep,
            {"pomcp_max_it": pomcp_max_it, "pomcp_max_depth": pomcp_max_depth},
        ),
    ]:
        if verbose:
            print(f"  Running {method_name} ...", end=" ", flush=True)
        result = fn(n_episodes=n_episodes, seed=seed, **extra)
        result["obs_model"] = obs_model
        result["cost_model"] = cost_model
        result["initial_state_prior"] = dict(INITIAL_STATE_PRIOR)
        result["target_holes"] = list(TARGET_HOLES)
        result["rollout_policy"] = "tri_state_one_step_expected_cost"
        result["pomcp_config"] = {"max_it": pomcp_max_it, "max_depth": pomcp_max_depth}
        result["pvep_initial_output_corruption_epsilon"] = get_pvep_initial_output_corruption()
        result["pvep_initial_output_corruption_applied"] = method_name == "PVEP"
        results.append(result)
        if verbose:
            print(
                f"done  pass={result['pass_rate']:.3f}  "
                f"cost={result['avg_total_cost']:.2f}  "
                f"insp={result['avg_inspections']:.2f}"
            )

    if verbose:
        _print_table(results)
    return results


def _print_table(results: List[Dict]) -> None:
    print("\n" + "=" * 118)
    print(
        f"{'Method':<18} {'Pass Rate':>10} {'Total':>9} {'Inspect':>9} "
        f"{'Insert':>9} {'Recover':>9} {'Failure':>9} {'Success':>9} {'Fail Holes':>10} "
        f"{'Insp':>7} {'Ins':>7} {'Rec':>7}"
    )
    print("-" * 118)
    for result in results:
        print(
            f"{result['method']:<18} "
            f"{result['pass_rate']:>10.3f} "
            f"{result['avg_total_cost']:>9.2f} "
            f"{result['avg_inspect_cost']:>9.2f} "
            f"{result['avg_insert_cost']:>9.2f} "
            f"{result['avg_recover_cost']:>9.2f} "
            f"{result['avg_failure_cost']:>9.2f} "
            f"{result['avg_success_reward']:>9.2f} "
            f"{result['avg_fail_holes']:>10.3f} "
            f"{result['avg_inspections']:>7.2f} "
            f"{result['avg_inserts']:>7.2f} "
            f"{result['avg_recovers']:>7.2f}"
        )
    print("=" * 118)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare tri-state screw-installation methods")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--p-good", type=float, default=0.0, help="Deprecated; tri-state prior is fixed.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pomcp-iter", type=int, default=500)
    parser.add_argument("--pomcp-depth", type=int, default=10)
    parser.add_argument("--yolo-acc", type=float, default=None)
    parser.add_argument("--yolo-corruption", type=float, default=None)
    parser.add_argument("--pvep-initial-corruption", type=float, default=None)
    parser.add_argument("--obs-optimal", type=float, default=None)
    parser.add_argument("--probe-cost-fixed", type=float, default=None)
    parser.add_argument("--probe-cost-optimal", type=float, default=None)
    parser.add_argument("--insert-cost", type=float, default=None)
    parser.add_argument("--recover-cost", type=float, default=None)
    parser.add_argument("--failure-penalty", type=float, default=None)
    parser.add_argument("--success-reward", type=float, default=None)
    parser.add_argument(
        "--method",
        choices=["all", "MLM", "Human", "pvep_no_pomdp", "pvep_np", "pvep"],
        default="all",
    )
    args = parser.parse_args()
    set_pvep_initial_output_corruption(args.pvep_initial_corruption)

    if args.method == "all":
        run_all(
            n_episodes=args.episodes,
            p_good=args.p_good,
            seed=args.seed,
            pomcp_max_it=args.pomcp_iter,
            pomcp_max_depth=args.pomcp_depth,
            yolo_acc=args.yolo_acc,
            yolo_corruption=args.yolo_corruption,
            pvep_initial_corruption=args.pvep_initial_corruption,
            obs_optimal=args.obs_optimal,
            probe_cost_fixed=args.probe_cost_fixed,
            probe_cost_optimal=args.probe_cost_optimal,
            insert_cost=args.insert_cost,
            recover_cost=args.recover_cost,
            failure_penalty=args.failure_penalty,
            success_reward=args.success_reward,
        )
    elif args.method == "MLM":
        print(run_MLM(args.episodes, seed=args.seed))
    elif args.method == "Human":
        print(run_Human(args.episodes, seed=args.seed))
    elif args.method == "pvep_no_pomdp":
        print(run_pvep_no_pomdp(args.episodes, seed=args.seed))
    elif args.method == "pvep_np":
        print(
            run_pvep_no_prompt(
                args.episodes,
                seed=args.seed,
                pomcp_max_it=args.pomcp_iter,
                pomcp_max_depth=args.pomcp_depth,
            )
        )
    elif args.method == "pvep":
        print(
            run_pvep(
                args.episodes,
                seed=args.seed,
                pomcp_max_it=args.pomcp_iter,
                pomcp_max_depth=args.pomcp_depth,
            )
        )
