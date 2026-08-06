"""
POMDP model for screw assembly under imperfect visual recognition.

Problem context
---------------
After `locating_screw` in the PDDL plan the robot has positioned a screw
over a hole using visual detection. Because recognition is not perfectly
accurate, the true alignment quality is a *hidden* variable:

  GOOD: detection was accurate → screw is well-aligned with the hole
  POOR: detection had an error → screw is misaligned

Before executing `insert_screw` the POMDP agent decides which action
minimises *expected total cost* from this point onward:

  PROCEED     – attempt insertion now (fast but risky if alignment is POOR)
  REINSPECT   – take another visual observation to update the belief, then
                decide again (costs time, provides information)
  RELOCATE    – redo the positioning with a corrective procedure that
                substantially raises the probability of GOOD alignment
                (higher cost, but nearly eliminates uncertainty)

Costs (each unit ≈ 1 normalised time/effort unit)
--------------------------------------------------
  PROCEED   success : 2
  PROCEED   failure : 2 + 20 rework penalty = 22
  REINSPECT         : 3
  RELOCATE          : 5

Observation model (REINSPECT only)
-----------------------------------
  If GOOD: P(GOOD_SIGNAL | GOOD) = 0.85, P(POOR_SIGNAL | GOOD) = 0.15
  If POOR: P(GOOD_SIGNAL | POOR) = 0.20, P(POOR_SIGNAL | POOR) = 0.80

Usage
-----
  This module exposes `ScrewAssemblyEnv` which implements the interface
  expected by `ibpomcp.pomcp.pomcp_planning` / `ibpomcp.ibpomcp.ibpomcp_planning`.
  Call `run_pomdp_for_screw()` from the hybrid controller to obtain the
  recommended action sequence for one screw.
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Action / Observation enumerations
# ---------------------------------------------------------------------------

class ScrewAction(IntEnum):
    PROCEED    = 0  # attempt screw insertion immediately
    REINSPECT  = 1  # take another visual observation to update belief
    RELOCATE   = 2  # redo screw positioning with higher accuracy


class ScrewObs(IntEnum):
    NONE         = 0  # no informative observation (after PROCEED / RELOCATE)
    GOOD_SIGNAL  = 1  # re-inspection suggests alignment is good
    POOR_SIGNAL  = 2  # re-inspection suggests alignment is poor


# ---------------------------------------------------------------------------
# State set (terminal-state checker required by pomcp)
# ---------------------------------------------------------------------------

class ScrewStateSet:
    """Checks whether a `ScrewAssemblyEnv` has reached a terminal state."""

    def is_final_state(self, state: "ScrewAssemblyEnv") -> bool:
        return state.phase in ("done", "failed")


# ---------------------------------------------------------------------------
# POMDP environment / state (combined, as expected by pomcp)
# ---------------------------------------------------------------------------

@dataclass
class ScrewAssemblyEnv:
    """
    Combined environment + state object for one screw assembly POMDP episode.

    Attributes that are *hidden* from the agent
    -------------------------------------------
    alignment_quality : str
        "good" or "poor" – the true alignment after locating_screw.

    Attributes that are *observable*
    ---------------------------------
    last_obs      : ScrewObs  – observation received on last step
    phase         : str       – "pre_insert" | "done" | "failed"
    step_count    : int       – number of POMDP steps taken

    Parameters
    ----------
    recognition_accuracy : float
        Probability that a single locating attempt yields GOOD alignment.
        Default 0.75 (i.e. 75% of locating calls are accurate).
    reinspect_cost       : float
    relocate_cost        : float
    insert_cost          : float
    failure_penalty      : float
        Extra cost charged when PROCEED is executed in a POOR alignment
        state and causes an assembly failure.
    p_insert_good        : float
        Probability that PROCEED succeeds given GOOD alignment.
    p_insert_poor        : float
        Probability that PROCEED succeeds given POOR alignment.
    p_tp                 : float
        True-positive rate of REINSPECT (P(GOOD_SIGNAL | GOOD)).
    p_fp                 : float
        False-positive rate of REINSPECT (P(GOOD_SIGNAL | POOR)).
    relocate_accuracy    : float
        Probability of GOOD alignment after RELOCATE (higher than initial).
    simulation           : bool
        Set to True when used as a simulation copy inside pomcp.
    max_steps            : int
        Maximum number of POMDP steps before forcing termination.
    """

    # Hidden state
    alignment_quality: str = "good"

    # Observable state
    last_obs: ScrewObs = ScrewObs.NONE
    phase: str = "pre_insert"
    step_count: int = 0

    # Parameters
    recognition_accuracy: float = 0.75
    reinspect_cost: float = 3.0
    relocate_cost: float = 5.0
    insert_cost: float = 2.0
    failure_penalty: float = 20.0
    p_insert_good: float = 0.92
    p_insert_poor: float = 0.35
    p_tp: float = 0.85   # P(GOOD_SIGNAL | GOOD)
    p_fp: float = 0.20   # P(GOOD_SIGNAL | POOR)
    relocate_accuracy: float = 0.93

    simulation: bool = False
    max_steps: int = 15

    # Required by pomcp node interface
    state_set: ScrewStateSet = field(default_factory=ScrewStateSet)

    # ------------------------------------------------------------------
    # Factory / sampling
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        *,
        recognition_accuracy: float = 0.75,
        reinspect_cost: float = 3.0,
        relocate_cost: float = 5.0,
        insert_cost: float = 2.0,
        failure_penalty: float = 20.0,
        p_insert_good: float = 0.92,
        p_insert_poor: float = 0.35,
        p_tp: float = 0.85,
        p_fp: float = 0.20,
        relocate_accuracy: float = 0.93,
        max_steps: int = 15,
        seed: Optional[int] = None,
    ) -> "ScrewAssemblyEnv":
        """Create and initialise an environment, sampling the hidden state."""
        if seed is not None:
            random.seed(seed)
        quality = "good" if random.random() < recognition_accuracy else "poor"
        return cls(
            alignment_quality=quality,
            last_obs=ScrewObs.NONE,
            phase="pre_insert",
            step_count=0,
            recognition_accuracy=recognition_accuracy,
            reinspect_cost=reinspect_cost,
            relocate_cost=relocate_cost,
            insert_cost=insert_cost,
            failure_penalty=failure_penalty,
            p_insert_good=p_insert_good,
            p_insert_poor=p_insert_poor,
            p_tp=p_tp,
            p_fp=p_fp,
            relocate_accuracy=relocate_accuracy,
            max_steps=max_steps,
            state_set=ScrewStateSet(),
        )

    def sample_state(self, _agent: Any) -> "ScrewAssemblyEnv":
        """
        Sample a new hidden state from the prior (used by pomcp particle filter).
        The observable parts are kept; only the hidden quality is resampled.
        """
        s = self.copy()
        s.alignment_quality = (
            "good" if random.random() < self.recognition_accuracy else "poor"
        )
        return s

    # ------------------------------------------------------------------
    # Core interface required by pomcp
    # ------------------------------------------------------------------

    def copy(self) -> "ScrewAssemblyEnv":
        """Deep-copy this env (used by pomcp for simulation branches)."""
        return copy.deepcopy(self)

    def get_observation(self) -> ScrewObs:
        """Return the last observation (visible to the agent)."""
        return self.last_obs

    def observation_is_equal(self, other: Any) -> bool:
        """Compare observations (used by pomcp to merge tree branches)."""
        return self.last_obs == other

    def get_actions_list(self) -> List[ScrewAction]:
        """Return valid actions in the current phase."""
        if self.phase != "pre_insert":
            return []
        return [ScrewAction.PROCEED, ScrewAction.REINSPECT, ScrewAction.RELOCATE]

    def default_policy(self) -> ScrewAction:
        """
        Heuristic rollout policy (used by pomcp during rollouts):
          - If last observation was GOOD_SIGNAL → PROCEED
          - If last observation was POOR_SIGNAL → RELOCATE
          - Otherwise → REINSPECT once, then PROCEED
        """
        if self.last_obs == ScrewObs.GOOD_SIGNAL:
            return ScrewAction.PROCEED
        if self.last_obs == ScrewObs.POOR_SIGNAL:
            return ScrewAction.RELOCATE
        return ScrewAction.REINSPECT

    def step(self, action: int) -> Tuple["ScrewAssemblyEnv", float, bool, Dict[str, Any]]:
        """
        Simulate one POMDP step.

        Returns
        -------
        next_state : ScrewAssemblyEnv
            The updated state (self mutated in place, also returned).
        reward : float
            Negative cost (pomcp maximises reward → minimises cost).
        done : bool
            True when phase is "done" or "failed", or max_steps reached.
        info : dict
            Diagnostic information.
        """
        self.step_count += 1
        reward = 0.0
        done = False
        info: Dict[str, Any] = {}

        action = ScrewAction(int(action))

        if action == ScrewAction.PROCEED:
            reward, done, info = self._step_proceed()

        elif action == ScrewAction.REINSPECT:
            reward, done, info = self._step_reinspect()

        elif action == ScrewAction.RELOCATE:
            reward, done, info = self._step_relocate()

        # Force termination if max_steps exceeded
        if self.step_count >= self.max_steps and not done:
            self.phase = "failed"
            done = True
            info["timeout"] = True

        return self, reward, done, info

    # ------------------------------------------------------------------
    # Internal step logic
    # ------------------------------------------------------------------

    def _step_proceed(self) -> Tuple[float, bool, Dict[str, Any]]:
        """Attempt screw insertion."""
        base_cost = self.insert_cost

        if self.alignment_quality == "good":
            success = random.random() < self.p_insert_good
        else:
            success = random.random() < self.p_insert_poor

        if success:
            self.phase = "done"
            self.last_obs = ScrewObs.NONE
            reward = -base_cost
            info = {"result": "success", "quality": self.alignment_quality}
        else:
            self.phase = "failed"
            self.last_obs = ScrewObs.NONE
            reward = -(base_cost + self.failure_penalty)
            info = {"result": "failure", "quality": self.alignment_quality}

        return reward, True, info

    def _step_reinspect(self) -> Tuple[float, bool, Dict[str, Any]]:
        """Visual re-inspection: provides a noisy observation about alignment."""
        reward = -self.reinspect_cost
        done = False

        if self.alignment_quality == "good":
            obs = ScrewObs.GOOD_SIGNAL if random.random() < self.p_tp else ScrewObs.POOR_SIGNAL
        else:
            obs = ScrewObs.GOOD_SIGNAL if random.random() < self.p_fp else ScrewObs.POOR_SIGNAL

        self.last_obs = obs
        info = {"result": "reinspect", "obs": obs.name, "quality": self.alignment_quality}
        return reward, done, info

    def _step_relocate(self) -> Tuple[float, bool, Dict[str, Any]]:
        """
        Redo screw positioning with a corrective procedure.
        Resamples alignment quality with higher accuracy than the initial attempt.
        """
        reward = -self.relocate_cost
        done = False

        self.alignment_quality = (
            "good" if random.random() < self.relocate_accuracy else "poor"
        )
        self.last_obs = ScrewObs.NONE
        info = {"result": "relocate", "new_quality": self.alignment_quality}
        return reward, done, info


# ---------------------------------------------------------------------------
# Ad-hoc agent (required by pomcp)
# ---------------------------------------------------------------------------

class ScrewAgent:
    """Minimal agent wrapper required by the pomcp planner."""

    def __init__(self) -> None:
        self.next_action: Optional[int] = None
        self.smart_parameters: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# High-level helper: run POMDP for one screw, return action sequence
# ---------------------------------------------------------------------------

def run_pomdp_for_screw(
    *,
    recognition_accuracy: float = 0.75,
    reinspect_cost: float = 3.0,
    relocate_cost: float = 5.0,
    insert_cost: float = 2.0,
    failure_penalty: float = 20.0,
    p_insert_good: float = 0.92,
    p_insert_poor: float = 0.35,
    p_tp: float = 0.85,
    p_fp: float = 0.20,
    relocate_accuracy: float = 0.93,
    max_steps: int = 15,
    pomcp_max_depth: int = 10,
    pomcp_max_it: int = 500,
    pomcp_discount: float = 0.95,
    pomcp_particles: int = 300,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run POMCP planning for a single screw assembly POMDP episode.

    The function creates an environment whose hidden state is sampled from the
    prior (based on `recognition_accuracy`), runs the POMDP agent until the
    screw is inserted (done/failed), and returns a summary.

    Returns
    -------
    dict with keys:
        actions      : List[str]  – sequence of action names taken
        total_cost   : float      – total cost incurred
        outcome      : str        – "success" | "failure" | "timeout"
        n_steps      : int        – number of POMDP steps
        true_quality : str        – "good" | "poor"  (for evaluation)
    """
    from ibpomcp.pomcp import pomcp_planning  # local import to avoid cycles

    env = ScrewAssemblyEnv.create(
        recognition_accuracy=recognition_accuracy,
        reinspect_cost=reinspect_cost,
        relocate_cost=relocate_cost,
        insert_cost=insert_cost,
        failure_penalty=failure_penalty,
        p_insert_good=p_insert_good,
        p_insert_poor=p_insert_poor,
        p_tp=p_tp,
        p_fp=p_fp,
        relocate_accuracy=relocate_accuracy,
        max_steps=max_steps,
        seed=seed,
    )

    agent = ScrewAgent()
    true_quality = env.alignment_quality

    total_cost = 0.0
    actions_taken: List[str] = []
    done = False

    for _t in range(max_steps):
        if env.state_set.is_final_state(env):
            break

        action, _info = pomcp_planning(
            env,
            agent,
            max_depth=pomcp_max_depth,
            max_it=pomcp_max_it,
            discount_factor=pomcp_discount,
            k=pomcp_particles,
        )
        agent.next_action = int(action)
        action_name = ScrewAction(int(action)).name

        _, reward, done, step_info = env.step(int(action))
        total_cost += -reward  # reward is negative cost

        if verbose:
            print(
                f"  t={_t} action={action_name} reward={reward:.1f} "
                f"obs={env.last_obs.name} phase={env.phase} info={step_info}"
            )

        actions_taken.append(action_name)
        if done:
            break

    outcome = env.phase  # "done" → success, "failed" → failure
    if outcome == "done":
        outcome = "success"

    return {
        "actions": actions_taken,
        "total_cost": total_cost,
        "outcome": outcome,
        "n_steps": env.step_count,
        "true_quality": true_quality,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run POMDP for screw assembly")
    parser.add_argument("--recognition-accuracy", type=float, default=0.75)
    parser.add_argument("--failure-penalty", type=float, default=20.0)
    parser.add_argument("--pomcp-iterations", type=int, default=500)
    parser.add_argument("--pomcp-depth", type=int, default=10)
    parser.add_argument("--pomcp-particles", type=int, default=300)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    result = run_pomdp_for_screw(
        recognition_accuracy=args.recognition_accuracy,
        failure_penalty=args.failure_penalty,
        pomcp_max_it=args.pomcp_iterations,
        pomcp_max_depth=args.pomcp_depth,
        pomcp_particles=args.pomcp_particles,
        seed=args.seed,
        verbose=not args.quiet,
    )
    print("\n=== Result ===")
    for k, v in result.items():
        print(f"  {k}: {v}")
