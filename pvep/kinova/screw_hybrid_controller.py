"""
Screw Assembly Hybrid Controller
=================================
Integrates classical PDDL planning with a POMDP layer for uncertain screw
assembly under imperfect visual recognition.

Architecture
------------
1. A cost-optimal classical planner (Fast Downward / SymK) produces an
   ordered action plan from the PDDL domain + problem.
2. The controller executes that plan action by action.
3. Before every `fasten_screw` action the controller pauses and invokes
   the per-screw fasten-decision (one of 4 methods).
4. The fasten decision is one of:
      pure_model    → trust visual label directly
      human         → confidence-threshold loop (≥0.4), max 3 reinspects
      pvep_no_prompt → POMCP with PROBE_FIXED (moderate accuracy)
      pvep          → POMCP with PROBE_OPTIMAL (high accuracy)
5. The method handles the screw and returns (cost, success).
6. The controller tracks the total accumulated cost and reports it at the end.

Running
-------
    python screw_hybrid_controller.py
    # or with explicit options:
    python screw_hybrid_controller.py \\
        --domain generated_pddl/domain.pddl \\
        --problem generated_pddl/p_real.pddl \\
        --method pvep \\
        --p-good 0.80 \\
        --pomcp-iterations 300
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Allow running as a standalone script from the project root.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
# unified_planning 在某些安装版本存在循环导入 bug（walkers→model→plans→walkers），
# 因此不在模块顶层导入，而是在 _plan() 内部延迟导入，绕过该问题。

from method_comparison import (
    P_GOOD_PRIOR,
    _run_one_screw_pure,
    _run_one_screw_human,
    _run_one_screw_pomcp,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_plan(plan: Any) -> List[str]:
    """
    Convert a unified_planning SequentialPlan to a list of action-name strings
    in the form  "action_name(arg1, arg2, ...)".
    """
    result: List[str] = []
    # unified_planning 1.3.0+ uses .actions instead of .action_instances
    actions = getattr(plan, 'actions', None) or getattr(plan, 'action_instances', [])
    for ai in actions:
        # Handle both ActionInstance and tuple formats
        if hasattr(ai, 'action'):
            name = ai.action.name
            params = [str(p) for p in ai.actual_parameters]
        else:
            # In newer versions, actions might be tuples (action, params)
            name = str(ai[0]) if isinstance(ai, tuple) else str(ai)
            params = [str(p) for p in ai[1]] if isinstance(ai, tuple) and len(ai) > 1 else []

        if params:
            result.append(f"{name}({', '.join(params)})")
        else:
            result.append(name)
    return result


def _action_name(action_str: str) -> str:
    """Extract the bare action name from 'action_name(arg1, arg2)'."""
    return action_str.split("(")[0].strip()


def _action_args(action_str: str) -> List[str]:
    """Extract argument list from 'action_name(arg1, arg2)'."""
    if "(" not in action_str:
        return []
    inner = action_str.split("(", 1)[1].rstrip(")")
    return [a.strip() for a in inner.split(",") if a.strip()]


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class ScrewHybridController:
    """
    Hybrid controller for TV screw assembly.

    Parameters
    ----------
    domain_path : str
        Path to the PDDL domain file.
    problem_path : str
        Path to the PDDL problem file.
    method : str
        Fasten-decision method: 'pure_model' | 'human' | 'pvep_no_prompt' | 'pvep'.
    p_good : float
        P(placement quality is GOOD after insert_screw).  Default 0.80.
    pomcp_max_it : int
        Number of POMCP simulations per decision.  Default 300.
    pomcp_max_depth : int
        Maximum search depth for POMCP.  Default 8.
    pomcp_particles : int
        Particle filter size.  Default 200.
    pomcp_discount : float
        Discount factor γ.  Default 0.95.
    verbose : bool
        Print per-step information.
    seed : int or None
        Random seed for reproducibility.
    """

    METHODS = ("pure_model", "human", "pvep_no_prompt", "pvep")

    def __init__(
        self,
        domain_path: str,
        problem_path: str,
        *,
        method: str = "pvep",
        p_good: float = P_GOOD_PRIOR,
        pomcp_max_it: int = 300,
        pomcp_max_depth: int = 8,
        pomcp_particles: int = 200,
        pomcp_discount: float = 0.95,
        verbose: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        if method not in self.METHODS:
            raise ValueError(f"method must be one of {self.METHODS}, got {method!r}")
        self.domain_path = str(domain_path)
        self.problem_path = str(problem_path)
        self.method = method
        self.p_good = float(p_good)
        self.pomcp_max_it = int(pomcp_max_it)
        self.pomcp_max_depth = int(pomcp_max_depth)
        self.pomcp_particles = int(pomcp_particles)
        self.pomcp_discount = float(pomcp_discount)
        self.verbose = bool(verbose)
        self.seed = seed

        self.total_cost: float = 0.0
        self.executed_actions: List[Dict[str, Any]] = []
        self.fasten_decisions: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def _plan(self) -> List[str]:
        """Call the classical planner and return the action sequence."""
        from unified_planning.io import PDDLReader
        from unified_planning.shortcuts import OneshotPlanner
        from unified_planning.engines.results import POSITIVE_OUTCOMES

        reader = PDDLReader()
        problem = reader.parse_problem(self.domain_path, self.problem_path)

        try:
            with OneshotPlanner(problem_kind=problem.kind) as planner:
                result = planner.solve(problem)
                if result.status in POSITIVE_OUTCOMES:
                    plan_str = _parse_plan(result.plan)
                    if self.verbose:
                        print(f"[Planner] {planner.name} found plan ({len(plan_str)} actions)")
                    return plan_str
                raise RuntimeError(f"Planner returned non-positive status: {result.status}")
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"Classical planner failed for {self.problem_path}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Fasten decision before fasten_screw
    # ------------------------------------------------------------------

    def _run_fasten_decision(
        self, screw: str, hole: str, screw_idx: int
    ) -> Tuple[float, bool]:
        """
        Run the selected fasten-decision method for one screw.

        Returns
        -------
        cost : float
            Total cost incurred (positive number).
        success : bool
            Whether the screw was fastened successfully.
        """
        # Sample the true placement quality for this screw
        rng_seed = (self.seed + screw_idx) if self.seed is not None else None
        if rng_seed is not None:
            random.seed(rng_seed)

        true_quality = "good" if random.random() < self.p_good else "poor"

        if self.method == "pure_model":
            reward, success = _run_one_screw_pure(true_quality)
        elif self.method == "human":
            reward, success = _run_one_screw_human(true_quality)
        elif self.method == "pvep_no_prompt":
            reward, success = _run_one_screw_pomcp(
                true_quality, "fixed",
                pomcp_max_it=self.pomcp_max_it,
                pomcp_max_depth=self.pomcp_max_depth,
                pomcp_particles=self.pomcp_particles,
                pomcp_discount=self.pomcp_discount,
            )
        else:  # pvep
            reward, success = _run_one_screw_pomcp(
                true_quality, "optimal",
                pomcp_max_it=self.pomcp_max_it,
                pomcp_max_depth=self.pomcp_max_depth,
                pomcp_particles=self.pomcp_particles,
                pomcp_discount=self.pomcp_discount,
            )

        # reward is signed (negative = cost), convert to positive cost
        cost = -reward
        return cost, success

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _record(self, action_str: str, *, fasten_triggered: bool = False) -> None:
        """Record an executed action."""
        self.executed_actions.append({"action": action_str, "fasten_triggered": fasten_triggered})
        if self.verbose:
            tag = "[FASTEN]" if fasten_triggered else "[PDDL]  "
            print(f"  {tag} {action_str}")

    def run(self) -> Dict[str, Any]:
        """
        Execute the hybrid plan.

        Returns
        -------
        summary : dict
            Keys: total_cost, executed_actions, fasten_decisions, plan_length,
                  n_failures, pass_rate
        """
        if self.verbose:
            print("=" * 60)
            print("Screw Assembly Hybrid Controller")
            print(f"  domain  : {self.domain_path}")
            print(f"  problem : {self.problem_path}")
            print(f"  method  : {self.method}")
            print(f"  P(good) : {self.p_good}")
            print(f"  POMCP iterations : {self.pomcp_max_it}")
            print("=" * 60)

        plan = self._plan()
        screw_idx = 0  # counter for screw-level episodes

        if self.verbose:
            print(f"\nClassical plan ({len(plan)} steps):")
            for i, a in enumerate(plan):
                print(f"  {i+1:2d}. {a}")
            print()

        n_failures = 0

        i = 0
        while i < len(plan):
            action_str = plan[i]
            aname = _action_name(action_str)
            aargs = _action_args(action_str)

            # ----------------------------------------------------------
            # Trigger: before fasten_screw → run fasten decision
            # ----------------------------------------------------------
            if aname == "fasten_screw" and len(aargs) >= 2:
                screw, hole = aargs[0], aargs[1]
                if self.verbose:
                    print(f"\n[Fasten trigger] Before fasten_screw({screw}, {hole})")

                cost, success = self._run_fasten_decision(screw, hole, screw_idx)
                screw_idx += 1

                self.total_cost += cost

                entry: Dict[str, Any] = {
                    "screw": screw,
                    "hole": hole,
                    "method": self.method,
                    "cost": cost,
                    "success": success,
                }
                self.fasten_decisions.append(entry)

                if not success:
                    n_failures += 1

                if self.verbose:
                    status = "SUCCESS" if success else "FAILURE"
                    print(
                        f"  Method={self.method}  cost={cost:.1f}  outcome={status}"
                    )

            # ----------------------------------------------------------
            # Execute the current PDDL action
            # ----------------------------------------------------------
            self._record(action_str, fasten_triggered=(aname == "fasten_screw"))
            i += 1

        pass_rate = (
            1.0 if n_failures == 0 else
            (screw_idx - n_failures) / screw_idx if screw_idx > 0 else 1.0
        )

        if self.verbose:
            print("\n" + "=" * 60)
            print(f"Assembly complete!")
            print(f"  Method              : {self.method}")
            print(f"  Total cost          : {self.total_cost:.1f}")
            print(f"  Total PDDL actions  : {len(self.executed_actions)}")
            print(f"  Screws processed    : {screw_idx}")
            print(f"  Failures            : {n_failures}")
            print(f"  Pass rate           : {pass_rate:.3f}")
            print("=" * 60)

        return {
            "total_cost": self.total_cost,
            "executed_actions": self.executed_actions,
            "fasten_decisions": self.fasten_decisions,
            "plan_length": len(plan),
            "n_screws": screw_idx,
            "n_failures": n_failures,
            "pass_rate": pass_rate,
        }


# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="screw_hybrid_controller",
        description="TV screw assembly: PDDL planning + fasten-decision POMDP (4 methods)",
    )
    parser.add_argument(
        "--domain",
        default=str(_HERE / "generated_pddl" / "domain.pddl"),
        help="Path to PDDL domain file",
    )
    parser.add_argument(
        "--problem",
        default=str(_HERE / "generated_pddl" / "p_real.pddl"),
        help="Path to PDDL problem file",
    )
    parser.add_argument(
        "--method",
        choices=["pure_model", "human", "pvep_no_prompt", "pvep"],
        default="pvep",
        help="Fasten-decision method. Default: pvep",
    )
    parser.add_argument(
        "--p-good",
        type=float,
        default=P_GOOD_PRIOR,
        help=f"P(placement quality GOOD after insert_screw). Default {P_GOOD_PRIOR}",
    )
    parser.add_argument(
        "--pomcp-iterations",
        type=int,
        default=300,
        help="POMCP simulation iterations per decision. Default 300",
    )
    parser.add_argument(
        "--pomcp-depth",
        type=int,
        default=8,
        help="POMCP max search depth. Default 8",
    )
    parser.add_argument(
        "--pomcp-particles",
        type=int,
        default=200,
        help="POMCP particle filter size. Default 200",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-step output",
    )
    args = parser.parse_args()

    ctrl = ScrewHybridController(
        domain_path=args.domain,
        problem_path=args.problem,
        method=args.method,
        p_good=args.p_good,
        pomcp_max_it=args.pomcp_iterations,
        pomcp_max_depth=args.pomcp_depth,
        pomcp_particles=args.pomcp_particles,
        verbose=not args.quiet,
        seed=args.seed,
    )
    summary = ctrl.run()
    print(f"\nFinal total cost : {summary['total_cost']:.1f}")
    print(f"Pass rate        : {summary['pass_rate']:.3f}")


if __name__ == "__main__":
    main()
