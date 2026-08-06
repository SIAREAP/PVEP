from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import re
from unified_planning.io import PDDLReader
from unified_planning.shortcuts import OneshotPlanner
import unified_planning as up
from unified_planning.shortcuts import CompilationKind, Compiler


def _read_text_utf8(path: str | Path) -> str:
    p = Path(path)
    for enc in ("utf-8", "utf-8-sig"):
        try:
            return p.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return p.read_text(encoding="utf-8", errors="replace")


_NOT_EQUAL_RE = re.compile(r"\(\s*not\s*\(\s*=\s+[^)]+\)\s*\)")
_EQUALITY_REQ_RE = re.compile(r"(?m)(:requirements\s+[^)]*)")


def _sanitize_domain_for_strips(domain_pddl: str) -> str:
    """
    Make a best-effort PDDL rewrite to avoid :equality / (not (= ...)) so that
    we can compile down to a planner like pyperplan when Fast Downward isn't available.

    This is intentionally conservative: we only remove the `:equality` requirement
    and drop `(not (= ...))` subformulas (used here to prevent no-op moves).
    """
    s = domain_pddl
    # Remove :equality from :requirements
    s = s.replace(":equality", "")
    # Drop all (not (= ...)) constraints (typically in move_floor/move_agv)
    s = _NOT_EQUAL_RE.sub("", s)
    return s


def _ai_to_str(ai) -> str:
    name = ai.action.name
    args = ", ".join(p.object().name for p in ai.actual_parameters)
    return f"{name}({args})"


@dataclass(frozen=True)
class TaskSpec:
    order: str
    agv: str
    tray: str
    required_parts: dict[str, str]  # part -> target agv_slot


class ClassicalPlanner:
    """
    Fast deterministic replanning with unified_planning (Fast Downward).

    The planner receives an already-updated UP Problem instance (agent view).
    """

    def __init__(
        self,
        domain_path: str | Path,
        *,
        planner_name: str = "symk",
        optimal: bool = True,
    ):
        self.domain_path = str(domain_path)
        self._domain_str_raw = _read_text_utf8(self.domain_path)
        # Use a sanitized domain for planning so we can fall back to compilation+pyperplan.
        self._domain_str = _sanitize_domain_for_strips(self._domain_str_raw)
        self._planner_name = planner_name
        self._optimal = bool(optimal)

    def parse_problem(self, problem_path: str | Path):
        problem_str = _read_text_utf8(problem_path)
        return PDDLReader().parse_problem_string(self._domain_str, problem_str)

    def _compile_for_strips(self, problem):
        """
        Compile an ADL-ish problem down to a fragment typically supported by STRIPS planners.

        Returns: (compiled_problem, map_back_action_instance_chain)
        """
        p = problem
        mappers: list[Any] = []
        for ck in (
            CompilationKind.QUANTIFIERS_REMOVING,
            CompilationKind.CONDITIONAL_EFFECTS_REMOVING,
            CompilationKind.DISJUNCTIVE_CONDITIONS_REMOVING,
            CompilationKind.NEGATIVE_CONDITIONS_REMOVING,
            CompilationKind.GROUNDING,
        ):
            with Compiler(problem_kind=p.kind, compilation_kind=ck) as comp:
                res = comp.compile(p)
            mappers.append(res.map_back_action_instance)
            p = res.problem
        return p, mappers

    def solve(self, problem) -> list[str]:
        try:
            planner_kwargs: dict[str, Any] = {}
            if self._optimal:
                planner_kwargs["optimality_guarantee"] = up.engines.OptimalityGuarantee.SOLVED_OPTIMALLY
            with OneshotPlanner(name=self._planner_name, problem_kind=problem.kind, **planner_kwargs) as planner:
                res = planner.solve(problem)
        except (up.exceptions.UPNoRequestedEngineAvailableException, up.exceptions.UPNoSuitableEngineAvailableException):
            # Fallback: compile to STRIPS-ish and use any suitable planner (typically pyperplan).
            compiled, mappers = self._compile_for_strips(problem)
            try:
                with OneshotPlanner(problem_kind=compiled.kind) as planner:
                    res = planner.solve(compiled)
            except up.exceptions.UPNoSuitableEngineAvailableException as e:  # pragma: no cover
                raise RuntimeError(
                    "No suitable classical planner engine is available in unified_planning for this domain. "
                    "Either install `up-fast-downward` (recommended) or ensure a STRIPS-capable engine is present."
                ) from e

            if res.plan is not None:
                # Map back through compiler chain so the controller sees original-domain actions.
                mapped_actions = []
                for ai in res.plan.actions:
                    cur = ai
                    for map_back in reversed(mappers):
                        cur = map_back(cur)
                    mapped_actions.append(_ai_to_str(cur))
                return mapped_actions
        if res.plan is None:
            raise RuntimeError(f"classical_planning_failed: {res.status}")
        return [_ai_to_str(ai) for ai in res.plan.actions]


