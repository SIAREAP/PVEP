from __future__ import annotations

import re
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import unified_planning as up
from unified_planning.engines.sequential_simulator import UPSequentialSimulator
from unified_planning.io import PDDLReader
from unified_planning.plans import ActionInstance


_ACTION_RE = re.compile(r"^\s*([a-zA-Z_][\w\-]*)\s*\((.*)\)\s*$")
_BIN_SLOT_RE = re.compile(r"^(bin\d+)_\d+$")


def _read_text_utf8(path: str | Path) -> str:
    p = Path(path)
    for enc in ("utf-8", "utf-8-sig"):
        try:
            return p.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return p.read_text(encoding="utf-8", errors="replace")


def _parse_action_str(action_str: str) -> tuple[str, list[str]]:
    s = action_str.strip()
    if s == "continue":
        return ("continue", [])
    m = _ACTION_RE.match(s)
    if not m:
        raise ValueError(f"Invalid action string: {action_str!r}")
    name = m.group(1)
    args_raw = m.group(2).strip()
    if not args_raw:
        return (name, [])
    args = [a.strip() for a in args_raw.split(",")]
    return (name, args)


@dataclass
class StepResult:
    ok: bool
    reason: str = ""


class RealWorldUPEnv:
    """
    Ground-truth environment simulated with unified_planning semantics.

    - Parses `domain.pddl + p_real.pddl`
    - Maintains a current `State`
    - Applies actions deterministically via `UPSequentialSimulator`
    - Samples inspect / QC observations from the ground-truth state
    """

    def __init__(self, domain_path: str | Path, problem_path: str | Path, *, seed: int = 0):
        self.domain_path = str(domain_path)
        self.problem_path = str(problem_path)
        self._rng = random.Random(int(seed))

        domain_str = _read_text_utf8(self.domain_path)
        problem_str = _read_text_utf8(self.problem_path)
        self.problem = PDDLReader().parse_problem_string(domain_str, problem_str)
        self.sim = UPSequentialSimulator(self.problem)
        self._state = self.sim.get_initial_state()

        self._obj = {o.name: o for o in self.problem.all_objects}
        self._flu = {f.name: f for f in self.problem.fluents}

        self._parts = [o for o in self.problem.all_objects if o.type.name == "part"]
        self._bins = [o for o in self.problem.all_objects if o.type.name == "bin_slot"]

    def reset(self, *, seed: int | None = None) -> None:
        if seed is not None:
            self._rng.seed(int(seed))
        self._state = self.sim.get_initial_state()

    @property
    def state(self) -> up.model.State:
        return self._state

    def _bool(self, fluent_name: str, *obj_names: str) -> bool:
        f = self._flu[fluent_name]
        objs = [self._obj[n] for n in obj_names]
        v = self._state.get_value(f(*objs))
        return bool(v.bool_constant_value())

    def apply_action(self, action_str: str) -> StepResult:
        name, args = _parse_action_str(action_str)
        if name == "continue":
            return StepResult(True, "noop")

        try:
            act = self.problem.action(name)
        except Exception as e:  # pragma: no cover
            return StepResult(False, f"unknown_action: {e}")

        try:
            params = [self._obj[a] for a in args]
        except KeyError as e:
            return StepResult(False, f"unknown_object: {e}")

        ai = ActionInstance(act, tuple(params))
        if not self.sim.is_applicable(self._state, ai):
            return StepResult(False, "not_applicable")

        ns = self.sim.apply(self._state, ai)
        if ns is None:
            return StepResult(False, "apply_failed")
        self._state = ns
        return StepResult(True, "")

    def check_action_applicable(self, action_str: str) -> StepResult:
        """
        Pre-check whether an action is applicable in the *current* ground-truth state, without executing it.

        This is useful for real-robot execution: we can avoid sending a command that we already know will fail.
        """
        name, args = _parse_action_str(action_str)
        if name == "continue":
            return StepResult(True, "noop")

        try:
            act = self.problem.action(name)
        except Exception as e:  # pragma: no cover
            return StepResult(False, f"unknown_action: {e}")

        try:
            params = [self._obj[a] for a in args]
        except KeyError as e:
            return StepResult(False, f"unknown_object: {e}")

        ai = ActionInstance(act, tuple(params))
        if not self.sim.is_applicable(self._state, ai):
            return StepResult(False, "not_applicable")
        return StepResult(True, "")

    def true_part_at_bin(self, bin_name: str) -> str:
        """Return the (unique) part at a bin slot, or 'none'."""
        for p in self._parts:
            if self._bool("part_on", p.name, bin_name):
                return p.name
        return "none"

    def true_bin_of_part(self, part_name: str) -> str | None:
        for b in self._bins:
            if self._bool("part_on", part_name, b.name):
                return b.name
        return None

    def true_quality(self, part_name: str) -> str:
        if "good" in self._flu and self._bool("good", part_name):
            return "good"
        if "bad" in self._flu and self._bool("bad", part_name):
            return "bad"
        return "unknown"

    def sample_inspect_obs(self, location: str, *, p_detect: float = 0.99) -> dict[str, Any]:
        """
        Inspect noise model (grouped by bin prefix):
        - If you inspect `bin1_9`, you get an observation of *all* `bin1_*` slots.
        - Each slot independently: see true part with p_detect else 'none'. No false positives.
        """
        m = _BIN_SLOT_RE.match(location)
        group = m.group(1) if m else None
        slots = [b.name for b in self._bins if (group is None or b.name.startswith(group + "_"))]
        slots = sorted(slots)

        slots_obs: dict[str, str] = {}
        for sl in slots:
            true_part = self.true_part_at_bin(sl)
            if true_part == "none":
                seen = "none"
            else:
                seen = true_part if (self._rng.random() < p_detect) else "none"
            slots_obs[sl] = seen

        return {"type": "inspect", "location": location, "bin_group": group or location, "slots": slots_obs}

    def sample_qc_obs(self, part: str, *, p_correct: float = 0.9) -> dict[str, Any]:
        """QC noise model: 90% correct label, 10% flipped."""
        q = self.true_quality(part)
        if q not in ("good", "bad"):
            # Unknown -> treat as maximally uncertain observation
            lbl = "good" if (self._rng.random() < 0.5) else "bad"
        else:
            if self._rng.random() < p_correct:
                lbl = q
            else:
                lbl = "bad" if q == "good" else "good"
        return {"type": "qc", "part": part, "label": lbl}

    def get_observable_facts(self) -> dict[str, Any]:
        """A minimal observable snapshot (expand as needed by the controller)."""
        out: dict[str, Any] = {}
        # Robot location (find first true at_robot)
        if "at_robot" in self._flu and "floor" in self._obj:
            floor = self._obj["floor"]
            at_robot = self._flu["at_robot"]
            loc_type = at_robot.signature[1].type
            loc = None
            for o in self.problem.all_objects:
                if o.type.is_subtype(loc_type):
                    if self._state.get_value(at_robot(floor, o)).bool_constant_value():
                        loc = o.name
                        break
            out["robot_loc"] = loc
        # Gripper
        for fn in ("has_part_gripper", "has_tray_gripper", "gripper_empty"):
            if fn in self._flu and "floor" in self._obj:
                out[fn] = self._bool(fn, "floor")
        return out

