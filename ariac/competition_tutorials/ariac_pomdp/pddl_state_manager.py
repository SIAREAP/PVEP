from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import unified_planning as up
from unified_planning.io import PDDLWriter


def _map_value_into_problem(val, prob) -> Any:
    """Map a UP state's value node into a python value / object tied to `prob`."""
    # Bool constants
    if val.is_bool_constant():
        return bool(val.bool_constant_value())
    # Object values
    if val.is_object_exp():
        return prob.object(val.object().name)
    # Numeric values
    if val.is_int_constant():
        return int(val.int_constant_value())
    if val.is_real_constant():
        return float(val.real_constant_value())
    # Fallback (rare in typical classical benchmarks)
    return val


def snapshot_problem_from_state(problem, state) -> up.model.Problem:
    """
    Create a new Problem with the same domain/goal/objects as `problem`,
    but with :init set to exactly match `state`.

    Note: unified_planning's UPState stores a complete assignment in `._values`
    (includes both true and false). We'll rebuild init by fluent+object names so
    it is consistent with the cloned problem's symbols.
    """
    snap = problem.clone()
    values = getattr(state, "_values", None)
    if not isinstance(values, dict):
        raise TypeError(f"Unexpected State storage (no _values dict): {type(state)}")

    for exp, val in values.items():
        if not exp.is_fluent_exp():
            continue
        f = snap.fluent(exp.fluent().name)
        args = [snap.object(a.object().name) for a in exp.args]
        snap_exp = f(*args)
        snap.set_initial_value(snap_exp, _map_value_into_problem(val, snap))
    return snap


def _extract_block(tag: str, text: str) -> tuple[int, int]:
    """
    Return [start,end) indices for the outermost '(:{tag} ...)' block.
    """
    start = text.find(f"(:{tag}")
    if start < 0:
        raise ValueError(f"Missing block '(:{tag}'")
    depth = 0
    for idx in range(start, len(text)):
        ch = text[idx]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return start, idx + 1
    raise ValueError(f"Unbalanced parentheses in '(:{tag}'")


_SUBMITTED_RE = re.compile(r"\(submitted\s+([^\s\)]+)\)")


def _replace_goal_with_unsubmitted(problem: up.model.Problem, state: up.model.State, pddl_text: str) -> str:
    """
    Replace the (:goal ...) block so it only contains (submitted <order>) for orders not yet submitted in `state`.
    If all orders are already submitted, goal becomes (:goal (and)).
    """
    try:
        submitted = problem.fluent("submitted")
    except Exception:
        return pddl_text

    orders = [o for o in problem.all_objects if o.type.name == "order"]
    unsubmitted: list[str] = []
    for o in orders:
        try:
            is_sub = bool(state.get_value(submitted(o)).bool_constant_value())
        except Exception:
            is_sub = False
        if not is_sub:
            unsubmitted.append(o.name)

    goal_lines = ["(:goal (and"]
    for o in sorted(unsubmitted):
        goal_lines.append(f"  (submitted {o})")
    goal_lines.append("))")
    new_goal = "\n".join(goal_lines)

    try:
        s, e = _extract_block("goal", pddl_text)
    except ValueError:
        # If no goal block exists, just append at end.
        return pddl_text.rstrip() + "\n\n" + new_goal + "\n"
    return pddl_text[:s] + new_goal + pddl_text[e:]


def _remove_part_from_objects(objects_block: str, part_name: str) -> str:
    """
    Best-effort removal of `part_name` token from (:objects ...) block text.
    Keeps formatting mostly intact.
    """
    # Replace standalone token, then compress multiple spaces.
    s = re.sub(rf"(?<![\w\-]){re.escape(part_name)}(?![\w\-])", "", objects_block)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s


def _remove_part_related_inits(init_block: str, part_name: str) -> str:
    """
    Remove any top-level init facts that mention part_name as a token.

    IMPORTANT:
    Some writers may put `(:init` and one or more facts on the SAME line. A naive line-based removal
    can accidentally delete the `(:init` header and make the PDDL invalid. Here we remove *facts*
    (top-level S-expressions inside :init) instead of whole lines.
    """
    token_re = re.compile(rf"(?<![\\w\\-]){re.escape(part_name)}(?![\\w\\-])")

    # Find the start of content after '(:init'
    s = init_block
    hdr_idx = s.find("(:init")
    if hdr_idx < 0:
        # Fallback to old behavior if the block is unexpected
        out_lines: list[str] = []
        for ln in s.splitlines():
            if token_re.search(ln):
                continue
            out_lines.append(ln)
        return "\n".join(out_lines)

    # Locate the first '(' after '(:init' to start scanning facts
    i = hdr_idx + len("(:init")
    # We'll scan until the final ')' that closes the init block (assumed included in init_block)
    # Extract top-level expressions at depth 1 (inside the init list).
    facts: list[str] = []
    depth = 0
    cur = []
    in_fact = False

    # Scan characters starting right after '(:init'
    while i < len(s):
        ch = s[i]
        if ch == "(":
            depth += 1
            if depth == 1:
                # Starting a top-level fact
                in_fact = True
                cur = ["("]
            elif in_fact:
                cur.append("(")
        elif ch == ")":
            if in_fact:
                cur.append(")")
            depth -= 1
            if in_fact and depth == 0:
                # End of a top-level fact
                fact = "".join(cur).strip()
                if fact and (not token_re.search(fact)):
                    facts.append(fact)
                in_fact = False
                cur = []
        else:
            if in_fact:
                cur.append(ch)
        i += 1

    # Rebuild a clean, multi-line init block to avoid header-on-same-line issues
    out_lines = ["(:init"]
    for f in facts:
        out_lines.append(f"  {f}")
    out_lines.append(")")
    return "\n".join(out_lines)


def _parse_top_level_sexps(block_text: str, *, start_marker: str) -> tuple[str, list[str]]:
    """
    Parse a PDDL block like '(:init ...)' into (header, facts).

    - header: the exact '(:init' string (and any whitespace after it on that line is discarded)
    - facts: list of top-level S-expressions strings, e.g. '(at_robot floor bin1_1)' or '(= (total-cost) 0)'

    We do NOT try to preserve original formatting; this is for robust patching.
    """
    s = block_text
    hdr_idx = s.find(start_marker)
    if hdr_idx < 0:
        return (start_marker, [])

    i = hdr_idx + len(start_marker)
    depth = 0
    cur: list[str] = []
    in_fact = False
    facts: list[str] = []

    while i < len(s):
        ch = s[i]
        if ch == "(":
            depth += 1
            if depth == 1:
                in_fact = True
                cur = ["("]
            elif in_fact:
                cur.append("(")
        elif ch == ")":
            if in_fact:
                cur.append(")")
            depth -= 1
            if in_fact and depth == 0:
                fact = "".join(cur).strip()
                if fact:
                    facts.append(fact)
                in_fact = False
                cur = []
        else:
            if in_fact:
                cur.append(ch)
        i += 1

    return (start_marker, facts)


def _rebuild_block(*, start_marker: str, facts: Iterable[str], indent: str = "  ") -> str:
    out_lines = [start_marker]
    for f in facts:
        out_lines.append(f"{indent}{f}")
    out_lines.append(")")
    return "\n".join(out_lines)


def _ensure_init_fact(init_block: str, fact_line: str) -> str:
    """
    Ensure `fact_line` (e.g., '(gripper_empty floor)') exists inside an (:init ...) block text.
    Assumes init_block contains '(:init' line.
    """
    if fact_line in init_block:
        return init_block
    lines = init_block.splitlines()
    # Insert before the final ')' of the init block.
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip() == ")":
            lines.insert(i, f"  {fact_line}")
            return "\n".join(lines)
    # Fallback: append
    return init_block.rstrip() + f"\n  {fact_line}\n)"


@dataclass
class DropContext:
    dropped_part: str
    dropped_order: str | None = None
    dropped_slot: str | None = None


class PDDLStateManager:
    """
    Maintain a 'latest problem PDDL' that mirrors the UP oracle state.
    Also supports applying a 'drop' patch and trimming goal orders already submitted.
    """

    def __init__(self, *, domain_path: str, init_problem_path: str, out_dir: str, snapshot_prefix: str = "snapshot_problem_"):
        self.domain_path = str(domain_path)
        self.init_problem_path = str(init_problem_path)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot_prefix = snapshot_prefix
        self._counter = 0
        self._latest_path: Path | None = None

    @property
    def latest_pddl_path(self) -> Path | None:
        return self._latest_path

    def latest_pddl_text(self) -> str | None:
        if self._latest_path is None:
            return None
        return self._latest_path.read_text(encoding="utf-8", errors="replace")

    def update_from_up(self, *, problem: up.model.Problem, state: up.model.State, trim_goal_submitted: bool = True) -> Path:
        snap = snapshot_problem_from_state(problem, state)
        out_path = self.out_dir / f"{self.snapshot_prefix}{self._counter:04d}.pddl"
        self._counter += 1
        PDDLWriter(snap).write_problem(str(out_path))

        if trim_goal_submitted:
            txt = out_path.read_text(encoding="utf-8", errors="replace")
            txt2 = _replace_goal_with_unsubmitted(problem, state, txt)
            if txt2 != txt:
                out_path.write_text(txt2, encoding="utf-8")

        self._latest_path = out_path
        return out_path

    # ---------------------- Action-based snapshotting (robust, avoids UP writer quirks) ----------------------
    _AT_ROBOT_RE = re.compile(r"^\(at_robot\s+([^\s\)]+)\s+([^\s\)]+)\)$")
    _HAS_PART_GRIPPER_RE = re.compile(r"^\(has_part_gripper\s+([^\s\)]+)\)$")
    _HAS_TRAY_GRIPPER_RE = re.compile(r"^\(has_tray_gripper\s+([^\s\)]+)\)$")
    _GRIPPER_EMPTY_RE = re.compile(r"^\(gripper_empty\s+([^\s\)]+)\)$")
    _HOLDING_TRAY_RE = re.compile(r"^\(holding_tray\s+([^\s\)]+)\s+([^\s\)]+)\)$")
    _HOLDING_PART_RE = re.compile(r"^\(holding_part\s+([^\s\)]+)\s+([^\s\)]+)\)$")
    _TRAY_ON_SLOT_RE = re.compile(r"^\(tray_on_slot\s+([^\s\)]+)\s+([^\s\)]+)\)$")
    _ON_AGV_RE = re.compile(r"^\(on_agv\s+([^\s\)]+)\s+([^\s\)]+)\)$")
    _PART_ON_RE = re.compile(r"^\(part_on\s+([^\s\)]+)\s+([^\s\)]+)\)$")
    _IN_SLOT_RE = re.compile(r"^\(in_slot\s+([^\s\)]+)\s+([^\s\)]+)\)$")
    _SLOT_EMPTY_RE = re.compile(r"^\(slot_empty\s+([^\s\)]+)\)$")
    _AGV_AT_RE = re.compile(r"^\(agv_at\s+([^\s\)]+)\s+([^\s\)]+)\)$")
    _SUBMITTED_RE2 = re.compile(r"^\(submitted\s+([^\s\)]+)\)$")

    def update_from_action(self, *, pddl_text: str, action_name: str, args: list[str]) -> str:
        """
        Apply a minimal, deterministic patch to the (:init ...) block based on a successfully executed action.

        This is intentionally "domain-specific" for ariac_kitting actions and is used to preserve progress
        across replans/restarts without relying on UP PDDLWriter snapshots.
        """
        try:
            is_, ie = _extract_block("init", pddl_text)
        except ValueError:
            return pddl_text

        init_block = pddl_text[is_:ie]
        _, facts = _parse_top_level_sexps(init_block, start_marker="(:init")
        fact_set = set(f.strip() for f in facts if f.strip())

        def _remove_matching(rx: re.Pattern[str]) -> None:
            nonlocal fact_set
            fact_set = {f for f in fact_set if not rx.match(f)}

        def _ensure(f: str) -> None:
            fact_set.add(f)

        # --- Action semantics (subset) ---
        if action_name == "move_floor":
            # args: [robot, l1, l2]
            if len(args) >= 3:
                r, _, l2 = args[0], args[1], args[2]
                _remove_matching(re.compile(rf"^\(at_robot\s+{re.escape(r)}\s+([^\s\)]+)\)$"))
                _ensure(f"(at_robot {r} {l2})")

        elif action_name == "change_gripper":
            # args: [robot, gripper_station]
            if len(args) >= 1:
                r = args[0]
                has_part = any(self._HAS_PART_GRIPPER_RE.match(f) and self._HAS_PART_GRIPPER_RE.match(f).group(1) == r for f in fact_set)
                has_tray = any(self._HAS_TRAY_GRIPPER_RE.match(f) and self._HAS_TRAY_GRIPPER_RE.match(f).group(1) == r for f in fact_set)
                # Toggle according to domain effect.
                if has_tray:
                    fact_set.discard(f"(has_tray_gripper {r})")
                    _ensure(f"(has_part_gripper {r})")
                elif has_part:
                    fact_set.discard(f"(has_part_gripper {r})")
                    _ensure(f"(has_tray_gripper {r})")

        elif action_name == "floor_pick_tray":
            # args: [robot, tray, tray_slot]
            if len(args) >= 3:
                r, t, s = args[0], args[1], args[2]
                fact_set.discard(f"(tray_on_slot {t} {s})")
                _ensure(f"(holding_tray {r} {t})")
                fact_set.discard(f"(gripper_empty {r})")

        elif action_name == "floor_place_tray":
            # args: [robot, tray, agv, order]
            if len(args) >= 3:
                r, t, a = args[0], args[1], args[2]
                # Remove holding tray, mark tray on agv, gripper empty
                _remove_matching(re.compile(rf"^\(holding_tray\s+{re.escape(r)}\s+{re.escape(t)}\)$"))
                _ensure(f"(on_agv {t} {a})")
                _ensure(f"(gripper_empty {r})")

        elif action_name == "floor_pick_part":
            # args: [robot, part, bin_slot]
            if len(args) >= 3:
                r, p, s = args[0], args[1], args[2]
                fact_set.discard(f"(part_on {p} {s})")
                _ensure(f"(holding_part {r} {p})")
                fact_set.discard(f"(gripper_empty {r})")

        elif action_name == "floor_pick_part_from_agv":
            # args: [robot, part, agv_slot]
            if len(args) >= 3:
                r, p, s = args[0], args[1], args[2]
                fact_set.discard(f"(in_slot {p} {s})")
                _ensure(f"(slot_empty {s})")
                _ensure(f"(holding_part {r} {p})")
                fact_set.discard(f"(gripper_empty {r})")

        elif action_name == "floor_place_part":
            # args: [robot, part, agv_slot, agv, order]
            if len(args) >= 3:
                r, p, s = args[0], args[1], args[2]
                _remove_matching(re.compile(rf"^\(holding_part\s+{re.escape(r)}\s+{re.escape(p)}\)$"))
                _ensure(f"(in_slot {p} {s})")
                fact_set.discard(f"(slot_empty {s})")
                _ensure(f"(gripper_empty {r})")

        elif action_name == "floor_place_part_to_trash":
            # args: [robot, part]
            if len(args) >= 2:
                r, p = args[0], args[1]
                _remove_matching(re.compile(rf"^\(holding_part\s+{re.escape(r)}\s+{re.escape(p)}\)$"))
                _ensure(f"(part_on {p} trash)")
                _ensure(f"(gripper_empty {r})")

        elif action_name == "move_agv":
            # args may be ["agv4", "init_agv4", "warehouse"] or similar
            if len(args) >= 3:
                a, d1, d2 = args[0], args[1], args[2]
                fact_set.discard(f"(agv_at {a} {d1})")
                _ensure(f"(agv_at {a} {d2})")

        elif action_name == "submit_order":
            # args: [order, agv]
            if len(args) >= 1:
                o = args[0]
                _ensure(f"(submitted {o})")

        # Keep total-cost fact if present; do not attempt to update numeric costs here.
        rebuilt_init = _rebuild_block(start_marker="(:init", facts=sorted(fact_set))
        return pddl_text[:is_] + rebuilt_init + pddl_text[ie:]

    def write_action_snapshot(self, *, action_name: str, args: list[str]) -> Path | None:
        """
        Convenience: read current latest PDDL, apply action patch, and write as next snapshot.
        """
        txt = self.latest_pddl_text()
        if not txt:
            return None
        txt2 = self.update_from_action(pddl_text=txt, action_name=action_name, args=args)
        out_path = self.out_dir / f"{self.snapshot_prefix}{self._counter:04d}.pddl"
        self._counter += 1
        out_path.write_text(txt2, encoding="utf-8")
        self._latest_path = out_path
        return out_path

    def apply_drop_patch(self, *, pddl_text: str, dropped_part: str) -> str:
        """
        Given a latest PDDL text, remove any drop part occurrences from :objects and :init.
        This is a domain-agnostic patch (string-level), used only because 'dropped to floor' is not modeled.
        """
        s = pddl_text
        try:
            os, oe = _extract_block("objects", s)
            objects_block = s[os:oe]
            s = s[:os] + _remove_part_from_objects(objects_block, dropped_part) + s[oe:]
        except ValueError:
            pass

        try:
            is_, ie = _extract_block("init", s)
            init_block = s[is_:ie]
            patched_init = _remove_part_related_inits(init_block, dropped_part)
            # After a drop, ensure the gripper is empty in the symbolic snapshot.
            patched_init = _ensure_init_fact(patched_init, "(gripper_empty floor)")
            s = s[:is_] + patched_init + s[ie:]
        except ValueError:
            pass
        return s

    # ---------------------- Drop recovery: safe, deterministic reassignment ----------------------
    _ORDER_NEEDS_RE = re.compile(r"\(order_needs_part\s+([^\s\)]+)\s+([^\s\)]+)\s+([^\s\)]+)\)")
    _PART_ON_RE = re.compile(r"\(part_on\s+([^\s\)]+)\s+([^\s\)]+)\)")

    @staticmethod
    def _base_part_type(part_name: str) -> str:
        """blue_pump_4 -> blue_pump, red_battery_1 -> red_battery"""
        toks = part_name.split("_")
        if len(toks) >= 2 and toks[-1].isdigit():
            return "_".join(toks[:-1])
        return part_name

    @staticmethod
    def _parse_objects_parts_text(pddl_text: str) -> list[str]:
        """Extract part object names from (:objects ...) block."""
        try:
            os, oe = _extract_block("objects", pddl_text)
            objects_block = pddl_text[os:oe]
        except ValueError:
            return []
        parts: list[str] = []
        for ln in objects_block.splitlines():
            if "- part" not in ln:
                continue
            left = ln.split("- part")[0]
            toks = [t for t in left.split() if t.strip() and not t.strip().startswith("(:objects")]
            parts.extend(toks)
        return parts

    @staticmethod
    def _parse_init_order_needs(pddl_text: str) -> list[tuple[str, str, str]]:
        """Return list of (order, part, slot) from init."""
        try:
            is_, ie = _extract_block("init", pddl_text)
            init_block = pddl_text[is_:ie]
        except ValueError:
            init_block = pddl_text
        out: list[tuple[str, str, str]] = []
        for o, p, s in PDDLStateManager._ORDER_NEEDS_RE.findall(init_block):
            out.append((o, p, s))
        return out

    @staticmethod
    def _parse_init_part_on(pddl_text: str) -> dict[str, str]:
        """Return part -> location mapping from init part_on facts."""
        try:
            is_, ie = _extract_block("init", pddl_text)
            init_block = pddl_text[is_:ie]
        except ValueError:
            init_block = pddl_text
        out: dict[str, str] = {}
        for p, loc in PDDLStateManager._PART_ON_RE.findall(init_block):
            out[p] = loc
        return out

    @staticmethod
    def _replace_or_insert_order_need(*, pddl_text: str, order: str, slot: str, new_part: str) -> str:
        """
        Replace (order_needs_part order <any> slot) if present, otherwise insert a new one.
        """
        try:
            is_, ie = _extract_block("init", pddl_text)
            init_block = pddl_text[is_:ie]
        except ValueError:
            return pddl_text

        pat = re.compile(rf"\(order_needs_part\s+{re.escape(order)}\s+([^\s\)]+)\s+{re.escape(slot)}\)")
        if pat.search(init_block):
            init_block2 = pat.sub(f"(order_needs_part {order} {new_part} {slot})", init_block)
        else:
            # Insert before final ')'
            lines = init_block.splitlines()
            inserted = False
            for i in range(len(lines) - 1, -1, -1):
                if lines[i].strip() == ")":
                    lines.insert(i, f"    (order_needs_part {order} {new_part} {slot})")
                    inserted = True
                    break
            init_block2 = "\n".join(lines) if inserted else (init_block.rstrip() + f"\n    (order_needs_part {order} {new_part} {slot})\n)")

        return pddl_text[:is_] + init_block2 + pddl_text[ie:]

    @staticmethod
    def _ensure_good(*, pddl_text: str, part: str) -> str:
        """Ensure (good part) exists in init."""
        try:
            is_, ie = _extract_block("init", pddl_text)
            init_block = pddl_text[is_:ie]
        except ValueError:
            return pddl_text
        init_block2 = _ensure_init_fact(init_block, f"(good {part})")
        return pddl_text[:is_] + init_block2 + pddl_text[ie:]

    def reassign_dropped_part(
        self,
        *,
        pddl_text: str,
        dropped_part: str,
        dropped_order: str,
        dropped_slot: str,
    ) -> tuple[str | None, str | None]:
        """
        Deterministically pick an unused same-base-type replacement part and patch the PDDL accordingly.

        Returns: (new_pddl_text, chosen_replacement_part) or (None, None) if no candidate is available.
        """
        base = self._base_part_type(dropped_part)
        all_parts = self._parse_objects_parts_text(pddl_text)
        needs = self._parse_init_order_needs(pddl_text)
        used_parts = {p for _, p, _ in needs}
        part_locs = self._parse_init_part_on(pddl_text)

        # Candidate: same base type, not used by any order_needs_part, not the dropped part itself.
        candidates = [p for p in all_parts if (self._base_part_type(p) == base and p not in used_parts and p != dropped_part)]
        if not candidates:
            return (None, None)

        # Prefer candidates that are still on bins (have part_on location).
        candidates_on_bins = [p for p in candidates if p in part_locs]
        chosen = sorted(candidates_on_bins or candidates)[0]

        out = pddl_text
        out = self._replace_or_insert_order_need(pddl_text=out, order=dropped_order, slot=dropped_slot, new_part=chosen)
        out = self._ensure_good(pddl_text=out, part=chosen)
        return (out, chosen)

    def find_order_need_for_part(self, *, pddl_text: str, part_name: str) -> tuple[str | None, str | None]:
        """
        Return (order_id, agv_slot) for a specific part by searching (order_needs_part O P S).
        """
        try:
            is_, ie = _extract_block("init", pddl_text)
            init_block = pddl_text[is_:ie]
        except ValueError:
            init_block = pddl_text
        m = re.search(rf"\(order_needs_part\s+([^\s\)]+)\s+{re.escape(part_name)}\s+([^\s\)]+)\)", init_block)
        if not m:
            return (None, None)
        return (m.group(1), m.group(2))

    def list_submitted_orders(self, *, pddl_text: str) -> list[str]:
        try:
            is_, ie = _extract_block("init", pddl_text)
            init_block = pddl_text[is_:ie]
        except ValueError:
            init_block = pddl_text
        return sorted(set(_SUBMITTED_RE.findall(init_block)))

    def write_text_as_latest(self, *, pddl_text: str, filename: str = "latest_problem.pddl") -> Path:
        out_path = self.out_dir / filename
        out_path.write_text(pddl_text, encoding="utf-8")
        self._latest_path = out_path
        return out_path


