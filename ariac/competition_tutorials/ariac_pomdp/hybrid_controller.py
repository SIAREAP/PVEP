from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from unified_planning.plans import ActionInstance

from .classical_planner import ClassicalPlanner, TaskSpec
from .real_env_up import RealWorldUPEnv, _parse_action_str

import re

_UNKNOWN_OBJ_RE = re.compile(r"unknown_object:\s*'([^']+)'")
_BIN_GROUP_RE = re.compile(r"^(bin\\d+)_\\d+$")


class JuliaInfoClient(Protocol):
    def init_prior(self) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
        ...

    def solve_pre_pick(
        self,
        *,
        part: str,
        robot: str,
        candidate_bins: list[str],
        loc_probs: list[float],
        inspect_cost: float,
        miss_penalty: float,
    ) -> str:
        ...

    def solve_quality(
        self,
        *,
        trigger_type: str,
        order: str,
        part: str,
        slot: str,
        agv: str,
        p_good: float,
        qc_cost: float,
        repair_cost: float,
        fail_penalty: float,
    ) -> str:
        ...

    def update_loc_target(self, belief_loc: dict[str, dict[str, float]], *, part: str, location: str, saw: bool) -> None:
        ...

    def update_quality(self, p_good: float, *, obs_label: str) -> float:
        ...


@dataclass
class HybridParams:
    loc_conf_threshold: float = 0.85
    topk_bins: int = 3
    inspect_cost: float = 2.0
    qc_cost: float = 3.0
    repair_cost: float = 5.0
    miss_penalty: float = 25.0
    submit_fail_penalty: float = 300.0
    macro_max_steps: int = 30  # execute up to N deterministic actions per controller step()


def extract_task_from_problem(problem) -> TaskSpec:
    """Extract (order, agv, tray, required_parts mapping) from a UP problem's init state."""
    prob = problem
    from unified_planning.engines.sequential_simulator import UPSequentialSimulator

    s0 = UPSequentialSimulator(prob).get_initial_state()

    order_uses_tray = prob.fluent("order_uses_tray")
    order_needs_part = prob.fluent("order_needs_part")

    orders = [o for o in prob.all_objects if o.type.name == "order"]
    trays = [o for o in prob.all_objects if o.type.name == "tray"]
    agvs = [o for o in prob.all_objects if o.type.name == "agv"]
    parts = [o for o in prob.all_objects if o.type.name == "part"]
    slots = [o for o in prob.all_objects if o.type.name == "agv_slot"]

    req: dict[str, str] = {}
    chosen_order = None
    for o in orders:
        for p in parts:
            for sl in slots:
                if s0.get_value(order_needs_part(o, p, sl)).bool_constant_value():
                    req[p.name] = sl.name
                    chosen_order = chosen_order or o.name

    chosen_tray = None
    chosen_agv = None
    if chosen_order is not None:
        o_obj = prob.object(chosen_order)
        for t in trays:
            for a in agvs:
                if s0.get_value(order_uses_tray(o_obj, t, a)).bool_constant_value():
                    chosen_tray = t.name
                    chosen_agv = a.name
                    break
            if chosen_tray is not None:
                break

    if chosen_order is None or chosen_tray is None or chosen_agv is None or not req:
        raise RuntimeError("Failed to extract task (order_needs_part / order_uses_tray) from init.")

    return TaskSpec(order=chosen_order, agv=chosen_agv, tray=chosen_tray, required_parts=req)


def _set_all(problem, fluent_name: str, tuples: list[tuple[str, ...]], value: bool) -> None:
    f = problem.fluent(fluent_name)
    for args in tuples:
        exp = f(*[problem.object(a) for a in args])
        problem.set_initial_value(exp, value)


def build_agent_problem_from(env: RealWorldUPEnv, base_problem, belief_loc, belief_good, task: TaskSpec):
    """
    Build the classical planning problem from:
    - observable fluents from the *real* UP state (treated as perfectly observed)
    - hidden fluents committed via belief (MAP)
    """
    prob = base_problem.clone()
    s = env.state

    # --- Object universe alignment (agent-side) ---
    # After an inspect, beliefs may correctly point to bin slots that were not present in the original
    # agent model problem file (p_vlm). If the agent problem can't represent those objects, the planner
    # will see required parts as "nowhere" and can return UNSOLVABLE_PROVEN.
    type_by_name = {t.name: t for t in prob.user_types}
    existing = {o.name for o in prob.all_objects}
    for o in env.problem.all_objects:
        if o.name in existing:
            continue
        t = type_by_name.get(o.type.name)
        if t is None:
            continue
        prob.add_object(o.name, t)
        existing.add(o.name)

    # Convenience
    objs = {o.name: o for o in prob.all_objects}
    flu = {f.name: f for f in prob.fluents}

    def bval(fname: str, *args: str) -> bool:
        try:
            v = s.get_value(flu[fname](*[objs[a] for a in args]))
            return bool(v.bool_constant_value())
        except Exception:
            # If object (e.g. 'missing' part) not in real state, assume False (CWA)
            return False

    # --- Robot location: ensure exactly one at_robot is true
    floor = "floor"
    at_robot = flu["at_robot"]
    loc_objs = [o for o in prob.all_objects if o.type.is_subtype(at_robot.signature[1].type)]
    for lo in loc_objs:
        prob.set_initial_value(at_robot(objs[floor], lo), False)
    true_loc = env.get_observable_facts().get("robot_loc")
    if true_loc is not None:
        prob.set_initial_value(at_robot(objs[floor], objs[true_loc]), True)

    # --- Simple robot booleans
    for fname in ("has_part_gripper", "has_tray_gripper", "gripper_empty"):
        if fname in flu:
            prob.set_initial_value(flu[fname](objs[floor]), bval(fname, floor))

    # --- Holding predicates (observable)
    if "holding_part" in flu:
        holding_part = flu["holding_part"]
        # Sync for all parts (not just required parts), to keep the agent problem consistent.
        all_parts = [o for o in prob.all_objects if o.type.name == "part"]
        for p_obj in all_parts:
            prob.set_initial_value(
                holding_part(objs[floor], p_obj),
                bval("holding_part", floor, p_obj.name),
            )
    if "holding_tray" in flu:
        holding_tray = flu["holding_tray"]
        # Sync for all trays (multi-order support).
        all_trays = [o for o in prob.all_objects if o.type.name == "tray"]
        for t_obj in all_trays:
            prob.set_initial_value(
                holding_tray(objs[floor], t_obj),
                bval("holding_tray", floor, t_obj.name),
            )

    # --- AGV position (sync all AGVs)
    if "agv_at" in flu:
        agv_at = flu["agv_at"]
        dests = [o for o in prob.all_objects if o.type.is_subtype(agv_at.signature[1].type)]
        all_agvs = [o for o in prob.all_objects if o.type.name == "agv"]
        for a_obj in all_agvs:
            for d in dests:
                prob.set_initial_value(agv_at(a_obj, d), False)
            for d in dests:
                if bval("agv_at", a_obj.name, d.name):
                    prob.set_initial_value(agv_at(a_obj, d), True)
                    break

    # --- Tray placement (tray_on_slot / on_agv) (sync all trays)
    if "tray_on_slot" in flu:
        tray_on_slot = flu["tray_on_slot"]
        slots = [o for o in prob.all_objects if o.type.is_subtype(tray_on_slot.signature[1].type)]
        all_trays = [o for o in prob.all_objects if o.type.name == "tray"]
        for t_obj in all_trays:
            for sl in slots:
                prob.set_initial_value(tray_on_slot(t_obj, sl), False)
            for sl in slots:
                if bval("tray_on_slot", t_obj.name, sl.name):
                    prob.set_initial_value(tray_on_slot(t_obj, sl), True)
                    break
    if "on_agv" in flu:
        on_agv = flu["on_agv"]
        all_trays = [o for o in prob.all_objects if o.type.name == "tray"]
        all_agvs = [o for o in prob.all_objects if o.type.name == "agv"]
        for t_obj in all_trays:
            for a_obj in all_agvs:
                prob.set_initial_value(on_agv(t_obj, a_obj), bval("on_agv", t_obj.name, a_obj.name))

    # --- Slots: in_slot and slot_empty
    if "in_slot" in flu and "slot_empty" in flu:
        in_slot = flu["in_slot"]
        slot_empty = flu["slot_empty"]
        all_slots = [o for o in prob.all_objects if o.type.name == "agv_slot"]
        # default: slot_empty true, then mark occupied
        for sl in all_slots:
            prob.set_initial_value(slot_empty(sl), True)
        # Sync in_slot for all parts/slots based on observation, and set slot_empty accordingly.
        all_parts = [o for o in prob.all_objects if o.type.name == "part"]
        for p_obj in all_parts:
            for sl in all_slots:
                val = bval("in_slot", p_obj.name, sl.name)
                prob.set_initial_value(in_slot(p_obj, sl), val)
                if val:
                    prob.set_initial_value(slot_empty(sl), False)

    # --- Hidden: part_on for required parts only (MAP commitment)
    if "part_on" in flu:
        part_on = flu["part_on"]
        bins = [o for o in prob.all_objects if o.type.name == "bin_slot"]
        for p in task.required_parts.keys():
            # If already in slot or being held, don't place it on a bin.
            sl_req = task.required_parts[p]
            placed = ("in_slot" in flu) and bval("in_slot", p, sl_req)
            held = ("holding_part" in flu) and bval("holding_part", "floor", p)
            if placed or held:
                # Must not be on any bin if it's already placed/held.
                for b in bins:
                    prob.set_initial_value(part_on(objs[p], b), False)
                continue
            dist = belief_loc.get(p, {})
            if dist:
                # Override base_problem guess with the MAP bin from belief.
                for b in bins:
                    prob.set_initial_value(part_on(objs[p], b), False)
                b_hat = max(dist.items(), key=lambda kv: kv[1])[0]
                if b_hat in objs:
                    prob.set_initial_value(part_on(objs[p], objs[b_hat]), True)
            else:
                # If belief is empty/uninitialized, keep whatever base_problem currently encodes for part_on.
                # Clearing here would make the part "nowhere" and can lead to UNSOLVABLE_PROVEN.
                pass

    # --- Hidden: quality commitment
    if "good" in flu and "bad" in flu:
        good = flu["good"]
        bad = flu["bad"]
        for p in task.required_parts.keys():
            pg = float(belief_good.get(p, 0.5))
            is_good = pg >= 0.5
            prob.set_initial_value(good(objs[p]), is_good)
            prob.set_initial_value(bad(objs[p]), not is_good)

    # --- Submitted flag
    if "submitted" in flu:
        submitted = flu["submitted"]
        # Multi-order: sync all orders' submitted flags.
        all_orders = [o for o in prob.all_objects if o.type.name == "order"]
        for o_obj in all_orders:
            prob.set_initial_value(submitted(o_obj), bval("submitted", o_obj.name))

    return prob


class HybridController:
    def __init__(
        self,
        env: RealWorldUPEnv,
        planner: ClassicalPlanner,
        info: JuliaInfoClient,
        *,
        base_problem_path: str | None = None,
        params: HybridParams | None = None,
    ):
        self.env = env
        self.planner = planner
        self.info = info
        self.params = params or HybridParams()

        # Agent-side planning model should come from p_vlm (what the robot \"knows\" initially),
        # not from p_real (ground-truth).
        if base_problem_path is None:
            base_problem_path = getattr(info, "p_vlm_path", None)
        if not base_problem_path:
            raise ValueError("base_problem_path must be provided (typically p_vlm.pddl).")
        self.base_problem = planner.parse_problem(base_problem_path)
        self.task = extract_task_from_problem(self.base_problem)

        self.belief_loc, self.belief_good = info.init_prior()

        self.plan: list[str] = []
        # Simple memory to avoid spamming repeated QC triggers on the same part.
        self._qc_attempts: dict[str, int] = {}
        # Execution-time discovered unreachable/invalid locations (e.g., object missing from p_real).
        # We inject these into agent replanning as `floor_forbid_reach` to avoid repeated failures.
        self._exec_forbid_locs: set[str] = set()
        # Anti-loop: remember repeated inspect decisions per part.
        self._last_inspect_loc: dict[str, str] = {}
        self._repeat_inspect_count: dict[tuple[str, str], int] = {}
        # Cache: bin_group -> representative bin_slot name (a reachable slot to stand on for inspect)
        self._bin_group_rep: dict[str, str] = {}

        # --- Prior smoothing (user request) ---
        # Convert a sparse VLM prior into a slot-level prior over a fixed candidate set (default: 32 slots):
        # - If VLM gave slots for a part: put `p_high` mass on those slots (proportional to their prior weights)
        #   and distribute the remaining (1-p_high) uniformly over the other candidate slots.
        # - If VLM gave nothing for a part: distribute uniformly over all candidate slots.
        p_high = float(getattr(info, "p_high", 0.8))
        self._smooth_prior_loc(p_high=p_high, max_slots=32)

    def _bin_group(self, bin_slot: str) -> str | None:
        m = _BIN_GROUP_RE.match(str(bin_slot))
        return m.group(1) if m else None

    def _reachable_bin_groups(self) -> list[str]:
        """Return bin groups that have at least one representative bin_slot in the real env."""
        groups: set[str] = set()
        for o in self.env.problem.all_objects:
            if o.type.name != "bin_slot":
                continue
            g = self._bin_group(o.name)
            if g:
                groups.add(g)
        # Keep only groups for which we can pick a representative slot (prefer not forbidden).
        out: list[str] = []
        for g in sorted(groups):
            if self._get_bin_group_rep(g) is not None:
                out.append(g)
        return out

    def _prior_candidate_slots(self, *, max_slots: int = 32) -> list[str]:
        """
        Return a stable list of candidate bin slots for the initial location prior.

        We default to a capped set (e.g., 32 slots) to keep the POMDP's effective search space small
        while still allowing exploration beyond VLM detections.
        """
        max_slots = max(1, int(max_slots))
        slots = [
            o.name
            for o in self.env.problem.all_objects
            if (o.type.name == "bin_slot" and not self._is_forbidden_loc(o.name))
        ]
        slots = sorted(slots)
        # Cap to the first N slots (stable order).
        return slots[:max_slots] if len(slots) > max_slots else slots

    def _smooth_prior_loc(self, *, p_high: float = 0.8, max_slots: int = 32) -> None:
        """
        Prior smoothing over a fixed candidate set of bin *slots* (default: 32):
        - If VLM gave slots for a part: allocate `p_high` mass to those detected slots (proportional to their mass),
          and allocate the remaining (1-p_high) uniformly over all other candidate slots.
        - If VLM gave nothing for a part: allocate uniformly over all candidate slots.
        """
        p_high = max(0.0, min(1.0, float(p_high)))
        cand = self._prior_candidate_slots(max_slots=max_slots)
        if not cand:
            return
        cand_set = set(cand)

        for part in self.task.required_parts.keys():
            dist0 = dict(self.belief_loc.get(part, {}) or {})
            # Keep only detected slots that are in the candidate set.
            detected = {sl: float(p) for sl, p in dist0.items() if sl in cand_set and float(p) > 0.0}

            new_dist: dict[str, float] = {}
            if detected:
                z = sum(detected.values())
                if z > 0:
                    for sl, p in detected.items():
                        new_dist[sl] = new_dist.get(sl, 0.0) + p_high * (p / z)
            # Remaining mass distributed uniformly over all other candidate slots.
            rem_mass = 1.0 - (p_high if detected else 0.0)
            others = [sl for sl in cand if sl not in detected]
            if rem_mass > 0:
                if others:
                    per = rem_mass / float(len(others))
                    for sl in others:
                        new_dist[sl] = new_dist.get(sl, 0.0) + per
                else:
                    # All mass was on detected; nothing else to spread.
                    pass

            # Normalize (defensive)
            z2 = sum(float(v) for v in new_dist.values())
            if z2 > 0:
                for k in list(new_dist.keys()):
                    new_dist[k] = float(new_dist[k]) / z2
                self.belief_loc[part] = new_dist

    def _is_forbidden_loc(self, loc: str) -> bool:
        # If floor_forbid_reach isn't present, nothing is forbidden.
        try:
            f = self.env.problem.fluent("floor_forbid_reach")
        except Exception:
            return False
        try:
            o = self.env.problem.object(str(loc))
        except Exception:
            return False
        try:
            return bool(self.env.state.get_value(f(o)).bool_constant_value())
        except Exception:
            return False

    def _get_bin_group_rep(self, group: str) -> str | None:
        """
        Return a representative bin_slot for a bin group (e.g. 'bin5' -> 'bin5_1' or another reachable slot).

        We prefer a non-forbidden slot so that `move_floor` and `inspect` are applicable under the PDDL rules.
        """
        g = str(group)
        if g in self._bin_group_rep:
            return self._bin_group_rep[g]
        # Collect all bin slots in this group from the real env object universe.
        slots = [
            o.name
            for o in self.env.problem.all_objects
            if (o.type.name == "bin_slot" and o.name.startswith(g + "_"))
        ]
        slots = sorted(slots)
        # Prefer a reachable (not forbidden) slot.
        for s in slots:
            if not self._is_forbidden_loc(s):
                self._bin_group_rep[g] = s
                return s
        # Fall back to any slot (even if forbidden) so we at least have a stable mapping.
        if slots:
            self._bin_group_rep[g] = slots[0]
            return slots[0]
        return None

    def _topk_bin_groups(self, part: str) -> tuple[list[str], list[float], float]:
        """
        Aggregate belief over bin slots into bin groups.
        Returns (groups, probs, confidence_of_top_group).
        """
        dist = self.belief_loc.get(part, {}) or {}
        if not dist:
            return ([], [], 0.0)
        agg: dict[str, float] = {}
        for sl, p in dist.items():
            g = self._bin_group(sl)
            if not g:
                continue
            agg[g] = agg.get(g, 0.0) + float(p)
        if not agg:
            return ([], [], 0.0)
        items = sorted(agg.items(), key=lambda kv: kv[1], reverse=True)
        top = items[: max(1, int(self.params.topk_bins))]
        groups = [g for g, _ in top]
        probs = [float(p) for _, p in top]
        conf = probs[0] if probs else 0.0
        return (groups, probs, conf)

    def _python_update_belief_from_inspect(self, *, part: str, obs: dict[str, Any], p_detect: float = 0.99) -> None:
        """
        Robust belief update from an inspect observation, independent of Julia.

        Julia's updater may keep a fixed support set; inspect can reveal the true bin slot (e.g. bin1_1)
        even if it wasn't present in the prior. Without this, the classical planner can keep chasing
        stale locations and we loop on inspect/replan.
        """
        if not isinstance(obs, dict):
            return
        slots_obs = obs.get("slots")
        if not isinstance(slots_obs, dict) or not slots_obs:
            return

        dist = self.belief_loc.setdefault(part, {})
        # If the part is seen in any slot, collapse belief to that slot.
        for sl, seen in slots_obs.items():
            if str(seen) == part:
                dist.clear()
                dist[str(sl)] = 1.0
                return

        # Otherwise downweight inspected slots (miss probability).
        miss = max(0.0, min(1.0, 1.0 - float(p_detect)))
        for sl in slots_obs.keys():
            sl = str(sl)
            if sl in dist:
                dist[sl] *= miss
        z = sum(float(v) for v in dist.values())
        if z > 0:
            for k in list(dist.keys()):
                dist[k] = float(dist[k]) / z

    def _update_beliefs_from_inspect_obs(self, *, target_part: str, obs: dict[str, Any]) -> None:
        """
        Apply an inspect observation to beliefs.

        Critical: inspect may reveal other required parts (e.g. blue_regulator_1) while the trigger
        was for a different target_part (e.g. blue_battery_1). If we only update the target_part,
        the planner can keep generating stale picks for other parts.
        """
        if not isinstance(obs, dict):
            return
        slots_obs = obs.get("slots")
        if not isinstance(slots_obs, dict) or not slots_obs:
            return

        # 1) Update the target part (pure Python). We avoid calling Julia here because the Julia updater
        # can assume a fixed support and may clear / overwrite belief_loc in ways that make the classical
        # replanning problem UNSOLVABLE when a distribution becomes empty.
        self._python_update_belief_from_inspect(part=target_part, obs=obs)

        # 2) Positive updates for any other required parts observed in these slots.
        for sl, seen_part in slots_obs.items():
            sp = str(seen_part)
            if sp == "none":
                continue
            if sp in self.task.required_parts and sp != target_part:
                # Collapse belief for that part to the observed slot.
                self.belief_loc.setdefault(sp, {})
                self.belief_loc[sp].clear()
                self.belief_loc[sp][str(sl)] = 1.0

    def _replan(self) -> None:
        agent_prob = build_agent_problem_from(self.env, self.base_problem, self.belief_loc, self.belief_good, self.task)
        # Inject execution-time forbiddens into the agent model (only if the object exists in the agent problem).
        if self._exec_forbid_locs and "floor_forbid_reach" in {f.name for f in agent_prob.fluents}:
            ffr = agent_prob.fluent("floor_forbid_reach")
            objs = {o.name: o for o in agent_prob.all_objects}
            for loc in sorted(self._exec_forbid_locs):
                o = objs.get(loc)
                if o is not None:
                    agent_prob.set_initial_value(ffr(o), True)
        try:
            self.plan = self.planner.solve(agent_prob)
        except RuntimeError as e:
            # Do not crash the loop; surface as empty plan and let caller decide.
            self.plan = []
            self._last_plan_error = str(e)

    def _note_unknown_object(self, reason: str, *, action_name: str, action_args: list[str]) -> None:
        """
        If execution fails due to `unknown_object`, remember the relevant location so replanning avoids it.

        Rationale: planning is done on the agent-side `p_vlm`, execution on the real `p_real`.
        If `p_vlm` contains a location absent from `p_real`, repeatedly replanning from the same
        agent model can loop forever. We treat such locations as forbidden destinations.
        """
        m = _UNKNOWN_OBJ_RE.search(str(reason))
        if not m:
            return
        missing = m.group(1)
        # For motion/inspect-like actions, the missing object is usually the destination/location.
        if action_name in ("move_floor", "inspect"):
            if action_args:
                self._exec_forbid_locs.add(action_args[-1])
        # Always forbid the missing token itself as a last resort (if it's a location name).
        self._exec_forbid_locs.add(missing)

    def _qc_worth_it(self, p_good: float, *, qc_cost: float, repair_cost: float, fail_penalty: float) -> bool:
        """
        Heuristic VOI gate: do QC if expected failure penalty dwarfs the cost of information + (maybe) repair.

        Approximation: QC then repair-if-bad (expected) vs continuing and risking failure.
        """
        p_bad = max(0.0, min(1.0, 1.0 - float(p_good)))
        expected_continue = p_bad * float(fail_penalty)
        expected_qc_then_repair = float(qc_cost) + p_bad * float(repair_cost)
        return expected_qc_then_repair < expected_continue

    def _risk_ranked_parts(self) -> list[str]:
        """Return required parts sorted by decreasing expected submit risk (p_bad * penalty)."""
        fp = float(self.params.submit_fail_penalty)
        scored = []
        for p in self.task.required_parts.keys():
            pg = float(self.belief_good.get(p, 0.5))
            scored.append(((1.0 - pg) * fp, p))
        scored.sort(reverse=True)
        return [p for _, p in scored]

    def _topk_bins(self, part: str) -> tuple[list[str], list[float], float]:
        dist = self.belief_loc.get(part, {})
        if not dist:
            return ([], [], 0.0)
        items = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)
        top = items[: max(1, self.params.topk_bins)]
        bins = [b for b, _ in top]
        probs = [float(p) for _, p in top]
        return (bins, probs, probs[0] if probs else 0.0)

    def step(self) -> dict[str, Any]:
        """
        Macro-step controller:
        - executes up to `macro_max_steps` classical actions per call
        - only calls Julia at triggers
        - after any info action, immediately replans and continues (so we \"exit POMDP\" back into PDDL)
        """
        events: list[dict[str, Any]] = []

        for _ in range(max(1, int(self.params.macro_max_steps))):
            if not self.plan:
                self._replan()
            if not self.plan:
                reason = getattr(self, "_last_plan_error", None) or "empty_plan"
                return {"done": True, "reason": reason, "events": events}

            a = self.plan[0]
            aname, args = _parse_action_str(a)

            # Trigger A: before picking a required part (location uncertainty)
            if aname == "floor_pick_part":
                part = args[1]
                # Always consult the info-POMDP before a pick (user request).
                # Keep candidate set bounded and *bin-level*: planned bin-group + belief top-k bin-groups.
                planned_bin = args[2]
                planned_group = self._bin_group(planned_bin) or planned_bin
                groups, probs, _conf = self._topk_bin_groups(part)

                # If belief is empty, start from the planned bin so POMDP can at least confirm/refute it.
                if not groups:
                    groups = [planned_group]
                    probs = [1.0]

                # Ensure the planned group is included (might not be in belief support).
                if planned_group not in groups:
                    groups = [planned_group] + groups
                    probs = [1e-3] + probs

                # Renormalize probabilities for Julia
                z = float(sum(float(x) for x in probs)) if probs else 0.0
                if z > 0:
                    probs = [float(x) / z for x in probs]
                else:
                    probs = [1.0 / len(groups)] * len(groups)

                # Map bin-groups to representative bin-slots (what we actually execute inspect on).
                candidate_bins: list[str] = []
                for g in groups:
                    rep = self._get_bin_group_rep(g)
                    if rep is not None:
                        candidate_bins.append(rep)
                if not candidate_bins:
                    # Last resort: inspect at the planned bin-slot itself.
                    candidate_bins = [planned_bin]

                info_action = self.info.solve_pre_pick(
                    part=part,
                    robot=args[0],
                    candidate_bins=candidate_bins,
                    loc_probs=probs,
                    inspect_cost=self.params.inspect_cost,
                    miss_penalty=self.params.miss_penalty,
                )

                # Enforce candidate-set safety: if Julia proposes inspecting a location outside candidate_bins,
                # clamp to inspecting the planned bin (or continue).
                if info_action != "continue":
                    _nm, _ia = _parse_action_str(info_action)
                    if _nm == "inspect" and len(_ia) >= 2:
                        proposed_loc = _ia[1]
                        if proposed_loc not in candidate_bins:
                            # If it's in the same group, map to the group's representative.
                            pg = self._bin_group(proposed_loc)
                            rep = self._get_bin_group_rep(pg) if pg else None
                            if rep is not None and rep in candidate_bins:
                                info_action = f"inspect(floor, {rep})"
                            else:
                                info_action = f"inspect(floor, {planned_bin})"

                if info_action != "continue":
                    _, iargs = _parse_action_str(info_action)
                    loc = iargs[1]
                    # Anti-loop: if POMDP keeps requesting the same inspect for the same part, skip once.
                    prev = self._last_inspect_loc.get(part)
                    if prev == loc:
                        key = (part, loc)
                        c = int(self._repeat_inspect_count.get(key, 0)) + 1
                        self._repeat_inspect_count[key] = c
                        if c >= 2:
                            # Force progress: treat as "continue" so we can attempt the pick.
                            info_action = "continue"
                    else:
                        self._last_inspect_loc[part] = loc
                        self._repeat_inspect_count[(part, loc)] = 1

                if info_action != "continue":
                    _, iargs = _parse_action_str(info_action)
                    loc = iargs[1]
                    # Ensure inspect is applicable: if we're not at `loc`, first move there.
                    cur_loc = self.env.get_observable_facts().get("robot_loc")
                    if cur_loc is not None and cur_loc != loc:
                        move_act = f"move_floor(floor, {cur_loc}, {loc})"
                        move_res = self.env.apply_action(move_act)
                        events.append({"action": move_act, "ok": move_res.ok, "reason": move_res.reason})
                        if not move_res.ok:
                            self._note_unknown_object(
                                move_res.reason,
                                action_name="move_floor",
                                action_args=["floor", str(cur_loc), str(loc)],
                            )
                            self._replan()
                            return {"ok": False, "events": events}

                    info_res = self.env.apply_action(info_action)
                    events.append({"action": info_action, "ok": info_res.ok, "reason": info_res.reason})
                    if not info_res.ok:
                        self._note_unknown_object(info_res.reason, action_name="inspect", action_args=["floor", str(loc)])

                    obs = None
                    if info_res.ok:
                        obs = self.env.sample_inspect_obs(loc)
                        self._update_beliefs_from_inspect_obs(target_part=part, obs=obs)

                    # IMPORTANT: after any info action, discard the old plan and replan from the updated problem/belief.
                    self.plan = []
                    self._replan()
                    events.append(
                        {
                            "trigger": "pre_pick",
                            "at_action": a,
                            "info_action": info_action,
                            "info_ok": info_res.ok,
                            "info_reason": info_res.reason,
                            "obs": obs,
                        }
                    )
                    continue

            # Per user request: no pre-submit trigger before moving AGV to warehouse.

            # Execute next classical action
            # Pre-check pick actions to avoid sending obviously failing commands (real-robot friendly).
            if aname == "floor_pick_part":
                res = self.env.check_action_applicable(a)
                if res.ok:
                    res = self.env.apply_action(a)
            else:
                res = self.env.apply_action(a)
            if not res.ok:
                # If execution fails due to an unknown object (model mismatch), remember it to avoid looping.
                self._note_unknown_object(res.reason, action_name=aname, action_args=args)
                if aname == "floor_pick_part":
                    part = args[1]
                    bin_tried = args[2]
                    dist = self.belief_loc.get(part)
                    if dist and bin_tried in dist:
                        dist[bin_tried] = 0.0
                        z = sum(dist.values())
                        if z > 0:
                            for k in list(dist.keys()):
                                dist[k] /= z
                    # On pick failure, do an inspect at the attempted bin to refresh beliefs and then replan.
                    inspect_act = f"inspect(floor, {bin_tried})"
                    insp_res = self.env.apply_action(inspect_act)
                    events.append({"action": inspect_act, "ok": insp_res.ok, "reason": insp_res.reason})
                    if insp_res.ok:
                        obs = self.env.sample_inspect_obs(bin_tried)
                        self._update_beliefs_from_inspect_obs(target_part=part, obs=obs)
                        events.append({"trigger": "pick_failed_inspect", "at_action": a, "obs": obs})
                        # Discard old plan and replan based on updated beliefs.
                        self.plan = []
                        self._replan()
                        continue
                self._replan()
                events.append({"action": a, "ok": False, "reason": res.reason, "replanned": True})
                return {"ok": False, "events": events}

            # Success: pop + record
            self.plan.pop(0)
            events.append({"action": a, "ok": True})

            # User request: do NOT use 'after_place' or any submit-time triggers; only 'pre_pick' remains.

            # Stopping condition: submitted?
            if "submitted" in {f.name for f in self.env.problem.fluents}:
                submitted = self.env.problem.fluent("submitted")
                orders = [o for o in self.env.problem.all_objects if o.type.name == "order"]
                if orders and all(
                    self.env.state.get_value(submitted(o)).bool_constant_value() for o in orders
                ):
                    return {"done": True, "reason": "submitted_all", "events": events}

        return {"ok": True, "events": events}


