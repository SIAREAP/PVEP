from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Set
import re

def pred(name: str, *args: Sequence[str]) -> str:
    """Helper to construct a PDDL atom string."""
    args_str = " ".join(map(str, args))
    return f"({name} {args_str})"

def format_pddl_problem(
    problem_name: str,
    domain_name: str,
    objects: Dict[str, Iterable[str]],
    init: Iterable[str],
    goal: Sequence[str],
) -> str:
    """Format a PDDL problem string."""
    pddl_str = [f"(define (problem {problem_name})", f"  (:domain {domain_name})", ""]

    pddl_str.append("  (:objects")
    for obj_type, obj_list in sorted(objects.items()):
        unique_objs = sorted(set(obj_list))
        if unique_objs:
            pddl_str.append(f"    {' '.join(unique_objs)} - {obj_type}")
    pddl_str.append("  )\n")

    pddl_str.append("  (:init")
    for fact in sorted(set(init)):
        pddl_str.append(f"    {fact}")
    pddl_str.append("  )\n")

    pddl_str.append("  (:goal (and")
    for g in goal:
        pddl_str.append(f"    {g}")
    pddl_str.append("  ))")
    pddl_str.append(")")

    return "\n".join(pddl_str) + "\n"

def generate_problem_pddl(
    world_state: Dict,
    order_task: Dict | List[Dict],
    problem_name: str = "ariac_problem",
    domain_name: str = "ariac_kitting",
    *,
    fallback_world_state: Optional[Dict] = None,
    allow_missing: bool = True,
    include_all_bin_slots: bool = False,
) -> str:
    """
    Generate PDDL problem string from world state and order task(s).
    """
    objects: Dict[str, Set[str]] = defaultdict(set)
    init_facts: Set[str] = set()
    goals: List[str] = []
    
    # Normalize order_task to a list
    if isinstance(order_task, dict):
        orders = [order_task]
    else:
        orders = order_task
    
    # Constants and Locations
    ROBOT_NAME = "floor"
    ROBOT_START_LOCATION = "floor_init"
    GRIPPER_STATIONS = ["gripper_station1", "gripper_station2"]
    FORBIDDEN_BIN_IDS = {3, 4, 7, 8}
    
    objects["robot"].add(ROBOT_NAME)
    objects["location"].add(ROBOT_START_LOCATION)
    objects["gripper_station"].update(GRIPPER_STATIONS)
    # NOTE: do NOT auto-add 'trash' into problem objects.
    # Domain may define `trash` as a constant; adding it here can create type mismatches
    # (e.g., 'trash - location' in problem vs 'trash - bin_slot' constant in domain).
    # objects["location"].update(GRIPPER_STATIONS)  <-- Removed to avoid duplicate definition
    
    # Initial Robot State
    init_facts.update({
        pred("at_robot", ROBOT_NAME, ROBOT_START_LOCATION),
        pred("has_part_gripper", ROBOT_NAME),
        pred("gripper_empty", ROBOT_NAME),
    })

    # AGV Destinations
    for agv_idx in range(1, 5):
        home_destination = f"init_agv{agv_idx}"
        objects["agv_destination"].add(home_destination)

    def _bin_id_from_slot(name: str) -> Optional[int]:
        m = re.match(r"^bin(\d+)_\d+$", str(name))
        return int(m.group(1)) if m else None

    # Optionally declare *all* floor-reachable bin slots as objects (36 slots: bin 1,2,5,6).
    # This keeps the oracle/env able to parse and apply planner actions that reference any reachable slot.
    # Note: Duplicates are handled automatically since objects[...] is a Set.
    if include_all_bin_slots:
        floor_reachable_bins = [1, 2, 5, 6]
        for b in floor_reachable_bins:
            for s in range(1, 10):     # _1.._9
                sl = f"bin{b}_{s}"
                objects["bin_slot"].add(sl)

    # --- Process Trays ---
    # world_state['trays'] = [{'name': 'tray_1', 'id': 1, 'location': 'kts1'}, ...]
    for t in world_state.get('trays', []):
        t_name = t['name']
        t_loc = t['location']
        
        objects["tray"].add(t_name)
        objects["tray_slot"].add(t_loc) 
        
        init_facts.add(pred("tray_on_slot", t_name, t_loc))

    # --- Process Parts ---
    # world_state['parts'] = [{'name': 'red_pump_1', 'type':..., 'location': 'bin1_2', ...}]
    for p in world_state.get('parts', []):
        p_name = p['name']
        p_loc = p['location']
        p_flipped = p.get('flipped', False)
        
        objects["part"].add(p_name)
        # Only add a part_on fact if the location is a valid bin_slot name.
        if isinstance(p_loc, str) and _bin_id_from_slot(p_loc) is not None:
            # Ensure this slot is defined in objects (set handles deduplication)
            objects["bin_slot"].add(p_loc)
            
            init_facts.add(pred("part_on", p_name, p_loc))
            
            # If the part is on a forbidden bin (e.g. 3, 4, 7, 8), mark it as forbid_reach.
            # This handles the case where a part is detected outside the standard reachable bins.
            b_id = _bin_id_from_slot(p_loc)
            if b_id in FORBIDDEN_BIN_IDS:
                init_facts.add(pred("floor_forbid_reach", p_loc))
        
        # Default all parts to good as requested
        init_facts.add(pred("good", p_name))
        
        if p_flipped:
            init_facts.add(pred("flipped", p_name))
        
        # No robot_reach facts are emitted.

    # --- Process Orders ---
    # Identify available parts for greedy allocation across ALL orders
    # Need careful allocation if multiple orders need same part type.
    
    available_parts = defaultdict(list)
    for p in world_state.get('parts', []):
        # Key by "color_type" to match order
        key = f"{p['color']}_{p['type']}"
        available_parts[key].append(p['name'])

    fallback_parts = defaultdict(list)
    if fallback_world_state is not None:
        for p in fallback_world_state.get("parts", []):
            key = f"{p['color']}_{p['type']}"
            fallback_parts[key].append(p["name"])

    # Ensure deterministic allocation order
    for k in list(available_parts.keys()):
        available_parts[k] = list(available_parts[k])
    for k in list(fallback_parts.keys()):
        fallback_parts[k] = list(fallback_parts[k])

    missing_counter: Dict[str, int] = defaultdict(int)
        
    # --- Order precedence (priority / submission sequence) ---
    # If the domain defines a `precedes` predicate, we can constrain submission order.
    # Policy:
    # - All priority orders must precede all non-priority orders.
    # - Within the same priority group, preserve the input order (stable) by chaining precedes.
    if len(orders) >= 2:
        prio_ids: List[str] = []
        normal_ids: List[str] = []
        for t in orders:
            oid = str(t.get("order_id"))
            if bool(t.get("priority", False)):
                prio_ids.append(oid)
            else:
                normal_ids.append(oid)

        # Priority orders before normal orders (strong constraint).
        for hi in prio_ids:
            for lo in normal_ids:
                if hi and lo and hi != lo:
                    init_facts.add(pred("precedes", hi, lo))

        # Preserve submission sequence within each group by chaining.
        def _chain_precedes(ids: List[str]) -> None:
            for a, b in zip(ids, ids[1:]):
                if a and b and a != b:
                    init_facts.add(pred("precedes", a, b))

        _chain_precedes(prio_ids)
        _chain_precedes(normal_ids)

    for task in orders:
        o_id = task['order_id']
        objects["order"].add(o_id)
        
        kt = task.get('kitting_task', {})
        agv_num = kt.get('agv_number')
        agv_name = f"agv{agv_num}"
        objects["agv"].add(agv_name)
        
        # AGV Home Configuration
        home_dest = f"init_agv{agv_num}"
        init_facts.add(pred("home", agv_name, home_dest))
        init_facts.add(pred("agv_at", agv_name, home_dest))
        init_facts.add(pred("agv_reach", agv_name, home_dest))
        init_facts.add(pred("agv_reach", agv_name, "warehouse"))
        
        tray_id_req = kt.get('tray_id')
        req_tray_name = f"tray_{tray_id_req}"
        
        # Check if this tray exists in world_state, if not we might need to assume it exists or fail?
        # For now, add it to objects if not present (though it should be detected)
        objects["tray"].add(req_tray_name) 
        
        init_facts.add(pred("order_uses_tray", o_id, req_tray_name, agv_name))
        
        for prod in kt.get('products', []):
            p_type = prod['type']
            p_color = prod['color']
            quadrant = prod['quadrant']
            
            key = f"{p_color}_{p_type}"
            
            # Greedy allocation
            if available_parts[key]:
                part_name = available_parts[key].pop(0)
            else:
                if not allow_missing:
                    # If VLM missed it but REAL has it, borrow a REAL part object name so the plan can proceed.
                    # Do NOT add (part_on ...) for this borrowed part here (we don't "fix" perception),
                    # the controller's belief model will handle localization via inspect.
                    if fallback_parts.get(key):
                        part_name = fallback_parts[key].pop(0)
                        objects["part"].add(part_name)
                        init_facts.add(pred("good", part_name))
                    else:
                        raise ValueError(f"Missing required part type in world_state (and fallback): {key}")
                else:
                    # Legacy: create a placeholder part object (not recommended for execution).
                    idx = int(missing_counter[key])
                    missing_counter[key] = idx + 1
                    part_name = f"{key}_missing_{idx}"
                    objects["part"].add(part_name)
                    init_facts.add(pred("good", part_name))

            slot_name = f"{agv_name}_{quadrant}"
            objects["agv_slot"].add(slot_name)
            
            init_facts.add(pred("slot_of", slot_name, agv_name))
            init_facts.add(pred("slot_empty", slot_name))
            init_facts.add(pred("order_needs_part", o_id, part_name, slot_name))
            # AGV slots are locations; no robot_reach needed.

        goals.append(pred("submitted", o_id))

    # Add action-cost initialization and metric (per 修改点)
    init_facts.add("(= (total-cost) 0)")
    pddl = format_pddl_problem(problem_name, domain_name, objects, init_facts, goals)
    # Insert metric at the end (format_pddl_problem currently doesn't support it)
    if "(:metric" not in pddl:
        pddl = pddl.rstrip()[:-1] + "\n  (:metric minimize (total-cost))\n)\n"
    return pddl
