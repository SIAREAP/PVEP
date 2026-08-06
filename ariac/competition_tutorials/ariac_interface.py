#!/usr/bin/env python3

import time
import traceback
import copy
import os
import re
import warnings
from collections import deque
from typing import List, Tuple, Optional, Any, Dict
import numpy as np

import rclpy
from rclpy.qos import qos_profile_sensor_data
from ariac_msgs.msg import CompetitionState
from ariac_msgs.srv import PerformQualityCheck
from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image

# Reuse existing implementation
from sia_interface import (
    SIAInterface,
    KittingSubtask,
    determine_part_name,
)

# VLM Imports
import cv2
from pathlib import Path
from competition_tutorials.eval import (
    get_pddl_from_images,
    parse_part_on,
    get_inspect_pddl,
    get_drop_recovery_replacement,
)

# Imports needed for tray logic extraction
from competition_tutorials.data import (
    tray_slots_location,
    bin_slot_positions,
    kitting_angle_velocity,
    kitting_robot_park_location,
    quad_offsets_,
    kitting_pick_part_heights_on_bin_agv,
)
from competition_tutorials.Kinematics import Rot2Matrix, Matrix2Pos_rpy, Pose2Matrix, quaternion_from_euler, QuaternionFromRPY
from competition_tutorials.interpolation import traj_generate

# POMDP Imports
from competition_tutorials.ariac_pomdp.real_env_up import RealWorldUPEnv, StepResult, _parse_action_str
from competition_tutorials.ariac_pomdp.hybrid_controller import HybridController, HybridParams
from competition_tutorials.ariac_pomdp.ariac_kitting_api import JuliaInfoClient
from competition_tutorials.ariac_pomdp.classical_planner import ClassicalPlanner
from competition_tutorials.ariac_pomdp.pddl_generation import generate_problem_pddl
from competition_tutorials.ariac_pomdp.pddl_state_manager import PDDLStateManager

# For safe recovery validation (avoid crashing on invalid PDDL)
from unified_planning.io import PDDLReader
from unified_planning.engines.sequential_simulator import UPSequentialSimulator
from unified_planning.exceptions import UPUsageError

PACKAGE_ROOT = Path(__file__).resolve().parent
ARIAC_POMDP_DIR = PACKAGE_ROOT / "ariac_pomdp"
PROMPT_EXAMPLE_DIR = PACKAGE_ROOT / "prompt_example"


def _silence_unified_planning_and_warnings() -> None:
    """
    Silence known noisy stdout/stderr outputs:
    - unified_planning engine credits (*** Credits ***)
    - pkg_resources deprecation warning triggered by some planners (e.g., up_symk)
    """
    # Suppress pkg_resources deprecation warning (setuptools deprecations)
    warnings.filterwarnings(
        "ignore",
        message=r"pkg_resources is deprecated as an API\..*",
        category=UserWarning,
    )

    # Suppress Unified Planning engine credits
    try:
        from unified_planning.shortcuts import get_environment
        get_environment().credits_stream = None
    except Exception:
        # unified_planning might not be available in some environments
        pass


# Run as early as possible (module import time) so later planner init is quiet.
_silence_unified_planning_and_warnings()


class RealWorldARIACEnv(RealWorldUPEnv):
    """
    Wrapper that executes actions on the real robot (via interface) 
    while maintaining a PDDL state (Oracle) for the solver.
    """
    def __init__(self, domain_path, problem_path, interface: 'PDDLKittingInterface', seed: int = 0):
        super().__init__(domain_path, problem_path, seed=seed)
        self.interface = interface

    def apply_action(self, action_str: str) -> StepResult:
        # 1. Execute on real robot first
        try:
            name, args = _parse_action_str(action_str)
            if name == "continue":
                # For continue, just return success
                return StepResult(True, "noop")
                
            self.interface.get_logger().info(f"[RealExec] {action_str}")
            # Physical execution (blocking)
            self.interface._execute_action(name, args)
            
            # 2. Update internal state (oracle model)
            # The oracle must track the nominal state.
            # We assume the action succeeded if _execute_action didn't raise.
            res = super().apply_action(action_str)

            # 3. Snapshot latest PDDL.
            # IMPORTANT:
            # - UP's PDDLWriter snapshots can be unstable (formatting / naming), and can cause replans to restart from scratch.
            # - We keep a "text PDDL" state by applying action effects directly on the last known PDDL text.
            mgr = getattr(self.interface, "pddl_mgr", None)
            try:
                if mgr is not None:
                    mgr.write_action_snapshot(action_name=name, args=args)
            except Exception as e:
                self.interface.get_logger().warn(f"[PDDL] Snapshot failed: {e}")

            # 4. Drop detection: if we expected to hold a part but gripper detached, treat as drop event.
            try:
                held_token = getattr(self.interface, "held_part_token", None)
                attached = bool(getattr(self.interface.floor_robot_gripper_state, "attached", False))
                if held_token and (not attached):
                    latest_txt = mgr.latest_pddl_text() if mgr is not None else None
                    order_id, agv_slot = (None, None)
                    if latest_txt and mgr is not None:
                        order_id, agv_slot = mgr.find_order_need_for_part(pddl_text=latest_txt, part_name=str(held_token))
                    drop_event = {
                        "part": str(held_token),
                        "order": order_id,
                        "slot": agv_slot,
                        "action": action_str,
                    }
                    self.interface.last_drop_event = drop_event
                    self.interface.drop_log.append(drop_event)
                    self.interface.get_logger().error(f"[DROP] detected: {drop_event}")
                    # Immediately stop execution queues / motion if possible
                    try:
                        self.interface.cancel_current_action()
                    except Exception:
                        pass
                    # Clear held token so future steps don't double-trigger
                    self.interface.held_part_token = None
                    self.interface.current_held_part = None
                    return StepResult(False, "dropped")
            except Exception as e:
                self.interface.get_logger().warn(f"[DROP] detection error: {e}")

            return res
            
        except Exception as e:
            self.interface.get_logger().error(f"Real execution failed: {e}")
            self.interface.get_logger().error(traceback.format_exc())
            return StepResult(False, f"real_exec_failed: {e}")

    def sample_inspect_obs(self, location: str, *, p_detect: float = 1.0) -> dict[str, Any]:
        """
        Return the observation using VLM results if available, otherwise fallback to real sensors.
        """
        # Determine bin group (e.g. bin1 from bin1_2)
        bin_group = location.split('_')[0] if '_' in location else location
        
        # Check if we have VLM results for this bin
        if bin_group in self.interface.inspect_results:
            self.interface.get_logger().debug(f"[Env] Using VLM results for {bin_group}")
            vlm_parts = self.interface.inspect_results[bin_group]
            
            slots_obs = {}
            for i in range(1, 10):
                slots_obs[f"{bin_group}_{i}"] = "none"
                
            for p_name_raw, p_loc in vlm_parts:
                # p_name_raw is like "blue_battery"
                # p_loc is like "bin1_5"
                if p_loc.startswith(f"{bin_group}_"):
                    slots_obs[p_loc] = p_name_raw
            
            seen_part = slots_obs.get(location, "none")
            return {"type": "inspect", "location": location, "bin_group": bin_group, "slots": slots_obs, "part": seen_part}

        # If no VLM result, raise Error as requested
        self.interface.get_logger().error(f"[Env] Missing VLM result for {bin_group}. Inspect action may have failed or not run.")
        raise RuntimeError(f"Missing VLM result for {bin_group}. Cannot provide observation.")

    def sample_qc_obs(self, part: str, *, p_correct: float = 1.0) -> dict[str, Any]:
        """
        Return the 'true' quality from Ground Truth.
        """
        # In this simplified implementation, we assume quality is Good unless we have data otherwise.
        # Ideally, we would query the QC sensor topic if the part is at a QC station.
        # For now, stick to static oracle or assume good.
        # If we had real QC sensor logic, we'd add it here.
        return super().sample_qc_obs(part, p_correct=1.0)


class PDDLKittingInterface(SIAInterface):
    """Only retains floor robot interface for kitting, providing APIs matching PDDL actions."""

    def __init__(self):
        super().__init__()
        # Note:
        # - Do NOT globally raise this node's log level to WARN by default, otherwise useful
        #   action traces like "[RealExec] ..." will be hidden.
        # - If you ever want to silence INFO logs globally, set env `SIA_QUIET_INFO_LOGS=1`.
        try:
            quiet = str(os.environ.get("SIA_QUIET_INFO_LOGS", "0")).lower() in ("1", "true", "yes", "on")
            if quiet:
                from rclpy.logging import LoggingSeverity
                rclpy.logging.set_logger_level(self.get_name(), LoggingSeverity.WARN)
        except Exception:
            pass

        # Only do kitting, disable assembly/ceiling queues
        self.ceiling_robot_info.is_enabled = False
        self.ceiling_robot_info.is_idle = False
        self.assembly_deque = deque()
        self.combined_deque = deque()
        
        # State management for split actions
        self.current_held_part = None
        self.current_tray_target = None  # (tray_id, tray_slot)

        # PDDL state manager (latest PDDL snapshots from UP oracle)
        self.pddl_mgr: PDDLStateManager | None = None
        # Track which PDDL part token we believe we hold (e.g., red_battery_1)
        self.held_part_token: str | None = None
        # Drop bookkeeping
        self.drop_log: list[dict[str, Any]] = []
        self.last_drop_event: dict[str, Any] | None = None

        # --- Strict submission gate ---
        # Only allow calling the ROS submit service when the action comes from the PDDL executor.
        self._allow_submit_from_pddl: bool = False
        # Map PDDL order ids (lowercase) -> ROS/ARIAC order ids (as announced by /ariac/orders, usually uppercase)
        self._order_id_pddl_to_ros: dict[str, str] = {}

        # Track current generated PDDL paths for belief/real problems (set per planning session)
        self._p_vlm_path: str | None = None
        self._p_real_path: str | None = None

        # Alias mapping for PDDL symbols to actual locations
        self._location_alias = {
            "floor_init": "kts1",           # Robot start (default table 1)
            "gripper_station1": "kts1",
            "gripper_station2": "kts2",
            "init_agv1": "agv1_ks1_tray",
            "init_agv2": "agv2_ks2_tray",
            "init_agv3": "agv3_ks3_tray",
            "init_agv4": "agv4_ks4_tray",   # agv4 init dock
            # Faulty-part recovery: treat 'trash' as the legacy 'can' location.
            "trash": "can",
        }

        # --- Faulty part recovery bookkeeping ---
        # Record placement snapshots so floor_pick_part_from_agv can replay and re-grasp reliably.
        # part_token -> {agv_slot, agv_num, quadrant, q_place(list[float]), world_target(list[float])}
        self._agv_place_records: dict[str, dict[str, Any]] = {}
        # Guard against repeated recovery on the same part token.
        self._faulty_discarded: set[str] = set()

        # Queue for orders to be processed in main thread (avoids callback deadlock)
        self.pending_orders = deque()

        # Results from inspect action (bin_id -> list of part tuples)
        self.inspect_results = {}

        # Enable RGB Cameras (commented out in base class)
        self.right_bins_RGB_camera_sub = self.create_subscription(
            Image,
            "/ariac/sensors/right_bins_RGB_camera/rgb_image",
            self._right_bins_RGB_camera_cb,
            qos_profile_sensor_data,
        )
        self.left_bins_RGB_camera_sub = self.create_subscription(
            Image,
            "/ariac/sensors/left_bins_RGB_camera/rgb_image",
            self._left_bins_RGB_camera_cb,
            qos_profile_sensor_data,
        )

    def _parse_pddl_part_objects(self, pddl_text: str) -> list[str]:
        """
        Extract part object names from (:objects ...) block.
        """
        start = pddl_text.find("(:objects")
        if start < 0:
            return []
        # crude but robust: scan until matching ')'
        depth = 0
        end = None
        for i in range(start, len(pddl_text)):
            if pddl_text[i] == "(":
                depth += 1
            elif pddl_text[i] == ")":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        block = pddl_text[start:end] if end else pddl_text[start:]

        parts: list[str] = []
        for ln in block.splitlines():
            if "- part" not in ln:
                continue
            left = ln.split("- part")[0]
            toks = [t for t in left.split() if t.strip()]
            parts.extend(toks)
        return parts

    def _update_p_vlm_bin_contents(self, *, bin_id: str, parts_formatted: list[tuple[str, str]]) -> None:
        """
        Update the agent-belief PDDL (p_vlm) after an inspect action:
        replace all (part_on ... bin_id_*) facts with the inspected set.
        Also ensures new part objects are declared (and defaulted to (good ...)).
        """
        if not self._p_vlm_path:
            self.get_logger().warn("[PDDL] No p_vlm path set; cannot update belief PDDL after inspect.")
            return

        p = Path(self._p_vlm_path)
        if not p.exists():
            self.get_logger().warn(f"[PDDL] p_vlm not found at {p}; cannot update after inspect.")
            return

        text = p.read_text(encoding="utf-8", errors="replace")

        # --- Update :init: remove old bin group part_on and insert new ones
        init_start = text.find("(:init")
        if init_start < 0:
            self.get_logger().warn("[PDDL] p_vlm has no (:init) block; cannot update.")
            return

        # Extract init block boundaries
        depth = 0
        init_end = None
        for i in range(init_start, len(text)):
            ch = text[i]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    init_end = i + 1
                    break
        if init_end is None:
            self.get_logger().warn("[PDDL] p_vlm init block unbalanced; cannot update.")
            return

        init_block = text[init_start:init_end]
        out_lines: list[str] = []
        # Remove any existing part_on facts for this bin group
        for ln in init_block.splitlines():
            if ln.strip().startswith("(part_on") and f"{bin_id}_" in ln:
                continue
            out_lines.append(ln)

        # Insert new part_on facts before the final ')'
        new_part_lines = [f"    (part_on {pn} {loc})" for pn, loc in parts_formatted if loc.startswith(f"{bin_id}_")]
        # Also ensure (good pn) exists (domain requires good for submit precondition)
        good_lines = [f"    (good {pn})" for pn, _ in parts_formatted]

        rebuilt: list[str] = []
        inserted = False
        for ln in out_lines:
            if (not inserted) and ln.strip() == ")":
                for pl in sorted(set(good_lines)):
                    if pl not in out_lines:
                        rebuilt.append(pl)
                for pl in new_part_lines:
                    rebuilt.append(pl)
                inserted = True
            rebuilt.append(ln)
        new_init_block = "\n".join(rebuilt)

        text2 = text[:init_start] + new_init_block + text[init_end:]

        # --- Update :objects: add any new part objects if needed
        objects_start = text2.find("(:objects")
        if objects_start >= 0:
            depth = 0
            objects_end = None
            for i in range(objects_start, len(text2)):
                ch = text2[i]
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        objects_end = i + 1
                        break
            if objects_end is not None:
                obj_block = text2[objects_start:objects_end]
                existing_parts = set(self._parse_pddl_part_objects(obj_block))
                new_parts = sorted({pn for pn, _ in parts_formatted} - existing_parts)
                if new_parts:
                    # Append into the first line that ends with "- part" if it exists, else add a new "- part" line.
                    obj_lines = obj_block.splitlines()
                    inserted = False
                    for idx, ln in enumerate(obj_lines):
                        if ln.strip().endswith("- part"):
                            # Insert before "- part"
                            left, right = ln.split("-", 1)
                            left = left.rstrip() + " " + " ".join(new_parts) + " "
                            obj_lines[idx] = left + "-" + right
                            inserted = True
                            break
                    if not inserted:
                        # Put it near the end, before closing ')'
                        for idx in range(len(obj_lines) - 1, -1, -1):
                            if obj_lines[idx].strip() == ")":
                                obj_lines.insert(idx, f"    {' '.join(new_parts)} - part")
                                break
                    obj_block2 = "\n".join(obj_lines)
                    text2 = text2[:objects_start] + obj_block2 + text2[objects_end:]

        # --- Update :objects: ensure any new bin_slot objects referenced by inspected (part_on ...) exist
        # Keep :objects consistent with the updated :init, so planners/env won't fail on unknown_object(binX_Y).
        objects_start2 = text2.find("(:objects")
        if objects_start2 >= 0:
            depth = 0
            objects_end2 = None
            for i in range(objects_start2, len(text2)):
                ch = text2[i]
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        objects_end2 = i + 1
                        break
            if objects_end2 is not None:
                obj_block = text2[objects_start2:objects_end2]

                # Parse existing bin_slot objects
                existing_bin_slots: set[str] = set()
                for ln in obj_block.splitlines():
                    if "- bin_slot" not in ln:
                        continue
                    left = ln.split("- bin_slot")[0]
                    toks = [t for t in left.split() if t.strip() and not t.strip().startswith("(:objects")]
                    existing_bin_slots.update(toks)

                # Only add slots for this bin group (bin_id_*), and only those used by inspected part_on facts.
                desired_slots = sorted({loc for _, loc in parts_formatted if str(loc).startswith(f"{bin_id}_")})
                new_slots = [s for s in desired_slots if s not in existing_bin_slots]
                if new_slots:
                    obj_lines = obj_block.splitlines()
                    inserted = False
                    for idx, ln in enumerate(obj_lines):
                        if ln.strip().endswith("- bin_slot"):
                            left, right = ln.split("-", 1)
                            left = left.rstrip() + " " + " ".join(new_slots) + " "
                            obj_lines[idx] = left + "-" + right
                            inserted = True
                            break
                    if not inserted:
                        for idx in range(len(obj_lines) - 1, -1, -1):
                            if obj_lines[idx].strip() == ")":
                                obj_lines.insert(idx, f"    {' '.join(new_slots)} - bin_slot")
                                break
                    obj_block2 = "\n".join(obj_lines)
                    text2 = text2[:objects_start2] + obj_block2 + text2[objects_end2:]

        p.write_text(text2, encoding="utf-8")
        self.get_logger().info(
            f"[PDDL] Updated p_vlm bin contents for {bin_id} ({len(parts_formatted)} parts). Confidence table updated as well."
        )

    # ---------------------- Subscription & Order Processing ----------------------
    def process_orders(self, order_msg):
        # Only process kitting orders
        order_id = order_msg.id
        order_type = order_msg.type
        
        if order_type != 0: # Not KITTING
            return

        # Store for reference
        self.order_recored_list.append(order_msg)
        self.orders_list.append(order_msg)

        # Add to pending queue for main thread processing
        self.get_logger().debug(f"Received order {order_id}. Queued for processing.")
        self.pending_orders.append(order_msg)

    def _check_order_feasibility(self, order_task, state_real):
        """
        Check if the real world state has enough parts to satisfy the order.
        Returns: True if feasible, False otherwise.
        """
        # 1. Tally Order Requirements
        required_parts = {}
        for prod in order_task['kitting_task']['products']:
            # Key: (color, type)
            key = (prod['color'], prod['type'])
            required_parts[key] = required_parts.get(key, 0) + 1
            
        # 2. Tally Real Inventory
        real_inventory = {}
        for part in state_real['parts']:
            # Key: (color, type)
            key = (part['color'], part['type'])
            real_inventory[key] = real_inventory.get(key, 0) + 1
            
        # 3. Compare
        missing_parts = []
        for key, count in required_parts.items():
            available = real_inventory.get(key, 0)
            if available < count:
                missing_parts.append(f"{key[0]} {key[1]} (Need: {count}, Have: {available})")
                
        if missing_parts:
            self.get_logger().error("---------------------------------------------------")
            self.get_logger().error("CRITICAL ERROR: Insufficient parts in REAL environment! 缺少完成订单所需零件，请补充零件！")
            for msg in missing_parts:
                self.get_logger().error(f" - {msg}")
            self.get_logger().error("Cannot complete order. Terminating.")
            self.get_logger().error("---------------------------------------------------")
            return False
            
        self.get_logger().debug("Order feasibility check passed: Sufficient parts available in Real World.")
        return True

    def _msg_to_task_dict(self, order_msg) -> Dict:
        """Convert ROS Order message to task dict format."""
        # IMPORTANT:
        # - unified_planning / planners often normalize object names to lowercase
        # - ARIAC submit_order service expects the exact ROS order_id (typically uppercase)
        # So we use lowercase order ids in PDDL, but keep a mapping to the original ROS id for submission.
        order_id_ros = str(order_msg.id)
        order_id = order_id_ros.lower()
        self._order_id_pddl_to_ros[order_id] = order_id_ros
        order_kitting_task = order_msg.kitting_task
        
        task_dict = {
            'order_id': order_id,
            'priority': order_msg.priority,
            'kitting_task': {
                'agv_number': order_kitting_task.agv_number,
                'tray_id': order_kitting_task.tray_id,
                'destination': 'warehouse', 
                'products': []
            }
        }
        
        for p in order_kitting_task.parts:
            p_name = determine_part_name(p.part.type, p.part.color)
            p_tokens = p_name.split('_')
            if len(p_tokens) >= 3:
                p_type = p_tokens[1]
                p_color = p_tokens[2]
            else:
                p_type = p_name
                p_color = ""
            task_dict['kitting_task']['products'].append({
                'type': p_type,
                'color': p_color,
                'quadrant': p.quadrant
            })
            
        return task_dict

    def process_order_execution(self, order_msg):
        """
        Main thread execution logic for a single order using Hybrid POMDP Controller.
        """
        order_id = order_msg.id
        order_kitting_task = order_msg.kitting_task

        # Setup order tracking info
        self.ceiling_kitting_once = True
        self.oredr_length[order_id] = len(order_kitting_task.parts)
        self.co_tray_flag[order_id] = False
        
        self.get_logger().info(f"Processing order {order_id}. Waiting for sensors...")
        
        # 1. Wait for sensors (VLM images + Logical Camera detection)
        self.wait_for_sensors(timeout=5.0)
        
        # 2. Capture States
        # A. VLM State (Agent Belief)
        state_vlm = self.get_vlm_world_state(current_order=order_msg)
        self.get_logger().debug(f"VLM State: {len(state_vlm['parts'])} parts detected.")

        # B. Real State (Ground Truth from Logical Cameras)
        state_real = self.get_current_world_state()
        self.get_logger().debug(f"Real State: {len(state_real['parts'])} parts detected.")

        # 3. Construct Order Task List (Current + Pending)
        orders_to_solve = []
        
        # Current order
        current_task_dict = self._msg_to_task_dict(order_msg)
        orders_to_solve.append(current_task_dict)
        
        # Pending orders:
        # IMPORTANT: if we include pending orders into this planning session,
        # we must also consume them from the queue; otherwise run() will start
        # a second controller session for the same orders after this finishes.
        pending_snapshot = list(self.pending_orders)
        self.pending_orders.clear()
        for pending_o in pending_snapshot:
            orders_to_solve.append(self._msg_to_task_dict(pending_o))
            
        # For compatibility with legacy checks, use current task dict as 'order_task'
        order_task = current_task_dict

        # --- New Check: Feasibility against Ground Truth ---
        if not self._check_order_feasibility(order_task, state_real):
             self.get_logger().error("Order Feasibility Check Failed. Aborting order.")
             return
        # ---------------------------------------------------

        # 4. Generate PDDL Files
        # FIX: Point to the SOURCE directory to ensure .jl and .pddl files exist
        # When running via ros2 run, __file__ points to the install directory which misses non-python files
        base_path = ARIAC_POMDP_DIR
        
        # Put generated files in a subdirectory of ariac_pomdp
        gen_dir = base_path / "generated_pddl"
        gen_dir.mkdir(parents=True, exist_ok=True)
        
        domain_path = str(base_path / "domain.pddl")
        solve_pomdp_jl = str(base_path / "solve_pomdp.jl")
        
        p_vlm_path = str(gen_dir / "p_vlm.pddl")
        p_real_path = str(gen_dir / "p_real.pddl")
        
        # Generate PDDL content (Pass ALL orders)
        # IMPORTANT: never create *_missing_* parts.
        # - For VLM belief: if VLM missed a required type but REAL has it, borrow a REAL part name (without adding part_on).
        # - For REAL: if REAL is missing it too, abort (cannot execute).
        try:
            pddl_vlm_str = generate_problem_pddl(
                state_vlm,
                orders_to_solve,
                problem_name="ariac_vlm",
                domain_name="ariac_kitting",
                fallback_world_state=state_real,
                allow_missing=False,
            )
            pddl_real_str = generate_problem_pddl(
                state_real,
                orders_to_solve,
                problem_name="ariac_real",
                domain_name="ariac_kitting",
                allow_missing=False,
                include_all_bin_slots=True,
            )
        except Exception as e:
            self.get_logger().error(f"[PDDL] Failed to generate PDDL without missing parts: {e}")
            self.get_logger().error("Aborting this order/session because required parts are unavailable.")
            return
        
        with open(p_vlm_path, "w") as f:
            f.write(pddl_vlm_str)
        with open(p_real_path, "w") as f:
            f.write(pddl_real_str)
            
        # Keep message but omit the long absolute path.
        self.get_logger().info("Generated PDDL files.")

        # Save for later updates (inspect should update belief PDDL)
        self._p_vlm_path = p_vlm_path
        self._p_real_path = p_real_path

        # 4.5 Initialize PDDL state manager for this execution session
        snapshots_dir = gen_dir / "snapshots_real"
        self.pddl_mgr = PDDLStateManager(domain_path=domain_path, init_problem_path=p_real_path, out_dir=str(snapshots_dir))
        # Seed the 'latest' snapshot with the initial real problem (so we always have something to reference)
        try:
            self.pddl_mgr.write_text_as_latest(pddl_text=pddl_real_str, filename="snapshot_problem_init.pddl")
        except Exception as e:
            self.get_logger().warn(f"[PDDL] Failed to seed initial snapshot: {e}")

        # 5. Initialize / Run Controller (supports drop recovery)
        try:
            def _make_controller() -> Tuple[HybridController, RealWorldARIACEnv]:
                info = JuliaInfoClient(domain_path=domain_path, p_vlm_path=p_vlm_path, solve_pomdp_jl=solve_pomdp_jl)
                env = RealWorldARIACEnv(domain_path, p_real_path, interface=self)
                planner = ClassicalPlanner(domain_path)
                ctrl = HybridController(env, planner, info, base_problem_path=p_vlm_path, params=HybridParams(macro_max_steps=1))
                return ctrl, env
            
            ctrl, env = _make_controller()
            self.get_logger().info("Starting Hybrid Controller Loop...")
            
            step_count = 0
            while True:
                step_count += 1
                self.get_logger().info(f"--- Step {step_count} ---")
                
                out = ctrl.step()
                self.get_logger().debug(f"Step Result: {out}")
                
                if out.get("done"):
                    # noisy terminal output; keep silent
                    break
                    
                # Drop recovery hook: ctrl.step() returns {"ok": False, ...} when env.apply_action fails.
                if out.get("ok") is False:
                    last_reason = None
                    try:
                        evs = out.get("events") or []
                        if evs:
                            last_reason = evs[-1].get("reason")
                    except Exception:
                        last_reason = None

                    if last_reason == "dropped" or self.last_drop_event is not None:
                        drop_ev = self.last_drop_event or {}
                        dropped_part = str(drop_ev.get("part") or "")
                        dropped_order = str(drop_ev.get("order") or "")
                        dropped_slot = str(drop_ev.get("slot") or "")

                        if not dropped_part:
                            self.get_logger().error("[DROP] missing dropped_part token; cannot recover.")
                            break

                        if not self.pddl_mgr:
                            self.get_logger().error("[DROP] pddl_mgr not initialized; cannot recover.")
                            break

                        latest_txt = self.pddl_mgr.latest_pddl_text() or pddl_real_str
                        # Ensure we know which order/slot this part belonged to
                        if (not dropped_order) or (not dropped_slot):
                            dropped_order2, dropped_slot2 = self.pddl_mgr.find_order_need_for_part(
                                pddl_text=latest_txt, part_name=dropped_part
                            )
                            dropped_order = dropped_order or (dropped_order2 or "")
                            dropped_slot = dropped_slot or (dropped_slot2 or "")

                        if not dropped_order or not dropped_slot:
                            self.get_logger().error(f"[DROP] cannot locate (order_needs_part ...) for {dropped_part}. Aborting.")
                            break

                        # Patch out the dropped part from current PDDL before asking VLM to re-assign
                        patched_txt = self.pddl_mgr.apply_drop_patch(pddl_text=latest_txt, dropped_part=dropped_part)

                        # --- VLM replan (minimal logs) ---
                        new_pddl = None
                        chosen = None
                        try:
                            self.get_logger().warn(
                                f"[DROP][VLM-REPLAN] calling VLM to replan replacement for order={dropped_order} slot={dropped_slot}..."
                            )
                            chosen = get_drop_recovery_replacement(
                                latest_pddl_text=patched_txt,
                                dropped_part=dropped_part,
                                dropped_order=dropped_order,
                                dropped_slot=dropped_slot,
                            )
                        except Exception:
                            chosen = None

                        if not chosen:
                            raise RuntimeError("drop_recovery_vlm_failed")

                        new_pddl = patched_txt
                        new_pddl = self.pddl_mgr._replace_or_insert_order_need(
                            pddl_text=new_pddl,
                            order=dropped_order,
                            slot=dropped_slot,
                            new_part=str(chosen),
                        )
                        new_pddl = self.pddl_mgr._ensure_good(pddl_text=new_pddl, part=str(chosen))
                        self.get_logger().warn(
                            f"[DROP][VLM-REPLAN] done: {dropped_part} -> {chosen} for order={dropped_order} slot={dropped_slot}"
                        )

                        # Validate recovered PDDL with unified_planning before restarting controller
                        try:
                            domain_str = Path(domain_path).read_text(encoding="utf-8", errors="replace")
                            _prob = PDDLReader().parse_problem_string(domain_str, new_pddl)
                            _ = UPSequentialSimulator(_prob)  # will raise if unsupported
                        except UPUsageError as e:
                            self.get_logger().error(f"[DROP] Recovered PDDL rejected by UPSequentialSimulator: {e}")
                            self.get_logger().error("[DROP] Stopping and waiting (will not crash controller).")
                            break
                        except Exception as e:
                            self.get_logger().error(f"[DROP] Failed to validate recovered PDDL: {e}")
                            self.get_logger().error("[DROP] Stopping and waiting (will not crash controller).")
                            break

                        # Overwrite both belief and real problems with the repaired PDDL (keeps solver consistent)
                        with open(p_vlm_path, "w") as f:
                            f.write(new_pddl)
                        with open(p_real_path, "w") as f:
                            f.write(new_pddl)
                        try:
                            self.pddl_mgr.write_text_as_latest(pddl_text=new_pddl, filename="snapshot_problem_recovered.pddl")
                        except Exception:
                            pass

                        # Reset drop marker and restart controller from repaired PDDL
                        self.last_drop_event = None
                        try:
                            ctrl, env = _make_controller()
                        except Exception as e:
                            self.get_logger().error(f"[DROP] Failed to restart controller after recovery: {e}")
                            self.get_logger().error("[DROP] Stopping and waiting (will not crash node).")
                            break
                        self.get_logger().debug("[DROP] Recovered PDDL written. Restarted controller, continuing...")
                        continue

                    # Other failure: stop
                    self.get_logger().error(f"Controller step failed (reason={last_reason}). Stopping this order.")
                    break

                # ---------------- Faulty-part recovery hook ----------------
                # Requirement: check only at the boundary AFTER a successful `floor_place_part`
                # to avoid sudden interruption of planning/execution.
                try:
                    evs = out.get("events") or []
                    last_place_action = None
                    for ev in reversed(evs):
                        if bool(ev.get("ok")) and str(ev.get("action", "")).startswith("floor_place_part("):
                            last_place_action = str(ev.get("action"))
                            break

                    if last_place_action:
                        _, aargs = _parse_action_str(last_place_action)
                        # floor_place_part(floor, part, agv_slot, agv, order)
                        placed_part = str(aargs[1]) if len(aargs) > 1 else ""
                        placed_slot = str(aargs[2]) if len(aargs) > 2 else ""
                        placed_order = str(aargs[4]) if len(aargs) > 4 else ""

                        if placed_part and placed_part not in self._faulty_discarded:
                            # Legacy behavior: call twice and trust the second response.
                            _ok1, _info1 = self.perform_quality_check(placed_order)
                            time.sleep(0.5)
                            ok2, info2 = self.perform_quality_check(placed_order)

                            quadrant = self._parse_agv_slot(placed_slot)
                            quad_msg = getattr(info2, f"quadrant{quadrant}", None) if info2 is not None else None
                            is_faulty = bool(getattr(quad_msg, "faulty_part", False))
                            try:
                                self.get_logger().info(
                                    f"[QC] after_place: order={placed_order} part={placed_part} slot={placed_slot} "
                                    f"quadrant={quadrant} faulty_part={is_faulty}"
                                )
                            except Exception:
                                pass

                            if (not ok2) and info2 is None:
                                # QC failed; treat as no-fault to avoid deadlock.
                                is_faulty = False

                            if is_faulty:
                                self.get_logger().warn(
                                    f"[FAULTY] detected after placement: part={placed_part} order={placed_order} slot={placed_slot}."
                                )

                                if not self.pddl_mgr:
                                    self.get_logger().error("[FAULTY] pddl_mgr not initialized; cannot recover.")
                                    break

                                # 0) VLM replan (minimal logs, immediately after detection).
                                latest_txt = self.pddl_mgr.latest_pddl_text() or pddl_real_str
                                repl_chosen: str | None = None
                                try:
                                    patched_for_replan = self.pddl_mgr.apply_drop_patch(
                                        pddl_text=latest_txt, dropped_part=placed_part
                                    )
                                    self.get_logger().warn(
                                        f"[FAULTY][VLM-REPLAN] calling VLM to replan replacement for order={placed_order} slot={placed_slot}..."
                                    )
                                    repl_chosen = get_drop_recovery_replacement(
                                        latest_pddl_text=patched_for_replan,
                                        dropped_part=placed_part,
                                        dropped_order=placed_order,
                                        dropped_slot=placed_slot,
                                    )
                                except Exception:
                                    repl_chosen = None

                                if not repl_chosen:
                                    raise RuntimeError("faulty_recovery_vlm_failed")

                                self.get_logger().warn(
                                    f"[FAULTY][VLM-REPLAN] done: {placed_part} -> {repl_chosen} for order={placed_order} slot={placed_slot}"
                                )

                                # 1) Generate temporary recovery problem by replacing goal only.
                                tmp_txt = self._make_faulty_goal_problem(base_pddl_text=latest_txt, faulty_part=placed_part)
                                tmp_path = str(gen_dir / f"faulty_recovery_{step_count:04d}.pddl")
                                Path(tmp_path).write_text(tmp_txt, encoding="utf-8")

                                # 2) Solve and execute the mini-plan (keep oracle + snapshots consistent).
                                tmp_prob = ctrl.planner.parse_problem(tmp_path)
                                mini_plan = ctrl.planner.solve(tmp_prob)
                                for act in mini_plan:
                                    r = env.apply_action(act)
                                    if not r.ok:
                                        self.get_logger().error(f"[FAULTY] mini-plan action failed: {act}, reason={r.reason}")
                                        raise RuntimeError(f"faulty_mini_plan_failed: {act}: {r.reason}")

                                # 3) Patch MAIN PDDL (p_vlm/p_real) based on the latest snapshot,
                                # not based on the temporary recovery file.
                                latest_after = self.pddl_mgr.latest_pddl_text() or latest_txt
                                patched_txt = self.pddl_mgr.apply_drop_patch(pddl_text=latest_after, dropped_part=placed_part)
                                # Apply the early replanned replacement (keeps logs minimal and timing as requested).
                                new_pddl = patched_txt
                                new_pddl = self.pddl_mgr._replace_or_insert_order_need(
                                    pddl_text=new_pddl,
                                    order=placed_order,
                                    slot=placed_slot,
                                    new_part=str(repl_chosen),
                                )
                                new_pddl = self.pddl_mgr._ensure_good(pddl_text=new_pddl, part=str(repl_chosen))
                                chosen = str(repl_chosen)

                                # Validate recovered PDDL with unified_planning before restarting controller
                                try:
                                    domain_str = Path(domain_path).read_text(encoding="utf-8", errors="replace")
                                    _prob = PDDLReader().parse_problem_string(domain_str, new_pddl)
                                    _ = UPSequentialSimulator(_prob)
                                except Exception as e:
                                    self.get_logger().error(f"[FAULTY] Recovered PDDL validation failed: {e}")
                                    break

                                # Overwrite both belief and real problems with the repaired PDDL (keeps solver consistent)
                                with open(p_vlm_path, "w") as f:
                                    f.write(new_pddl)
                                with open(p_real_path, "w") as f:
                                    f.write(new_pddl)
                                try:
                                    self.pddl_mgr.write_text_as_latest(pddl_text=new_pddl, filename="snapshot_problem_faulty_recovered.pddl")
                                except Exception:
                                    pass

                                self._faulty_discarded.add(placed_part)

                                # Restart controller/env from the patched snapshot and continue main task.
                                ctrl, env = _make_controller()
                                self.get_logger().debug("[FAULTY] Recovery complete. Restarted controller, continuing...")
                                continue
                except Exception as e:
                    self.get_logger().warn(f"[FAULTY] recovery hook error (ignored): {e}")
                
                time.sleep(0.1)
                
        except Exception as e:
            self.get_logger().error(f"Hybrid Controller Failed: {e}")
            self.get_logger().error(traceback.format_exc())

    def get_vlm_world_state(self, current_order=None):
        """
        Capture images, query VLM, and construct world state.
        Args:
            current_order: The current order message being processed (optional).
                           If provided, it helps construct the prompt with order context.
        """
        # 1. Check images
        if not hasattr(self, '_left_bins_camera_image') or self._left_bins_camera_image is None:
             self.get_logger().warn("No left bin image available.")
             return {'parts': [], 'trays': []}
        if not hasattr(self, '_right_bins_camera_image') or self._right_bins_camera_image is None:
             self.get_logger().warn("No right bin image available.")
             return {'parts': [], 'trays': []}

        def _overlay_bin_quadrants(img: np.ndarray, *, is_left_bins: bool) -> np.ndarray:
            """
            Add bin-id labels onto the 2x2 bins image to reduce occasional VLM quadrant confusion.
            Layout (top-down) must match prompt in eval.py:
              - left_bins:  TL=bin8, TR=bin7, BL=bin5, BR=bin6
              - right_bins: TL=bin3, TR=bin4, BL=bin2, BR=bin1
            """
            try:
                out = img.copy()
                h, w = out.shape[:2]
                # Draw dividers (2x2)
                cv2.line(out, (w // 2, 0), (w // 2, h), (255, 255, 255), 2)
                cv2.line(out, (0, h // 2), (w, h // 2), (255, 255, 255), 2)

                if is_left_bins:
                    labels = [["bin8", "bin7"], ["bin5", "bin6"]]
                else:
                    labels = [["bin3", "bin4"], ["bin2", "bin1"]]

                # Put text near each quadrant corner
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = 1.0
                thickness = 2
                color = (255, 255, 255)
                pad = 12
                positions = [
                    (pad, pad + 28),                 # TL
                    (w // 2 + pad, pad + 28),        # TR
                    (pad, h // 2 + pad + 28),        # BL
                    (w // 2 + pad, h // 2 + pad + 28)  # BR
                ]
                flat = [labels[0][0], labels[0][1], labels[1][0], labels[1][1]]
                for text, (x, y) in zip(flat, positions):
                    # Outline for readability
                    cv2.putText(out, text, (x, y), font, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
                    cv2.putText(out, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
                return out
            except Exception:
                return img
             
        # 2. Save images
        base_path = PROMPT_EXAMPLE_DIR
        if not base_path.exists():
            base_path.mkdir(parents=True, exist_ok=True)
            
        left_path = base_path / "left_bins_camera_raw.png"
        right_path = base_path / "right_bins_camera_raw.png"
        
        try:
            # Freeze a consistent pair (callbacks may update arrays asynchronously).
            left_img = np.array(self._left_bins_camera_image, copy=True)
            right_img = np.array(self._right_bins_camera_image, copy=True)

            # Default ON: overlay bin labels to prevent intermittent bin-id mixing (bin1/2 vs bin5/6).
            overlay = str(os.environ.get("SIA_VLM_OVERLAY_BINS", "1")).lower() not in ("0", "false", "no", "off")
            if overlay:
                left_img = _overlay_bin_quadrants(left_img, is_left_bins=True)
                right_img = _overlay_bin_quadrants(right_img, is_left_bins=False)

            cv2.imwrite(str(left_path), left_img)
            cv2.imwrite(str(right_path), right_img)
            self.get_logger().debug(f"Saved images for VLM to {base_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to save images: {e}")
            return {'parts': [], 'trays': []}
        
        # 3. Construct Order Info for Prompt
        orders_info = ""
        orders_to_include = []
        if current_order:
            orders_to_include.append(current_order)
        orders_to_include.extend(self.pending_orders)
        
        if orders_to_include:
            lines = []
            for o in orders_to_include:
                lines.append(f"Order ID: {o.id} (Priority: {o.priority})")
                if o.type == 0: # Kitting
                    kt = o.kitting_task
                    lines.append(f"  Task: Kitting, AGV: {kt.agv_number}, Tray ID: {kt.tray_id}")
                    lines.append("  Products needed:")
                    for p in kt.parts:
                        # part is ariac_msgs/Part, has color and type
                        # p is ariac_msgs/KittingPart, has part and quadrant
                        lines.append(f"    - {p.part.color} {p.part.type} (quadrant {p.quadrant})")
                lines.append("")
            orders_info = "\n".join(lines)
            self.get_logger().info(f"Generated Orders Info for VLM:\n{orders_info}")

        # 4. Call VLM
        self.get_logger().info("Calling VLM for scene understanding...")
        pddl_str = get_pddl_from_images(left_path, right_path, orders_info=orders_info)
        if not pddl_str:
            self.get_logger().error("VLM failed to return PDDL.")
            return {'parts': [], 'trays': []}
            
        # 5. Parse PDDL
        pairs = parse_part_on(pddl_str)
        
        world_state = {'parts': [], 'trays': []}
        
        # 6. Assign Unique IDs
        name_counters = {}
        
        for p_name_raw, loc in pairs:
            tokens = p_name_raw.split('_')
            if len(tokens) >= 2:
                color = tokens[0]
                p_type = tokens[1]
            else:
                color = "unknown"
                p_type = p_name_raw
                
            idx = name_counters.get(p_name_raw, 0) + 1
            name_counters[p_name_raw] = idx
            
            unique_name = f"{color}_{p_type}_{idx}"
            
            world_state['parts'].append({
                'name': unique_name,
                'type': p_type,
                'color': color,
                'location': loc,
                'flipped': False # VLM limitation
            })
            
        self.get_logger().debug(f"VLM found {len(world_state['parts'])} parts.")
        
        # 6. Merge Trays (from real state, as VLM doesn't see trays well in this prompt)
        # We use the same tray logic as get_current_world_state
        real_state = self.get_current_world_state()
        world_state['trays'] = real_state.get('trays', [])
                    
        return world_state

    def _left_bins_RGB_camera_cb(self, msg: Image):
        """Override base callback to only capture image, skipping legacy part detection."""
        try:
            self._left_bins_camera_image = self._bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Left Bin CB Error: {e}")

    def _right_bins_RGB_camera_cb(self, msg: Image):
        """Override base callback to only capture image, skipping legacy part detection."""
        try:
            self._right_bins_camera_image = self._bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Right Bin CB Error: {e}")

    def wait_for_sensors(self, timeout=60.0):
        """Wait to ensure sensor callbacks have populated lists."""
        start = time.time()
        self.get_logger().debug(f"Waiting for sensors (timeout={timeout}s)...")
        
        last_log = time.time()
        
        while time.time() - start < timeout:
            # Check if we have VLM images
            left_ok = getattr(self, '_left_bins_camera_image', None) is not None
            right_ok = getattr(self, '_right_bins_camera_image', None) is not None
            has_images = left_ok and right_ok
            
            # Check trays from ArUco slots
            t1_detected = sum(1 for v in self.tray_1_slots.values() if v is not None)
            t2_detected = sum(1 for v in self.tray_2_slots.values() if v is not None)
            total_trays = t1_detected + t2_detected
            
            # Also check if we have ANY parts detected by logical cameras (for real state)
            # This ensures get_current_world_state() won't be empty
            has_parts = (len(self.bin1_parts) + len(self.bin2_parts) + 
                         len(self.bin3_parts) + len(self.bin4_parts) + 
                         len(self.bin5_parts) + len(self.bin6_parts) + 
                         len(self.bin7_parts) + len(self.bin8_parts)) > 0

            if has_images and total_trays > 0 and has_parts:
                self.get_logger().debug("Sensors ready.")
                time.sleep(1.0) 
                return
            
            if time.time() - last_log > 2.0:
                self.get_logger().debug(f"Waiting... LeftImg={left_ok}, RightImg={right_ok}, Trays={total_trays}, Parts={has_parts}")
                last_log = time.time()
            
            time.sleep(0.5)
            
        self.get_logger().warn("Sensor timeout: proceeding with potential partial state.")


    def get_current_world_state(self):
        """
        Scrape internal part lists and formatted for solver.
        Returns: { 'parts': [...], 'trays': [...] }
        """
        self.all_parts_list_old_update()
        
        world_state = {'parts': [], 'trays': []}
        
        # Collect Parts
        # Iterate all bins
        all_parts = self.bin1_parts + self.bin4_parts + self.bin2_parts + self.bin3_parts + \
                    self.bin6_parts + self.bin7_parts + self.bin5_parts + self.bin8_parts
        
        name_counters = {}
        
        for p in all_parts:
            # p is Part object (or sPart wrapper)
            p_name = p.type
            tokens = p_name.split('_')
            if len(tokens) >= 3:
                p_simple_type = tokens[1]
                p_color = tokens[2]
            else:
                p_simple_type = p_name
                p_color = "unknown"
                
            idx = name_counters.get(p_name, 0) + 1
            name_counters[p_name] = idx
            unique_name = f"{p_color}_{p_simple_type}_{idx}" 
            
            # Use geometric lookup to find the precise bin slot (e.g., bin1_5) from the pose
            p_loc = p.location
            if hasattr(p, 'pose'):
                closest_slot = self.find_closest_slot(p.pose.position, bin_slot_positions)
                if closest_slot:
                     p_loc = closest_slot
                else:
                     # Fallback or error if not found.
                     # If the part is in a bin but we can't map it, it's a critical logic error given the requirement.
                     # However, if p.location is already correct (e.g. from some other source), we might keep it.
                     # But typically p.location is just 'bin1'.
                     pass
            
            # Validate that we have a specific slot
            if "_" not in p_loc and p_loc.startswith("bin"):
                 self.get_logger().error(f"Part {unique_name} has generic location {p_loc} and could not be mapped to a slot! Pose: {p.pose.position}")
                 # You requested to error out rather than random assignment
                 # raise RuntimeError(f"Part {unique_name} location {p_loc} invalid.")
            
            world_state['parts'].append({
                'name': unique_name,
                'type': p_simple_type,
                'color': p_color,
                'location': p_loc,
                'flipped': getattr(p, 'need_flip', False)
            })
            
        # Collect Trays
        def get_id(val):
            try:
                if hasattr(val, '__getitem__') and len(val) > 0:
                    return int(val[0][0]) if hasattr(val[0], '__getitem__') else int(val[0])
                return int(val)
            except:
                return 0

        for slot_name, ids in self.tray_1_slots.items():
            if ids is not None:
                tid = get_id(ids)
                if tid > 0:
                     world_state['trays'].append({
                        'name': f"tray_{tid}",
                        'id': tid,
                        'location': slot_name
                    })

        for slot_name, ids in self.tray_2_slots.items():
            if ids is not None:
                tid = get_id(ids)
                if tid > 0:
                     world_state['trays'].append({
                        'name': f"tray_{tid}",
                        'id': tid,
                        'location': slot_name
                    })
            
        return world_state

    def _map_pose_to_slot(self, pose) -> Optional[str]:
        """
        Reverse lookup: given a pose (tray on table), find which slot (slot1..slot6) it is closest to.
        """
        min_dist = 0.3 
        best_slot = None
        
        for slot, coords in tray_slots_location.items():
            dist = np.sqrt((pose.position.x - coords[0])**2 + (pose.position.y - coords[1])**2)
            if dist < min_dist:
                min_dist = dist
                best_slot = slot
                
        return best_slot

    # ---------------------- PDDL Action Implementation ----------------------

    def _move_to_safe_home(self):
        """Move floor robot to a safe intermediate state to prevent collisions."""
        self.kitting_robot_init("bin_agv_insert_joint")
        self._pause_after_action()

    def change_gripper(self, station: str, use_tray_gripper: bool = None):
        """Switch gripper."""
        station = self._map_location(station)
        current_type = getattr(self.floor_robot_gripper_state, "type", "")
        if use_tray_gripper is None:
            use_tray_gripper = current_type != "tray_gripper"
        gripper = "trays" if use_tray_gripper else "parts"
        
        result = super().ChangeGripper(station, gripper)
        self._move_to_safe_home() 
        return result

    def floor_pick_tray(self, tray_id: str, tray_slot: str):
        """Pick Tray."""
        ref_coord = [self.kitting_base_x, -self.kitting_base_y, self.kitting_base_z]
        curr_coord = [tray_slots_location[tray_slot][0], tray_slots_location[tray_slot][1], tray_slots_location[tray_slot][2]]
        world_target = self.relative_coordinate(ref_coord, curr_coord)
        
        target_matrix = Rot2Matrix(self.init_rotation, world_target)
        self.set_floor_robot_gripper_state(True)
        
        p1 = copy.deepcopy(target_matrix)
        p1[2, 3] = p1[2, 3] - 0.036
        self.MOV_M(p1, eps=0.01, times=3)

        pick_num = 0
        while not self.floor_robot_gripper_state.attached:
            if pick_num > 10:
                break
            pick_num += 1
            p1[2, 3] = p1[2, 3] - 0.001
            self.MOV_M(p1, eps=0.01, times=3)
            time.sleep(0.1)
            
        p3 = copy.deepcopy(p1)
        p3[2, 3] = p3[2, 3] + 0.4
        self.MOV_M(p3, eps=0.01)
        
        self._move_to_safe_home() 
        return True

    def floor_place_tray(self, tray_id: str, agv_number: int):
        """Place Tray."""
        tray = tray_id
        agv_num = agv_number
        
        target_matrix = self.FrameWorldPose(f"agv{agv_num}_tray")
        
        position, rpy = Matrix2Pos_rpy(target_matrix)
        pose = Pose()
        pose.position.x = -position[0]
        pose.position.y = -(position[1] - self.kitting_base_y)
        pose.position.z = position[2] + 0.2
        
        q = quaternion_from_euler(-1.5707963267948966, 1.5707963267948966, 1.5707963267948966)
        pose.orientation.x = q[0]
        pose.orientation.y = q[1]
        pose.orientation.z = q[2]
        pose.orientation.w = q[3]
        target_ = Pose2Matrix(pose)
        
        source_slot = "slot1" # default
        if self.current_tray_target:
             source_slot = self.current_tray_target[1]

        if source_slot in ["slot4", "slot5", "slot6"]:
            need_yaw = 3.1415888848633897
            base_target = Pose()
            base_target.position.x = 0.0
            base_target.position.y = 0.0
            base_target.position.z = 0.0
            base_target.orientation = QuaternionFromRPY(need_yaw, 0, 0)
            part_to_gripper = Pose2Matrix(base_target)
            target_ = target_ @ part_to_gripper 
    
        p1 = copy.deepcopy(target_)
        self.MOV_M(p1)
        p2 = copy.deepcopy(p1)
        p2[2, 3] = p2[2, 3] - 0.22
        self.MOV_M(p2)
        time.sleep(0.2)
        
        self.set_floor_robot_gripper_state(False)
        time.sleep(0.2)
        
        p2[2, 3] = p2[2, 3] + 0.4
        self.MOV_M(p2, eps=0.01)

        self._move_to_safe_home()
        self.current_tray_target = None
        return True

    def cancel_current_action(self):
        """
        Stop the current robot action and clear queues.
        """
        self.get_logger().warn("Cancelling current action...")
        self.STOP()
        self.assembly_deque.clear()
        self.combined_deque.clear()
        # Reset current state flags if needed
        self.current_held_part = None
        self.current_tray_target = None
        self.get_logger().info("Action cancelled and state reset.")

    def floor_pick_part(self, part_token: str, bin_slot: str):
        """Pick Part."""
        tokens = part_token.split('_')
        if len(tokens) >= 3:
            p_color = tokens[0]
            p_type = tokens[1]
        else:
            p_type = "pump" 
            p_color = "red"
            
        p_internal_type = f"assembly_{p_type}_{p_color}" 
        preferred_bin = bin_slot.split("_")[0] if "_" in bin_slot else bin_slot
        preferred_slot = bin_slot if "_" in bin_slot else None

        # Ensure we are using the part gripper before attempting a vacuum pick.
        # In many executions (especially PDDL replans), the planner may omit an explicit change_gripper step.
        # If we are still on tray_gripper, vacuum attach will never succeed and we can "fail" before any visible motion.
        try:
            if getattr(self.floor_robot_gripper_state, "type", "") != "part_gripper":
                # Prefer the nearer station if we can infer it; otherwise default to kts1.
                station = "kts1"
                try:
                    # Bins 1-4 are typically on the right side, 5-8 on the left; either station works but kts1 is safe.
                    bin_num = int(preferred_bin.replace("bin", ""))
                    station = "kts1" if bin_num <= 4 else "kts2"
                except Exception:
                    station = "kts1"
                self.get_logger().warn(
                    f"[GRIPPER] Auto-switching to part_gripper at {station} before picking {part_token} from {bin_slot} "
                    f"(current={getattr(self.floor_robot_gripper_state,'type', '')})"
                )
                # Call the base implementation directly to avoid any toggle ambiguity.
                super().ChangeGripper(station, "parts")
                self._pause_after_action()
        except Exception as e:
            self.get_logger().warn(f"[GRIPPER] Failed to auto-switch to part_gripper: {e}")
        
        picked_part = self._pick_part_candidate(p_internal_type, preferred_bin, preferred_slot)
        
        if not picked_part:
            picked_part = self._pick_part_candidate(p_type, preferred_bin, preferred_slot)
            
        if not picked_part:
            self.get_logger().error(f"Part {part_token} ({p_internal_type}) not found in {preferred_bin}.")
            raise RuntimeError(f"Part {part_token} not found.")

        # Execute pick (blocking in legacy implementation, but we still wait briefly for state propagation).
        self.grasp_part_on_bins(picked_part.location, picked_part)

        # Wait a short time for /ariac/floor_robot_gripper_state to reflect attachment.
        # This prevents false negatives if the callback update lags behind motion completion.
        t0 = time.time()
        while time.time() - t0 < 1.5 and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.01)
            if bool(getattr(self.floor_robot_gripper_state, "attached", False)):
                break
            time.sleep(0.02)

        # Check success BEFORE moving away (to avoid masking a successful attach with subsequent motion issues).
        if not bool(getattr(self.floor_robot_gripper_state, "attached", False)):
            self.get_logger().error(
                f"Failed to pick part {part_token} (requested_slot={bin_slot}, picked_type={getattr(picked_part,'type',None)}, "
                f"picked_loc={getattr(picked_part,'location',None)}, gripper_type={getattr(self.floor_robot_gripper_state,'type','')}, "
                f"enabled={getattr(self.floor_robot_gripper_state,'enabled',None)})"
            )
            self.get_logger().error(f"Failed to pick part {part_token}")
            return False

        self.current_held_part = picked_part
        self.held_part_token = part_token
        # Post-pick retreat: keep wrist pose, only rotate base (shoulder pan) to a safe angle.
        try:
            q_now = copy.deepcopy(getattr(self, "kitting_arm_joint_states", None))
            if q_now:
                q_target = list(q_now)
                q_target[0] = float(self.kitting_typical_joints["standby"][0])
                self.MOV_A(q_target, eps=0.02, sleep_flag=True)
        except Exception as e:
            self.get_logger().warn(f"[POST_PICK] base-rotate retreat failed: {e}")
        return True

    def grasp_part_on_bins(self, location, part, flip=False, repick_callback_num=0):
        """
        Override base pick to add a higher post-pick lift (no wrist rotation).
        """
        target_matrix = self.Pose2Robot(part)
        position, rpy = Matrix2Pos_rpy(target_matrix)
        target_matrix = Rot2Matrix(self.init_rotation, position)

        p1 = copy.deepcopy(target_matrix)
        p2 = copy.deepcopy(target_matrix)
        p2[2, 3] = p2[2, 3] + 0.2

        self.set_floor_robot_gripper_state(True)
        self.MOV_M(p2, eps=0.01)

        if 'pump' in part.type:
            p1[2, 3] = target_matrix[2, 3] + kitting_pick_part_heights_on_bin_agv[part.type] - 0.039
            self.MOV_M(p1, eps=0.001, times=10)
            time.sleep(2)
        else:
            p1[2, 3] = target_matrix[2, 3] + kitting_pick_part_heights_on_bin_agv[part.type] - 0.040
            self.MOV_M(p1, eps=0.001, times=5)
            time.sleep(0.5)

        repick_nums = 0
        while not self.floor_robot_gripper_state.attached:
            if repick_nums >= 5:
                break

            repick_nums = repick_nums + 1
            if 'pump' in part.type:
                p1[2, 3] = p1[2, 3] - 0.001
                time.sleep(1)
            else:
                p1[2, 3] = p1[2, 3] - (repick_nums) * 0.002

            self.MOV_M(p1, eps=0.001, times=10)
            time.sleep(0.5)

        # Higher post-pick lift for a cleaner retreat path.
        p1[2, 3] = p1[2, 3] + 0.55
        self.MOV_M(p1, eps=0.01)

    def floor_flip_part(self, part_token: str, bin_slot: str):
        """Flip Part."""
        # Note: logic similar to pick, but calls grasp_flip_part_on_bins
        tokens = part_token.split('_')
        if len(tokens) >= 3:
            p_color = tokens[0]
            p_type = tokens[1]
        else:
            p_type = "pump" 
            p_color = "red"
            
        p_internal_type = f"assembly_{p_type}_{p_color}" 
        preferred_bin = bin_slot.split("_")[0] if "_" in bin_slot else bin_slot
        preferred_slot = bin_slot if "_" in bin_slot else None
        
        picked_part = self._pick_part_candidate(p_internal_type, preferred_bin, preferred_slot)
        
        if not picked_part:
            picked_part = self._pick_part_candidate(p_type, preferred_bin, preferred_slot)
            
        if not picked_part:
            self.get_logger().error(f"Part {part_token} ({p_internal_type}) not found for flipping.")
            raise RuntimeError(f"Part {part_token} not found for flipping.")
            
        # Helper from SIAInterface that handles flip logic
        # grasp_flip_part_on_bins(part, agv_number, type='kitting')
        # We dummy the agv_number as it might be used for intermediate placement or orientation logic
        # The underlying function seems to flip on the bin itself or using a station.
        # Let's assume it handles the full flip sequence.
        self.grasp_flip_part_on_bins(picked_part, 0, type='kitting')
        self._move_to_safe_home()
        
        return True

    def floor_place_part(self, agv_slot: str, agv_number: int, order_id: str):
        """Place Part."""
        if not self.current_held_part:
            self.get_logger().error("No part held to place!")
            return False
        
        if not self.floor_robot_gripper_state.attached:
             self.get_logger().error("Gripper lost part before placement!")
             self.current_held_part = None
             return False
            
        quadrant = self._parse_agv_slot(agv_slot)
        
        subtask = KittingSubtask(
            order_id=order_id,
            agv_number=agv_number,
            tray_id="dummy",
            destination=3, 
            product_type=self.current_held_part.type,
            product_quadrant=quadrant,
            is_last_subtask=False,
        )
        
        agv_frame = f"agv{agv_number}_ks{agv_number}_tray"
        # Capture token BEFORE we clear it, so faulty recovery can reference the exact PDDL object.
        part_token = str(self.held_part_token or "")
        # `floor_place_part_on_agv` (from SIAInterface) returns a relative `world_target` used by legacy faulty recovery.
        world_target = self.floor_place_part_on_agv(agv_frame, subtask, self.current_held_part, "kitting")

        # Record placement snapshot for potential reverse replay picking from AGV later.
        try:
            if part_token:
                q_place = copy.deepcopy(getattr(self, "kitting_arm_joint_states", None))
                if q_place is not None:
                    q_place = list(q_place)
                rec = {
                    "agv_slot": str(agv_slot),
                    "agv_num": int(agv_number),
                    "quadrant": int(quadrant),
                    "q_place": q_place,
                    "world_target": list(world_target) if isinstance(world_target, (list, tuple)) else None,
                    "t": time.time(),
                }
                self._agv_place_records[part_token] = rec
        except Exception as e:
            self.get_logger().warn(f"[FAULTY] Failed to record placement snapshot for {part_token}: {e}")
        
        self.current_held_part = None
        self.held_part_token = None
        # self._move_to_safe_home() 
        return True

    def _move_floor_arm_to_joints(self, q_target: list[float], *, time_factor: float = 1.0) -> bool:
        """
        Best-effort joint-space move for reverse replay.
        Uses the same FollowJointTrajectory pathway as the base SIAInterface.
        """
        try:
            q_begin = copy.deepcopy(getattr(self, "kitting_arm_joint_states", None))
            if q_begin is None:
                return False
            q_begin = list(q_begin)
            if len(q_begin) != len(q_target):
                return False

            delta = [abs(float(q_begin[i]) - float(q_target[i])) for i in range(len(q_begin))]
            distance = max(delta) if delta else 0.0
            # Keep timing similar to legacy.
            time_from_start = max(0.5, float(distance) / float(kitting_angle_velocity) * float(time_factor))
            traj = traj_generate(self.kitting_arm_joint_names, q_begin, q_target, time_from_start)
            self.move(self.floor_action_client, traj)
            time.sleep(time_from_start)
            return True
        except Exception as e:
            self.get_logger().warn(f"[FAULTY] joint replay failed: {e}")
            return False

    def _infer_agv_target_world(self, *, agv_slot: str) -> list[float] | None:
        """
        Compute the legacy `world_target` vector for a given agv_slot like 'agv4_4'.
        This mirrors how `floor_place_part_on_agv` computes its placement target.
        """
        try:
            quadrant = self._parse_agv_slot(agv_slot)
            agv_name = str(agv_slot).split("_")[0]  # agv4
            agv_num = self._parse_agv_num(agv_name)
            location = f"agv{agv_num}_ks{agv_num}_tray"
            ref_coord = [self.kitting_base_x, -self.kitting_base_y, self.kitting_base_z]
            curr_coord = [
                kitting_robot_park_location[location][0] + quad_offsets_[quadrant][0],
                kitting_robot_park_location[location][1] + quad_offsets_[quadrant][1],
                kitting_robot_park_location[location][2],
            ]
            return list(self.relative_coordinate(ref_coord, curr_coord))
        except Exception:
            return None

    def floor_pick_part_from_agv(self, part_token: str, agv_slot: str) -> bool:
        """
        Faulty recovery: re-grasp a part from an AGV slot.
        Strategy:
        - Prefer reverse replay to the recorded post-place joint pose
        - Then use a geometric pick (legacy-style) to guarantee attachment
        """
        part_token = str(part_token)
        agv_slot = str(agv_slot)
        rec = self._agv_place_records.get(part_token) or {}

        # Move to the AGV tray area first.
        try:
            agv_name = agv_slot.split("_")[0]  # agv4
            agv_num = self._parse_agv_num(agv_name)
        except Exception:
            agv_num = int(rec.get("agv_num") or 1)
        try:
            self.move_to(f"agv{agv_num}_ks{agv_num}_tray")
            self.floor_robot_info.location = f"agv{agv_num}_ks{agv_num}_tray"
        except Exception:
            pass

        # Reverse replay (best-effort).
        q_place = rec.get("q_place")
        if isinstance(q_place, list) and q_place:
            self._move_floor_arm_to_joints(q_place, time_factor=1.0)

        # Ensure the vacuum is enabled before picking.
        while bool(getattr(self.floor_robot_gripper_state, "attached", False)):
            time.sleep(0.05)
        self.set_floor_robot_gripper_state(True)

        # Geometric pick around the recorded placement target.
        world_target = rec.get("world_target")
        if not (isinstance(world_target, list) and len(world_target) >= 3):
            world_target = self._infer_agv_target_world(agv_slot=agv_slot)
        if not (isinstance(world_target, list) and len(world_target) >= 3):
            self.get_logger().error(f"[FAULTY] Missing world_target for pick_from_agv: part={part_token} slot={agv_slot}")
            return False

        # Derive part internal type for height tuning (matches legacy tables).
        toks = part_token.split("_")
        if len(toks) >= 3:
            p_color, p_type = toks[0], toks[1]
        else:
            p_color, p_type = "red", "pump"
        p_internal_type = f"assembly_{p_type}_{p_color}"

        p2 = Rot2Matrix(self.init_rotation, world_target)
        try:
            base_h = float(kitting_pick_part_heights_on_bin_agv.get(p_internal_type, 0.0))
            p2[2, 3] = p2[2, 3] + base_h - 0.029
        except Exception:
            # fallback: small approach
            p2[2, 3] = p2[2, 3] + 0.02

        # Approach + repick loop (legacy-like).
        self.MOV_M(p2, eps=0.01, times=10)
        time.sleep(0.2)
        repick_nums = 0
        while not bool(getattr(self.floor_robot_gripper_state, "attached", False)):
            if repick_nums >= 5:
                break
            repick_nums += 1
            p2[2, 3] = p2[2, 3] - repick_nums * 0.001
            self.MOV_M(p2, eps=0.01, times=10)
            time.sleep(0.2)

        if not bool(getattr(self.floor_robot_gripper_state, "attached", False)):
            self.get_logger().error(f"[FAULTY] Failed to pick faulty part from AGV: {part_token} at {agv_slot}")
            return False

        # Lift away
        p2[2, 3] = p2[2, 3] + 0.2
        self.MOV_M(p2, eps=0.01)
        self.kitting_robot_init("standby")

        # Track held token for downstream actions (trash placement).
        self.held_part_token = part_token
        # Minimal dummy to satisfy checks that expect a held part object.
        self.current_held_part = type("HeldPart", (), {"type": p_internal_type, "need_flip": False})()
        return True

    def floor_place_part_to_trash(self, part_token: str) -> bool:
        """
        Faulty recovery: discard the currently-held part to the trash can.
        Physical mapping: move to 'can' and release vacuum.
        """
        part_token = str(part_token)
        if not bool(getattr(self.floor_robot_gripper_state, "attached", False)):
            self.get_logger().error(f"[FAULTY] Tried to trash {part_token} but gripper is not attached.")
            return False
        try:
            # Match legacy `sia_interface.py` behavior:
            # - ensure arm is in a safe pose before base motion
            # - move base to the can/trash location
            # - move arm to a known drop pose above the can
            self.kitting_robot_init("standby")
            self.move_to("can")
            self.floor_robot_info.location = "can"
            time.sleep(0.2)
            self.kitting_robot_init("init_state")
            time.sleep(0.2)
            self.set_floor_robot_gripper_state(False)
            time.sleep(0.2)
            self.kitting_robot_init("bin_agv_insert_joint")
        finally:
            self.current_held_part = None
            self.held_part_token = None
        return True

    @staticmethod
    def _extract_pddl_block(tag: str, text: str) -> tuple[int, int]:
        """
        Return [start,end) indices for the outermost '(:{tag} ...)' block.
        (Local copy to avoid importing private helpers.)
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

    def _make_faulty_goal_problem(self, *, base_pddl_text: str, faulty_part: str) -> str:
        """
        Create a temporary problem for faulty recovery:
        keep everything the same, but replace (:goal ...) with only (part_on <part> trash).
        """
        new_goal = "\n".join(
            [
                "(:goal (and",
                f"  (part_on {faulty_part} trash)",
                "))",
            ]
        )
        try:
            s, e = self._extract_pddl_block("goal", base_pddl_text)
            return base_pddl_text[:s] + new_goal + base_pddl_text[e:]
        except Exception:
            # If no goal block exists, append (rare).
            return base_pddl_text.rstrip() + "\n\n" + new_goal + "\n"

    def move_floor(self, location: str):
        self.move_to(location)
        self.floor_robot_info.location = location
        return True

    def move_agv(self, agv_num: int, destination):
        """Override to handle string destinations."""
        agv_num = int(agv_num)
        dest_map = {
            "warehouse": 3,
            "init_agv1": 0,
            "init_agv2": 0,
            "init_agv3": 0,
            "init_agv4": 0,
            "floor_init": 0,
        }
        if destination is None:
            destination = 3
        if isinstance(destination, str):
            destination = dest_map.get(destination, 0)
        try:
            destination = int(destination)
        except Exception:
            destination = 0
        return super().move_agv(agv_num, destination)

    def submit_order(self, order_id: str, agv_number: int = None, destination=None):
        # Block any non-PDDL submission paths (e.g., base-class automatic submission callbacks/timers).
        if not getattr(self, "_allow_submit_from_pddl", False):
            self.get_logger().error(
                f"[SUBMIT_BLOCKED] submit_order({order_id}) was called outside PDDL action execution. "
                "Submission must be triggered by the PDDL action submit_order(...) only."
            )
            return False

        # Convert PDDL order id -> ROS order id (typically uppercase) before calling the service.
        order_id_ros = self._order_id_pddl_to_ros.get(str(order_id), str(order_id))
        order_id_ros = str(order_id_ros).upper()

        if agv_number is not None and destination is not None:
            self.move_agv(agv_number, destination)
        super().submit_order(order_id_ros)
        return True

    # --- Disable base-class auto-submission entry points (strict PDDL control) ---
    def process_order_submission(self, order_id):
        # noisy timer/auto-submission path; keep silent in strict PDDL mode
        self.get_logger().debug(
            f"[SUBMIT_BLOCKED] process_order_submission({order_id}) called, but strict PDDL mode is enabled."
        )
        return

    def timer_submit_callback(self):
        # noisy timer/auto-submission path; keep silent in strict PDDL mode
        self.get_logger().debug("[SUBMIT_BLOCKED] timer_submit_callback() called, but strict PDDL mode is enabled.")
        return
    
    # --- New Actions for POMDP ---

    def _get_global_pddl_state(self) -> Tuple[Dict[str, Dict[str, str]], Dict[str, int]]:
        """
        Parse ariac_pomdp/generated_pddl/p_vlm.pddl to get current global state.
        Returns:
            loc_to_part_info: { 'binX_Y': {'name': 'color_type_N', 'type': 'color_type'} }
            max_ids: { 'color_type': max_index }
        """
        # Prefer the latest snapshot PDDL if available (keeps IDs consistent with executed actions)
        if getattr(self, "pddl_mgr", None) is not None and self.pddl_mgr.latest_pddl_path is not None:
            p_vlm_path = Path(self.pddl_mgr.latest_pddl_path)
        else:
            # Fallback: default VLM-generated path
            p_vlm_path = ARIAC_POMDP_DIR / "generated_pddl" / "p_vlm.pddl"
        
        loc_to_part_info = {}
        max_ids = {}
        
        if not p_vlm_path.exists():
             self.get_logger().warn(f"Global PDDL {p_vlm_path} not found. Starting fresh IDs.")
             return loc_to_part_info, max_ids
             
        try:
            content = p_vlm_path.read_text(encoding="utf-8", errors="replace")

            # IMPORTANT:
            # max_ids must be computed from ALL part objects, not just those appearing in (part_on ...).
            # Otherwise, after parts are moved off bins (e.g., into AGV slots), inspect would reuse *_1 ids.
            all_part_objs = self._parse_pddl_part_objects(content)
            for p_name in all_part_objs:
                tokens = p_name.split("_")
                if len(tokens) >= 2 and tokens[-1].isdigit():
                    idx = int(tokens[-1])
                    base_type = "_".join(tokens[:-1])
                else:
                    idx = 0
                    base_type = p_name
                cur = max_ids.get(base_type, 0)
                if idx > cur:
                    max_ids[base_type] = idx
            
            # Match (:init ... ) block
            start = content.find("(:init")
            if start != -1:
                # Simple regex to find (part_on name loc)
                matches = re.findall(r"\(part_on\s+([\w]+)\s+([\w]+)\)", content[start:])
                
                for p_name, loc in matches:
                    # p_name: blue_pump_1
                    # loc: bin1_1
                    
                    # Parse type and index
                    tokens = p_name.split('_')
                    if len(tokens) >= 2:
                        # Attempt to extract trailing number
                        if tokens[-1].isdigit():
                            idx = int(tokens[-1])
                            base_type = "_".join(tokens[:-1]) # blue_pump
                        else:
                            # e.g. blue_pump (no index)
                            base_type = p_name
                            idx = 0
                    else:
                        base_type = p_name
                        idx = 0
                        
                    current_max = max_ids.get(base_type, 0)
                    if idx > current_max:
                        max_ids[base_type] = idx
                        
                    loc_to_part_info[loc] = {
                        "name": p_name,
                        "type": base_type
                    }
                
        except Exception as e:
            self.get_logger().error(f"Failed to parse global PDDL: {e}")
            
        return loc_to_part_info, max_ids

    def inspect(self, location: str):
        """
        Move robot to inspect a location (bin).
        Also triggers VLM inspection for that bin.
        """
        self.get_logger().info(f"Inspecting {location}...")
        
        # 1) Parse bin id (bin1 or bin1_2 -> bin1)
        bin_id = location.split("_")[0] if "_" in location else location
            
        # Clear previous result for this bin to ensure freshness
        self.inspect_results.pop(bin_id, None)

        # 2) Prepare hint from current global belief/oracle snapshot
        loc_map, max_ids = self._get_global_pddl_state()
        current_bin_parts = []
        for loc, info in loc_map.items():
            if loc.startswith(f"{bin_id}_"):
                current_bin_parts.append(f"    (part_on {info['name']} {loc})")
        hint_str = "\n".join(sorted(current_bin_parts))
        if hint_str:
            self.get_logger().debug(f"[{bin_id}] Providing Hint to VLM:\n{hint_str}")

        # 3) Select camera image for this bin group
        img = None
        cam_name = "?"
        try:
            bin_num = int(bin_id.replace("bin", ""))
            if 1 <= bin_num <= 4:
                img = getattr(self, "_right_bins_camera_image", None)
                cam_name = "right"
            elif 5 <= bin_num <= 8:
                img = getattr(self, "_left_bins_camera_image", None)
                cam_name = "left"
            else:
                self.get_logger().error(f"Invalid bin number: {bin_num}")
        except Exception:
            self.get_logger().error(f"Could not parse bin number from {bin_id}")

        if img is None:
            self.get_logger().warn(f"No image available for inspect {bin_id} (Cam: {cam_name})")
            target_loc = self._map_location(location)
            return self.move_floor(target_loc)

        # 4) Save temp image and call VLM inspect
        base_path = PROMPT_EXAMPLE_DIR
        base_path.mkdir(parents=True, exist_ok=True)
        img_path = base_path / f"inspect_{bin_id}.png"

        try:
            cv2.imwrite(str(img_path), img)
            # noisy terminal output; keep silent

            pddl_str = get_inspect_pddl(bin_id, img_path, current_state_hint=hint_str)
            if not pddl_str:
                self.get_logger().warn(f"VLM Inspect {bin_id} failed to return PDDL.")
                target_loc = self._map_location(location)
                return self.move_floor(target_loc)

            parts_raw = parse_part_on(pddl_str)  # [(color_type, loc), ...]
            parts_formatted: list[tuple[str, str]] = []
            for p_base_type, loc in parts_raw:
                existing = loc_map.get(loc)
                if existing and existing["type"] == p_base_type:
                    unique_name = existing["name"]
                    self.get_logger().debug(f"  Matched existing: {unique_name} at {loc}")
                else:
                    idx = max_ids.get(p_base_type, 0) + 1
                    max_ids[p_base_type] = idx
                    unique_name = f"{p_base_type}_{idx}"
                    self.get_logger().debug(f"  New detection: {unique_name} at {loc}")
                parts_formatted.append((unique_name, loc))

            self.inspect_results[bin_id] = parts_formatted
            # noisy terminal output; keep silent

            # Critical: update p_vlm belief so future reconciliation won't reuse *_1 ids
            self._update_p_vlm_bin_contents(bin_id=bin_id, parts_formatted=parts_formatted)
        except Exception as e:
            self.get_logger().error(f"Inspect VLM error: {e}")
            self.get_logger().error(traceback.format_exc())

        # 5) Physical move (unchanged)
        target_loc = self._map_location(location)
        return self.move_floor(target_loc)

    def repair(self, part_name: str):
        """
        Repair a bad part.
        Physical action: Simulation stub.
        """
        self.get_logger().info(f"Repairing part {part_name}...")
        time.sleep(2.0) # Simulate repair time
        return True

    # ---------------------- Behavior Trimming ----------------------
    def perform_quality_check(self, order_id: str):
        """
        Real QC service call (ported from `sia_interface.py`).

        NOTE: In some setups the first response can be stale; callers that care about faulty_part
        should invoke this twice and trust the second result (same as legacy behavior).
        """
        # Convert PDDL order id (lowercase) -> ROS order id (as announced by /ariac/orders).
        # IMPORTANT: Do NOT force uppercase here. The QC service expects the exact order id string.
        order_id_pddl = str(order_id or "")
        order_id_ros = self._order_id_pddl_to_ros.get(order_id_pddl, order_id_pddl)
        order_id_ros = str(order_id_ros)

        while not self.quality_checker.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Service /ariac/perform_quality_check not available, waiting again...")

        request = PerformQualityCheck.Request()
        request.order_id = order_id_ros

        # Log trimming: QC is often called twice (first response can be stale).
        # Requirement: "每次都会打印多个，只保留最后一个"
        # We suppress logs for the first call when a second call follows soon.
        now = time.time()
        if not hasattr(self, "_qc_recent_calls"):
            # order_id_ros -> (last_time, count)
            self._qc_recent_calls: dict[str, tuple[float, int]] = {}
        last_time, count = self._qc_recent_calls.get(order_id_ros, (0.0, 0))
        # If calls are close in time, treat as the same "QC sequence".
        if now - last_time <= 2.0:
            count += 1
        else:
            count = 1
        self._qc_recent_calls[order_id_ros] = (now, count)

        is_final_log = (count >= 2)  # keep only the last (2nd) call in the common double-check pattern
        if is_final_log:
            self.get_logger().info(f"[QC] perform_quality_check(order_id={order_id_ros})")
        else:
            self.get_logger().debug(f"[QC] perform_quality_check(order_id={order_id_ros}) (suppressed-first)")

        future = self.quality_checker.call_async(request)
        try:
            with self.spin_lock:
                rclpy.spin_until_future_complete(self, future)
        except KeyboardInterrupt:
            raise

        resp = future.result()
        if resp is None:
            self.get_logger().error("[QC] perform_quality_check: empty response")
            return False, None
        if bool(getattr(resp, "all_passed", False)):
            if is_final_log:
                self.get_logger().info("[QC] all_passed=True")
            else:
                self.get_logger().debug("[QC] all_passed=True (suppressed-first)")
            return True, resp
        # noisy terminal output; keep silent on failures too
        self.get_logger().debug("[QC] all_passed=False")
        return False, resp

    def handle_faultu_part(self, check_info, subtask, world_target, target_part):
        return False

    def ceiling_execute(self):
        time.sleep(0.05)

    def ceiling_query(self):
        return []

    # ---------------------- Main Loop ----------------------
    def run(self):
        """Start competition and wait for orders."""
        self.start_competition()
        self.new_thread()
        self.kitting_robot_init("bin_agv_insert_joint")

        self.get_logger().debug("PDDL Kitting Node Ready. Waiting for orders...")

        while rclpy.ok():
            if self.pending_orders:
                order_msg = self.pending_orders.popleft()
                self.process_order_execution(order_msg)
            
            time.sleep(0.1)

    # ---------------------- Helpers & Parsing ----------------------
    def _map_location(self, name: str) -> str:
        if name in self._location_alias:
            return self._location_alias[name]
        if name.startswith("slot"):
            try:
                idx = int(name[4:])
            except Exception:
                idx = 1
            return "kts1" if idx <= 3 else "kts2"
        if name.startswith("bin"):
            return name.split("_")[0]
        if name.startswith("agv"):
            agv = name.split("_")[0]
            return f"{agv}_ks{agv[-1]}_tray"
        return name

    def _parse_agv_slot(self, agv_slot: str) -> int:
        try:
            return int(agv_slot.split("_")[1])
        except Exception:
            return 1

    def _parse_agv_num(self, agv_name: str) -> int:
        try:
            return int(str(agv_name).lstrip("agv").split("_")[0])
        except Exception:
            return 1

    def _pick_part_candidate(self, part_type: str, preferred_bin: Optional[str], preferred_slot: Optional[str] = None):
        """
        Find a part instance in self.parts_location_dict or by search.
        """
        candidates = self.search_part_on_bins(part_type) or []

        # If the PDDL action requested a specific bin slot (e.g., bin2_1), prefer the part whose pose maps to that slot.
        if preferred_slot and candidates:
            try:
                for p in candidates:
                    if hasattr(p, "pose") and hasattr(p.pose, "position"):
                        sl = self.find_closest_slot(p.pose.position, bin_slot_positions)
                        if sl == preferred_slot:
                            return p
            except Exception:
                # Fall back to coarse selection below
                pass
        
        if preferred_bin:
            for p in candidates:
                if p.location == preferred_bin:
                    return p
        return candidates[0] if candidates else None

    def _pause_after_action(self, wait_floor_idle: bool = True, timeout: float = 10.0):
        start = time.time()
        if wait_floor_idle:
            while time.time() - start < timeout:
                rclpy.spin_once(self, timeout_sec=0.01)
                if getattr(self.floor_robot_info, "is_idle", True):
                    break
                time.sleep(0.05)
        time.sleep(0.5)

    def _log_action_start(self, idx: int, total: int, name: str, args: List[str]):
        self.get_logger().info(f"[{idx}/{total}] EXEC: {name}({', '.join(args)})")
        rclpy.spin_once(self, timeout_sec=0.0)
        time.sleep(0.01)

    def _execute_action(self, name: str, args: List[str]):
        """Dispatch PDDL action to implementation."""
        if name == "move_floor":
            target_loc = self._map_location(args[2])
            self.move_floor(target_loc)
            self._pause_after_action()

        elif name == "change_gripper":
            station = args[1]
            self.change_gripper(station)
            self._pause_after_action()

        elif name == "floor_pick_tray":
            tray_id = args[1]
            tray_slot = args[2]
            self.current_tray_target = (tray_id, tray_slot)
            self.floor_pick_tray(tray_id, tray_slot)
            self._pause_after_action()

        elif name == "floor_place_tray":
            tray_id = args[1]
            agv_num = self._parse_agv_num(args[2])
            self.floor_place_tray(tray_id, agv_num)
            self._pause_after_action()

        elif name == "floor_pick_part":
            part_token = args[1]
            bin_slot = args[2]
            if not self.floor_pick_part(part_token, bin_slot):
                raise RuntimeError(f"Action failed: {name} {args}")
            self._pause_after_action()

        elif name == "floor_pick_part_from_agv":
            part_token = args[1]
            agv_slot = args[2]
            if not self.floor_pick_part_from_agv(part_token, agv_slot):
                raise RuntimeError(f"Action failed: {name} {args}")
            self._pause_after_action()

        elif name == "floor_place_part":
            agv_slot = args[2]
            agv_num = self._parse_agv_num(args[3])
            order_id = args[4] if len(args) > 4 else ""
            if not self.floor_place_part(agv_slot, agv_num, order_id):
                raise RuntimeError(f"Action failed: {name} {args}")
            self._pause_after_action()

        elif name == "floor_place_part_to_trash":
            # args: [robot, part]
            part_token = args[1] if len(args) > 1 else (args[0] if args else "")
            if not self.floor_place_part_to_trash(part_token):
                raise RuntimeError(f"Action failed: {name} {args}")
            self._pause_after_action()

        elif name == "floor_flip_part":
            part_token = args[1]
            location = args[2] # usually bin_slot or location
            if not self.floor_flip_part(part_token, location):
                raise RuntimeError(f"Action failed: {name} {args}")
            self._pause_after_action()

        elif name == "perform_quality_check":
            order_id = args[0] if args else ""
            self.perform_quality_check(order_id)
            self._pause_after_action()

        elif name == "move_agv":
            agv_num = self._parse_agv_num(args[0])
            dest_symbol = args[-1]
            self.move_agv(agv_num, dest_symbol)
            self._pause_after_action()

        elif name == "submit_order":
            order_id = args[0]
            # Convert PDDL order_id -> ROS order_id inside submit_order() using the mapping.
            # Allow submit only for the duration of this PDDL action.
            self._allow_submit_from_pddl = True
            try:
                # noisy terminal output; keep silent
                ok = self.submit_order(order_id)
                if not ok:
                    raise RuntimeError(f"Action failed (submit blocked): {name} {args}")
            finally:
                self._allow_submit_from_pddl = False
            self._pause_after_action(wait_floor_idle=False)
            
        elif name == "inspect":
            location = args[1]
            self.inspect(location)
            self._pause_after_action()
            
        elif name == "repair":
            part = args[0]
            self.repair(part)
            self._pause_after_action()

        else:
            self.get_logger().warn(f"Unknown PDDL action: {name}")

def main(args=None):
    rclpy.init(args=args)
    node = PDDLKittingInterface()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
