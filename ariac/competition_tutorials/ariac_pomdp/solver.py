#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import traceback
import threading
import heapq
import cv2
from collections import deque
from pathlib import Path
from typing import Optional, List, Dict, Any

import rclpy
from rclpy.executors import MultiThreadedExecutor

# Interface
from competition_tutorials.ariac_interface import PDDLKittingInterface, determine_part_name

# VLM / PDDL
from competition_tutorials.eval import get_replanning_pddl, parse_part_on
from competition_tutorials.ariac_pomdp.pddl_generation import generate_problem_pddl
from competition_tutorials.ariac_pomdp.hybrid_controller import HybridController, HybridParams
from competition_tutorials.ariac_pomdp.classical_planner import ClassicalPlanner
from competition_tutorials.ariac_pomdp.ariac_kitting_api import JuliaInfoClient
from competition_tutorials.ariac_pomdp.real_env_up import RealWorldUPEnv, StepResult

ARIAC_POMDP_DIR = Path(__file__).resolve().parent
PROMPT_EXAMPLE_DIR = ARIAC_POMDP_DIR.parent / "prompt_example"

class OrderPriorityWrapper:
    """Wrapper to make orders comparable for PriorityQueue (heapq)."""
    def __init__(self, order_msg):
        self.msg = order_msg
        # High priority (1) should come before Low priority (0).
        # heapq is a min-heap. So we negate priority? 
        # Actually ARIAC priorities: usually boolean or int. 
        # Checking msg definition: priority is usually boolean (high_priority=True/False) or int.
        # Let's assume int: 1=High, 0=Low. We want High first.
        # So sort by -priority.
        self.priority = getattr(order_msg, 'priority', 0)
        self.id = order_msg.id
        self.time = time.time() # FIFO for same priority

    def __lt__(self, other):
        if self.priority != other.priority:
            return self.priority > other.priority
        return self.time < other.time

class AriacSolverNode:
    def __init__(self, interface: PDDLKittingInterface):
        self.interface = interface
        self.lock = threading.Lock()
        
        # Priority Queue for orders
        self.order_queue = [] # heap
        self.current_order_id = None
        self.stop_current_flag = False
        
        # Override interface's order processing to redirect to our queue
        # The interface.process_orders appends to self.pending_orders
        # We will monitor self.interface.pending_orders and move them to our heap
        
    def monitor_orders(self):
        """Check interface queue and update priority queue."""
        while self.interface.pending_orders:
            msg = self.interface.pending_orders.popleft()
            wrapped = OrderPriorityWrapper(msg)
            with self.lock:
                heapq.heappush(self.order_queue, wrapped)
                self.interface.get_logger().info(f"Solver: Added order {msg.id} to priority queue. Priority={msg.priority}")
                
                # Preemption Check
                if self.current_order_id and wrapped.priority > 0: # If new high priority
                    # Simple logic: If we are running a low priority order, request stop
                    # Need to know priority of current order.
                    # For now, just flag interruption if ANY new high priority comes in
                    self.interface.get_logger().warn(f"Solver: High Priority Order {msg.id} received! Requesting stop of {self.current_order_id}.")
                    self.stop_current_flag = True

    def get_next_order(self):
        with self.lock:
            if self.order_queue:
                return heapq.heappop(self.order_queue).msg
        return None

    def run(self):
        self.interface.get_logger().info("Solver: Loop started.")
        
        while rclpy.ok():
            self.monitor_orders()
            
            if self.current_order_id is None:
                order_msg = self.get_next_order()
                if order_msg:
                    self.execute_order(order_msg)
                else:
                    time.sleep(0.5)
            else:
                # Should not happen if execute_order blocks, but just in case
                time.sleep(0.5)

    def _convert_msg_to_task(self, order_msg, state_real) -> Optional[Dict]:
        """Convert ROS message to Task Dict for PDDL gen."""
        order_id = order_msg.id
        order_kitting_task = order_msg.kitting_task
        
        order_task = {
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
            order_task['kitting_task']['products'].append({
                'type': p_type,
                'color': p_color,
                'quadrant': p.quadrant
            })
            
        # Feasibility check
        if not self.interface._check_order_feasibility(order_task, state_real):
            return None
            
        return order_task

    def execute_order(self, order_msg):
        self.current_order_id = order_msg.id
        self.stop_current_flag = False
        
        self.interface.get_logger().info(f"Solver: Starting execution of order {order_msg.id}")
        
        # 1. Wait/Get Sensors
        self.interface.wait_for_sensors(timeout=5.0)
        
        # 2. Initial State Capture
        state_vlm = self.interface.get_vlm_world_state()
        state_real = self.interface.get_current_world_state()
        
        # 3. Task Dict
        order_task = self._convert_msg_to_task(order_msg, state_real)
        if not order_task:
            self.interface.get_logger().error(f"Solver: Order {order_msg.id} not feasible.")
            self.current_order_id = None
            return

        # 4. Generate Initial PDDL
        base_path = ARIAC_POMDP_DIR
        gen_dir = base_path / "generated_pddl"
        gen_dir.mkdir(parents=True, exist_ok=True)
        
        p_vlm_path = str(gen_dir / "p_vlm.pddl")
        p_real_path = str(gen_dir / "p_real.pddl")
        domain_path = str(base_path / "domain.pddl")
        solve_pomdp_jl = str(base_path / "solve_pomdp.jl")
        
        # Initial PDDL Generation
        # Never create *_missing_* placeholders; if VLM missed but REAL has, borrow REAL part names.
        pddl_vlm = generate_problem_pddl(
            state_vlm,
            order_task,
            "ariac_vlm",
            "ariac_kitting",
            fallback_world_state=state_real,
            allow_missing=False,
        )
        # REAL must be complete; otherwise abort early.
        pddl_real = generate_problem_pddl(state_real, order_task, "ariac_real", "ariac_kitting", allow_missing=False)
        
        with open(p_vlm_path, "w") as f: f.write(pddl_vlm)
        with open(p_real_path, "w") as f: f.write(pddl_real)

        # 5. Initialize Hybrid Controller
        try:
            info = JuliaInfoClient(domain_path=domain_path, p_vlm_path=p_vlm_path, solve_pomdp_jl=solve_pomdp_jl)
            # IMPORTANT: Pass self.interface to RealWorldUPEnv wrapper
            # We use a custom Env wrapper in this file to handle replanning triggers? 
            # Actually we use RealWorldARIACEnv from interface file logic.
            # But here we are in solver.py.
            # We can import RealWorldARIACEnv from ariac_interface.
            from competition_tutorials.ariac_interface import RealWorldARIACEnv
            
            env = RealWorldARIACEnv(domain_path, p_real_path, interface=self.interface)
            planner = ClassicalPlanner(domain_path)
            ctrl = HybridController(env, planner, info, base_problem_path=p_vlm_path, params=HybridParams())
            
            # 6. Step Loop
            step_count = 0
            while True:
                # Check Interruption
                self.monitor_orders() # Update queue
                if self.stop_current_flag:
                    self.interface.get_logger().warn(f"Solver: Preempting order {self.current_order_id}")
                    self.interface.cancel_current_action()
                    # Re-queue current order? 
                    # For simplicty: drop it or re-queue. Let's re-queue.
                    # Reset its priority? Or keep as is.
                    # wrapped = OrderPriorityWrapper(order_msg)
                    # heapq.heappush(self.order_queue, wrapped)
                    self.current_order_id = None
                    return # Exit execution

                step_count += 1
                self.interface.get_logger().info(f"--- Step {step_count} ---")
                
                # Execute Step
                out = ctrl.step()
                
                if out.get("done"):
                    reason = out.get("reason")
                    self.interface.get_logger().info(f"Solver: Order {self.current_order_id} Done. Reason: {reason}")
                    if reason == "submitted":
                        break # Success
                    else:
                        break # Failed?
                
                if not out.get("ok"):
                    # ACTION FAILED (e.g. drop)
                    reason = out.get("events", [])[-1].get("reason", "unknown")
                    self.interface.get_logger().warn(f"Solver: Action Failed! Reason: {reason}. Triggering Replanning...")
                    
                    # REPLANNING LOGIC
                    # 1. Capture Images
                    img_dir = PROMPT_EXAMPLE_DIR
                    img_dir.mkdir(parents=True, exist_ok=True)
                    left_p = img_dir / "left_bins_camera_raw.png"
                    right_p = img_dir / "right_bins_camera_raw.png"
                    
                    # Save current images from interface
                    if self.interface._left_bins_camera_image is not None:
                         cv2.imwrite(str(left_p), self.interface._left_bins_camera_image)
                    if self.interface._right_bins_camera_image is not None:
                         cv2.imwrite(str(right_p), self.interface._right_bins_camera_image)

                    # 2. Get Hint (current PDDL init)
                    # We can read the last p_vlm.pddl or better: construct from ctrl.belief?
                    # HybridController doesn't easily expose "current PDDL string".
                    # But we can just use the last generated p_vlm.pddl content as hint.
                    with open(p_vlm_path, "r") as f:
                        current_hint = f.read()

                    # 3. Call VLM Replanning
                    # Description of order
                    order_desc = str(order_task)
                    
                    new_pddl = get_replanning_pddl(order_desc, current_hint, left_p, right_p)
                    
                    if new_pddl:
                        self.interface.get_logger().info("Solver: Replanning successful. Updating PDDL...")
                        with open(p_vlm_path, "w") as f:
                            f.write(new_pddl)
                            
                        # Re-init Controller with new problem
                        # Note: We keep the same Env/Planner, but update the "Agent Problem"
                        # HybridController takes base_problem_path.
                        # We might need to re-instantiate logic or update ctrl.
                        # Easiest: Re-instantiate HybridController
                        ctrl = HybridController(env, planner, info, base_problem_path=p_vlm_path, params=HybridParams())
                        continue # Retry loop
                    else:
                        self.interface.get_logger().error("Solver: Replanning Failed (VLM error). Aborting Order.")
                        break

                time.sleep(0.1)

        except Exception as e:
            self.interface.get_logger().error(f"Solver: Execution Error: {e}")
            self.interface.get_logger().error(traceback.format_exc())
            
        self.current_order_id = None
        self.interface.get_logger().info("Solver: Order execution finished.")


def main():
    rclpy.init()
    
    # 1. Create Interface Node
    interface = PDDLKittingInterface()
    
    # 2. Create Solver Manager
    solver = AriacSolverNode(interface)
    
    # 3. Spin Interface in Background Thread
    executor = MultiThreadedExecutor()
    executor.add_node(interface)
    spinner_thread = threading.Thread(target=executor.spin, daemon=True)
    spinner_thread.start()
    
    # Init Robot
    time.sleep(2.0) # Allow connections to establish
    interface.get_logger().info("Solver: Starting Competition...")
    interface.start_competition()
    interface.kitting_robot_init("bin_agv_insert_joint")
    
    # 4. Run Solver Main Loop
    try:
        solver.run()
    except KeyboardInterrupt:
        pass
    finally:
        interface.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
