(define (domain ariac_kitting)
  (:requirements
    :strips :typing :negative-preconditions :equality
    :disjunctive-preconditions :quantified-preconditions :adl
    :numeric-fluents :action-costs)

  ;; ---------- Types ----------
  (:types
    robot tray part order agv  location - object
    tray_slot bin_slot agv_slot agv_destination gripper_station - location
  )

  ;; ---------- Constants ----------
  (:constants
    warehouse - agv_destination
    trash - bin_slot
  )

  ;; ---------- Predicates ----------
  (:predicates
    ;; 位置/可达
    (at_robot ?r - robot ?l - location)
    ;; 不再使用 robot_reach 正向列举可达点；改用 floor_forbid_reach 反向约束
    (agv_at ?a - agv ?d - agv_destination)
    (agv_reach ?a - agv ?d - agv_destination)            ; AGV 可达目的地
    (home ?a - agv ?d - agv_destination)                 ; AGV 的初始停靠点

    (has_tray_gripper ?r - robot)
    (has_part_gripper ?r - robot)
    (gripper_empty ?r - robot)
    (holding_tray ?r - robot ?t - tray)
    (holding_part ?r - robot ?p - part)

    ;; 托盘与零件
    (tray_on_slot ?t - tray ?s - tray_slot)              ; 托盘在固定托位(非 AGV)
    (part_on ?p - part ?s - bin_slot)                    ; 零件在料箱/台面
    (on_agv ?t - tray ?a - agv)                          ; 托盘装到 AGV
    (in_slot ?p - part ?s - agv_slot)                    ; 零件在 AGV 槽位
    (slot_empty ?s - agv_slot)
    (slot_of ?s - agv_slot ?a - agv)                     ; 槽位归属哪台 AGV

    ;; 订单/质检（改为三元）
    (order_uses_tray ?o - order ?t - tray ?a - agv)      ; 订单-托盘-AGV 绑定
    (order_needs_part ?o - order ?p - part ?s - agv_slot); 订单-零件-目标槽位
    (flipped ?p - part)
    (good    ?p - part)
    (bad ?p - part)
    (submitted ?o - order)
    (precedes ?o_hi - order ?o_lo - order)
    (floor_forbid_reach ?l - location) 
  )

  (:functions (total-cost))
  ;; 更换夹爪(托盘<->零件)，要求空手
  (:action change_gripper
    :parameters (?r - robot ?gs - gripper_station)
    :precondition (and (at_robot ?r ?gs) (gripper_empty ?r))
    :effect (and
      (when (has_tray_gripper ?r)
        (and (has_part_gripper ?r) (not (has_tray_gripper ?r))))
      (when (has_part_gripper ?r)
        (and (has_tray_gripper ?r) (not (has_part_gripper ?r)))))
  )

  ;; 机器人从托位取托盘
  (:action floor_pick_tray
    :parameters (?r - robot ?t - tray ?s - tray_slot)
    :precondition (and
      (at_robot ?r ?s) (tray_on_slot ?t ?s)
      (has_tray_gripper ?r) (gripper_empty ?r))
    :effect (and
      (holding_tray ?r ?t)
      (not (tray_on_slot ?t ?s))
      (not (gripper_empty ?r)))
  )

  ;; 放托盘到 AGV（仅在初始对接位对接）
  (:action floor_place_tray
    :parameters (?r - robot ?t - tray ?a - agv ?o - order)
    :precondition (and
      (holding_tray ?r ?t)
      (exists (?d - agv_destination)
        (and (home ?a ?d) (agv_at ?a ?d) (at_robot ?r ?d)))
      (order_uses_tray ?o ?t ?a))
    :effect (and
      (on_agv ?t ?a)
      (gripper_empty ?r)
      (not (holding_tray ?r ?t)))
  )

  ;; 从料箱取零件
  (:action floor_pick_part
    :parameters (?r - robot ?p - part ?s - bin_slot)
    :precondition (and
      (at_robot ?r ?s) (part_on ?p ?s)
      (has_part_gripper ?r) (gripper_empty ?r))
    :effect (and
      (holding_part ?r ?p)
      (not (part_on ?p ?s))
      (not (gripper_empty ?r)))
  )

  (:action floor_pick_part_from_agv
    :parameters (?r - robot ?p - part ?s - agv_slot)
    :precondition (and
      (at_robot ?r ?s) (in_slot ?p ?s)
      (has_part_gripper ?r) (gripper_empty ?r))
    :effect (and
      (holding_part ?r ?p)
      (not (in_slot ?p ?s))
      (not (gripper_empty ?r)))
  )

  (:action floor_place_part_to_trash
    :parameters (?r - robot ?p - part )
    :precondition (and
      (at_robot ?r trash) (holding_part ?r ?p))
    :effect (and
      (part_on ?p trash)
      (not (holding_part ?r ?p))
      (gripper_empty ?r))
  )

  ;; 将零件放入 AGV 槽位（AGV 必须在初始对接位，且该订单托盘已在该 AGV；零件必须放到订单指定的槽位）
  (:action floor_place_part
    :parameters (?r - robot ?p - part ?s - agv_slot ?a - agv ?o - order)
    :precondition (and
      (at_robot ?r ?s)
      (holding_part ?r ?p)
      (slot_of ?s ?a) (slot_empty ?s)
      (exists (?d - agv_destination)
        (and (home ?a ?d) (agv_at ?a ?d)))
      ;; 该订单确实需要把该零件放到这个槽位
      (order_needs_part ?o ?p ?s)
      ;; 该订单使用的托盘在这台 AGV 上
      (exists (?t - tray) (and (order_uses_tray ?o ?t ?a) (on_agv ?t ?a))))
    :effect (and
      (in_slot ?p ?s)
      (gripper_empty ?r)
      (not (holding_part ?r ?p))
      (not (slot_empty ?s)))
  )

  ;; 在当前位置翻转零件：允许在料箱面翻转
  (:action floor_flip_part
    :parameters (?r - robot ?p - part ?l - location)
    :precondition (and (at_robot ?r ?l) (flipped ?p))
    :effect (not (flipped ?p))
  )

  ;; 机器人移动（受导轨拓扑约束）
  (:action move_floor
    :parameters (?r - robot ?l1 - location ?l2 - location)
    :precondition (and
      (at_robot ?r ?l1)
      (not (= ?l1 ?l2))
      (not (floor_forbid_reach ?l2)))
    :effect (and (at_robot ?r ?l2) (not (at_robot ?r ?l1)) (increase (total-cost) 1))
  )

  ;; AGV 移动（受可达拓扑约束）
  (:action move_agv
    :parameters (?a - agv ?d1 - agv_destination ?d2 - agv_destination)
    :precondition (and (agv_at ?a ?d1) (agv_reach ?a ?d2)(not (= ?d1 ?d2)))
    :effect (and (agv_at ?a ?d2) (not (agv_at ?a ?d1)))
  )

  ;; 质量检查：在 AGV 槽位上执行
  (:action perform_quality_check
    :parameters (?o - order ?p - part ?s - agv_slot ?a - agv)
    :precondition (and
      (order_needs_part ?o ?p ?s)
      (in_slot ?p ?s)
      (slot_of ?s ?a)
      (exists (?d - agv_destination)
        (and (home ?a ?d) (agv_at ?a ?d))))
    :effect (and (in_slot ?p ?s) )
  )

  (:action inspect
    :parameters (?r - robot ?l - location )
    :precondition (and (gripper_empty ?r) (at_robot ?r ?l))
    :effect (and
      (gripper_empty ?r)
      (at_robot ?r ?l)
    )
  )

  ;; 提交订单：订单托盘所在 AGV 已到仓库，且订单所需零件均在对应槽位且合格
  (:action submit_order
    :parameters (?o - order ?a - agv)
    :precondition (and
      ;; 订单绑定的托盘在该 AGV 上
      (exists (?t - tray) (and (order_uses_tray ?o ?t ?a) (on_agv ?t ?a)))
      (agv_at ?a warehouse)
      ;; 优先级/前置订单约束：所有 precedes(?o2, ?o) 的订单必须先提交
      (forall (?o2 - order) (imply (precedes ?o2 ?o) (submitted ?o2)))
      (forall (?p - part ?s - agv_slot)
        (imply (order_needs_part ?o ?p ?s)(and (in_slot ?p ?s) (slot_of ?s ?a) (good ?p))))
      (not (submitted ?o)))
    :effect (submitted ?o)
  )

)
