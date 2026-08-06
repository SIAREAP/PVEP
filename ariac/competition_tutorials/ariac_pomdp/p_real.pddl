(define (problem Trial_new_1)
  (:domain ariac_kitting)

  (:objects
    agv3 - agv
    init_agv1 init_agv2 init_agv3 init_agv4 - agv_destination
    agv3_3 agv3_4 - agv_slot
    bin1_1 bin1_2 bin1_3 bin2_1 bin2_2 bin2_3 bin4_2 bin4_3 bin4_6 - bin_slot
    gripper_station1 gripper_station2 - gripper_station
    floor_init trash - location
    iq2kq3z3 - order
    blue_battery_1 green_battery_1 green_regulator_1 orange_regulator_1 purple_battery_1 red_battery_1 red_sensor_1 red_sensor_2 - part
    floor - robot
    tray7 - tray
    slot2 - tray_slot
  )

  (:init
    (agv_at agv3 init_agv3)
    (agv_reach agv3 init_agv3)
    (agv_reach agv3 warehouse)
    (at_robot floor floor_init)
    (floor_forbid_reach bin4_2)
    (floor_forbid_reach bin4_3)
    (floor_forbid_reach bin4_6)
    (good blue_battery_1)
    (good green_battery_1)
    (good green_regulator_1)
    (good orange_regulator_1)
    (good purple_battery_1)
    (good red_battery_1)
    (good red_sensor_1)
    (good red_sensor_2)
    (gripper_empty floor)
    (has_part_gripper floor)
    (home agv3 init_agv3)
    (order_needs_part iq2kq3z3 green_battery_1 agv3_3)
    (order_needs_part iq2kq3z3 red_sensor_1 agv3_4)
    (order_uses_tray iq2kq3z3 tray7 agv3)
    (part_on blue_battery_1 bin4_2)
    (part_on green_battery_1 bin2_3)
    (part_on green_regulator_1 bin2_2)
    (part_on orange_regulator_1 bin4_3)
    (part_on purple_battery_1 bin4_6)
    (part_on red_battery_1 bin1_2)
    (part_on red_sensor_1 bin1_1)
    (part_on red_sensor_2 bin2_1)
    (slot_empty agv3_3)
    (slot_empty agv3_4)
    (slot_of agv3_3 agv3)
    (slot_of agv3_4 agv3)
    (tray_on_slot tray7 slot2)
    (= (total-cost) 0)
  )

  (:goal (and
    (submitted iq2kq3z3)
  ))
  (:metric minimize (total-cost))
)
