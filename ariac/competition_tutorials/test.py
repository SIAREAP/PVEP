#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from ariac_msgs.msg import (
    CompetitionState as CompetitionStateMsg,
    BreakBeamStatus as BreakBeamStatusMsg,
    AdvancedLogicalCameraImage as AdvancedLogicalCameraImageMsg,
    Part as PartMsg,
    PartPose as PartPoseMsg,
    Order as OrderMsg,
    AssemblyPart as AssemblyPartMsg,
    AGVStatus as AGVStatusMsg,
    AssemblyTask as AssemblyTaskMsg,
    AssemblyState as AssemblyStateMsg,
    CombinedTask as CombinedTaskMsg,
    VacuumGripperState,
)

class OrderSubscriber(Node):
    def __init__(self):
        super().__init__('order_subscriber')
        self.subscription = self.create_subscription(
            OrderMsg,
            '/ariac/orders',
            self.orders_callback,
            10
        )
        self.subscription  # prevent unused variable warning

    def orders_callback(self, msg):
        self.get_logger().info(f'Received order: {msg} ')

def main(args=None):
    rclpy.init(args=args)
    order_subscriber = OrderSubscriber()
    rclpy.spin(order_subscriber)
    rclpy.shutdown()

if __name__ == '__main__':
    main()