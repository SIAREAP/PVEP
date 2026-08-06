#!/usr/bin/env python3

import rclpy
import time
import threading
from rclpy.executors import MultiThreadedExecutor,SingleThreadedExecutor
from competition_tutorials.sia_interface import SIAInterface




def main(args=None):
    rclpy.init(args=args)


    interface = SIAInterface()
    
    interface.new_thread()
    
    # interface.wait(2)
    
    interface.start_competition()
    
    interface.wait(2)

    interface.run()
    


    interface.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

