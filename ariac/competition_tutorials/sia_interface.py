#!/usr/bin/env python3

import rclpy
from collections import deque
import threading
import cv2
from cv_bridge import CvBridge, CvBridgeError 
import numpy as np
from rclpy.node import Node
from ament_index_python import get_package_share_directory
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Duration
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image as ImageMsg
from ariac_msgs.msg import *
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

from imutils.object_detection import non_max_suppression
from std_srvs.srv import Trigger
from ariac_msgs.srv import *
from tf2_ros import TransformException
from competition_tutorials.data import *
from geometry_msgs.msg import Pose,PoseStamped,Vector3,Quaternion
from sensor_msgs.msg import JointState,Image
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointControllerState,JointTrajectoryControllerState
from competition_tutorials.Kinematics import *
from competition_tutorials.interpolation import *
from competition_tutorials.my_tf import *
from rclpy.timer import Rate
import functools
from concurrent.futures import ThreadPoolExecutor
from os import path

class Subtask:
    def __init__(self, order_id, product_type, is_last_subtask):
        self.order_id = order_id
        self.is_done = False
        self.is_flip = False
        self.product_type = product_type
        self.is_last_subtask = is_last_subtask

class KittingSubtask(Subtask):
    def __init__(self, order_id, agv_number, tray_id, destination, product_type,product_quadrant, is_last_subtask):
        super().__init__(order_id, product_type, is_last_subtask)
        self.agv_number = agv_number
        self.tray_id = tray_id
        self.destination = destination
        self.product_quadrant = product_quadrant

class AssemblySubtask(Subtask):
    def __init__(self, order_id, agv_numbers, station, product_type, is_last_subtask,assembled_pose,install_direction,grap_pose):
        super().__init__(order_id, product_type, is_last_subtask)
        self.agv_numbers =agv_numbers
        self.station = station
        self.assembled_pose=assembled_pose
        self.install_direction=install_direction
        self.grap_pose=grap_pose
        

class CombinedSubtask(Subtask):
    def __init__(self, order_id, station, product_type, is_last_subtask,assembled_pose,install_direction,grap_pose):
        super().__init__(order_id, product_type, is_last_subtask)
        self.station = station
        self.assembled_pose=assembled_pose
        self.install_direction=install_direction
        self.grap_pose=grap_pose

class FloorRobotFaultException(Exception):
    pass

class CeilingRobotFaultException(Exception):
    pass

class Command:
    def __init__(self):
        self.command_id = None
        self.robot_name = None
        self.type = None
        self.pick_part = Part()
        self.pick_part_on = None
        self.target_position = None
        self.is_done = True
        self.exe_time = 50
        self.start_time  = None
        self.task=None

    def __eq__(self,other):
        return self.command_id==other.command_id and self.robot_name==other.robot_name and self.type ==other.type and self.pick_part == other.pick_part and self.pick_part_on == other.pick_part_on \
            and self.target_position == other.target_position and self.is_done == other.is_done and self.task == other.task
            
def floor_fault_detector(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # 检查机器人故障，如果故障，等待恢复
        while not self.floor_robot_info.is_enabled:
            # print("等待中floor机器人的状态:", self.floor_robot_info.is_enabled)
            sleep(0.5)

        result = func(self, *args, **kwargs)

        # 再次检查机器人故障，如果故障，等待恢复，然后重新执行函数
        if not self.floor_robot_info.is_enabled:
            # print("执行结束后，floor机器人的状态:", self.floor_robot_info.is_enabled)
            while not self.floor_robot_info.is_enabled:
                # print("等待中floor机器人的状态:", self.floor_robot_info.is_enabled)
                sleep(0.5)
            result = func(self, *args, **kwargs)

        return result

    return wrapper

def ceiling_fault_detector(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # 检查机器人故障，如果故障，等待恢复
        while not self.ceiling_robot_info.is_enabled :
            # print("等待中ceiling机器人的状态:", self.ceiling_robot_info.is_enabled)
            sleep(0.5)

            
        result = func(self, *args, **kwargs)

        # 再次检查机器人故障，如果故障，等待恢复，然后重新执行函数
        if not self.ceiling_robot_info.is_enabled :
            # print("执行结束后，ceiling机器人的状态:", self.ceiling_robot_info.is_enabled)
            while not self.ceiling_robot_info.is_enabled :
                # print("等待中ceiling机器人的状态:", self.ceiling_robot_info.is_enabled)
                sleep(0.5)
            result = func(self, *args, **kwargs)

        return result

    return wrapper

def update_ceiling_status_decorator(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        self.ceiling_robot_info.is_idle = False
        result = func(self, *args, **kwargs)
        self.ceiling_robot_info.is_idle = True
        return result
    return wrapper

def update_floor_status_decorator(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        self.floor_robot_info.is_idle = False
        result = func(self, *args, **kwargs)
        self.floor_robot_info.is_idle = True
        return result
    return wrapper

class SIAInterface(Node):
    # Dictionary to convert competition_state constants to strings
    states_ = {
        CompetitionState.IDLE: 'idle',
        CompetitionState.READY: 'ready',
        CompetitionState.STARTED: 'started',
        CompetitionState.ORDER_ANNOUNCEMENTS_DONE: 'order_announcements_done',
        CompetitionState.ENDED: 'ended',
    }

    gripper_states_ = {
        True: 'enabled',
        False: 'disabled'
    }
   
    def __init__(self):
        super().__init__('competition_node')

        sim_time = Parameter(
            "use_sim_time",
            rclpy.Parameter.Type.BOOL,
            True
        )

        self.set_parameters([sim_time])

        self.competition_state = None

        self.spin_lock = threading.Lock()
        
        self.subscription = self.create_subscription(
            CompetitionStateMsg, 
            '/ariac/competition_state',
            self.competition_state_cb,
            10)
        
        self.starter = self.create_client(Trigger, '/ariac/start_competition')
        
        self.pre_assembly_poses_getter = self.create_client(GetPreAssemblyPoses, '/ariac/get_pre_assembly_poses')
        
        self.floor_robot_fault_triggered = False
        self.floor_robot_fault_event = threading.Event()
        self.ceiling_robot_fault_triggered = False
        self.ceiling_robot_fault_event = threading.Event()
        
        self.floor_has_flip=False
        
        self.timer = self.create_timer(350.0, self.timer_submit_callback)  # 创建定时器，300秒后强制提交订单

        self.convey_parts=[]
        self.convey_parts_lock = threading.Lock()
        self.pre_length=0

    ## ###################################### Order  ###############################################################################
            # 定义子任务列表
        self.kitting_subtasks = []
        self.assembly_subtasks = []
        self.combined_subtasks = []
        self.order_listsub_ = self.create_subscription(OrderMsg,'/ariac/orders',self.process_orders,10)
        self.order_listsub_       # 防止被python垃圾回收机制自动释放
        
        self.client = self.create_client(SubmitOrder, '/ariac/submit_order')
        self.quality_checker = self.create_client(PerformQualityCheck, '/ariac/perform_quality_check')
        self.orders = {"kitting_subtasks": deque(), "assembly_subtasks": deque(), "combined_subtasks": deque()}
        self.kitting_deque=deque()
        self.assembly_deque=deque()
        self.combined_deque=deque()
        self.agv_has_tray=[False,False,False,False]
        self.floor_Drop=False

        # 处理assembly任务
        self.assembly_arrived=False
        self.assembly_agvs=[]
        self.assembly_agvs_parts_poses=None
        self.assembly_order=None
        self.assembly_destion=None
        
        self.combined_products=deque()
        self.combined_floor_flag=True
        self.combined_agvs_parts_poses=None
        
        self.combined_tasks=[]
        
        self.co_tray_flag={}
        
        self.orders_list=[]
        self.orders_wait_list=[]
        self.order_recored_list=[]       # 记录所有订单
        
        self.last_order=None  

        self.last_ass_kitting_cmd=None
        self.last_ass_kitting_cmd_conveyor=None
        self.combine_ass_task=[]
        self.combine_ass_task_con=[]
        
        self.task_executor = ThreadPoolExecutor(max_workers=2)
        self.combined_to_floor_cmd=[]
        self.floor_to_ceiling_cmd=[]
        self.ceiling_cmd_list=[]
        
        self.order_pose={}
        self.oredr_length={}
        self.combined_spilt={}
        self.combined_spilt_assembly={}
        
        self.ceiling_kitting_once=True
        

    ## ###################################### Floor Robot  ############################################################################### 
    
        self.floor_robot_info = Robot_Info("floor_robot")
        self.kitting_robot_idle_time = self.get_clock().now().nanoseconds/1e9
        self.kitting_arm_joint_state_subscriber = self.create_subscription(JointTrajectoryControllerState, '/floor_robot_controller/controller_state', self.kitting_arm_joint_state_callback, qos_profile_sensor_data)
        self.linear_joint_state_subscriber = self.create_subscription(JointTrajectoryControllerState, '/linear_rail_controller/controller_state', self.linear_joint_state_callback, qos_profile_sensor_data)
        self.robot_health_subscriber = self.create_subscription(Robots, '/ariac/robot_health', self.robot_health_state_callback, qos_profile_sensor_data)
        self.floor_action_client=ActionClient(self, FollowJointTrajectory, '/floor_robot_controller/follow_joint_trajectory')
        self.linear_action_client=ActionClient(self, FollowJointTrajectory, '/linear_rail_controller/follow_joint_trajectory')
        

        self.human_distance=100
        self.min_distance=0
        
        self.kitting_arm_joint_states=None
        self.kitting_arm_joint_names = [
            'floor_shoulder_pan_joint','floor_shoulder_lift_joint',  'floor_elbow_joint',
            'floor_wrist_1_joint', 'floor_wrist_2_joint','floor_wrist_3_joint'
        ]
        self.linear_joint_names = ['linear_actuator_joint']
        self.kitting_typical_joints = {
            "init_state" : [1.4820999965423397-pi/2, -1.6356888311746864, 1.9210404979505746, -1.8276216909939889, -1.5708049327960403, -3.1302637783910976],
            
            "standby" : [1.5820963319369694, -1.6356888311746314, 1.921040497950596, -1.8276216909939276, -1.5708049327979872, -3.1302637783918614],
            "bin_agv_insert_joint": [-9.908213582932035e-06, -1.6356881698442969, 1.9210396904708134, 4.434232672827393, -1.570804295066237, -3.130262516117118],
            "flip_init_state" : [3.139979598681924, -1.0823125299292018, 1.7835319716553002, 5.542528446925819, -1.4273425719694686 - pi/2, -3.1399976775082745],
            "conveyor_insert": [1.4820999965423397+pi/2, -1.6356888311746864, 1.9210404979505746, -1.8276216909939889, -1.5708049327960403, -3.1302637783910976],

        }  
        self.kitting_base_x = -1.30 
        self.kitting_base_y = 0
        self.kitting_base_z = 0.93 
        # self.gripper = GripperManager(ns='floor')
  
        self.init_matrix = numpy.matrix([[0.00000000e+00, -1.00000000e+00, 0.00000000e+00, 0.171000000e+00],
                                        [0.00000000e+00,  0.00000000e+00,  1.00000000e+00, -0.62300000e+00],
                                        [-1.00000000e+00, 0.00000000e+00,  0.00000000e+00, 0.500000e+00],
                                        [0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]])


        self.conveyor_insert_matrix = numpy.matrix([[0.00000000e+00, -1.00000000e+00, 0.00000000e+00, -0.612],
                                        [0.00000000e+00,  0.00000000e+00,  1.00000000e+00, -0.209],
                                        [-1.00000000e+00, 0.00000000e+00,  0.00000000e+00, 0.500000e+00],
                                        [0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]])

        self.bin_agv_insert_matrix = numpy.matrix([[0.00000000e+00, -1.00000000e+00, 0.00000000e+00, 0.626],
                                        [0.00000000e+00,  0.00000000e+00,  1.00000000e+00, 0.163],
                                        [-1.00000000e+00, 0.00000000e+00,  0.00000000e+00, 0.500000e+00],
                                        [0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]])
    
        self.conveyor_insert_joints =[1.8999203692233948,-1.6415419492462364, 0.07279109918787174,4.457661400146632, -1.5708109330044402, -4.6395942621955575]
        self.init_rotation = numpy.matrix([[0.00000000e+00, -1.00000000e+00, 0.00000000e+00],
                                        [0.00000000e+00,  0.00000000e+00,  1.00000000e+00],
                                        [-1.00000000e+00, 0.00000000e+00,  0.00000000e+00]])
        self.rotate_rotation = numpy.matrix([[-1.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                                        [0.00000000e+00,  -1.00000000e+00,  0.00000000e+00],
                                        [0.00000000e+00, 0.00000000e+00,  1.00000000e+00]])
        self.flip_rotation = {
                'left_roll_0' :  np.matrix([[0.00000000e+00, 1.00000000e+00, 0.00000000e+00],
                                           [1.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                                           [0.00000000e+00, 0.00000000e+00,  -1.00000000e+00]]),
                'left_roll_pi' :  np.matrix([[0.00000000e+00, 1.00000000e+00, 0.00000000e+00],
                                           [-1.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                                           [0.00000000e+00, 0.00000000e+00,  1.00000000e+00]]),
                'right_roll_0' :  np.matrix([[0.00000000e+00, -1.00000000e+00, 0.00000000e+00],
                                           [-1.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                                           [0.00000000e+00, 0.00000000e+00,  -1.00000000e+00]]),        
                'right_roll_pi' :  np.matrix([[0.00000000e+00, -1.00000000e+00, 0.00000000e+00],
                                           [1.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                                           [0.00000000e+00, 0.00000000e+00,  1.00000000e+00]]),        
        }
        
        self.floor_robot_gripper_state = VacuumGripperState()

        self.floor_gripper_enable = self.create_client(VacuumGripperControl, 
                                                       "/ariac/floor_robot_enable_gripper")
        
        self.floor_robot_gripper_state_sub = self.create_subscription(VacuumGripperState, 
                                                                      '/ariac/floor_robot_gripper_state', 
                                                                      self.floor_robot_gripper_state_cb, 
                                                                      qos_profile_sensor_data)
        self.floor_robot_tool_changer_ = self.create_client(ChangeGripper,'/ariac/floor_robot_change_gripper')
        
        # Setup TF listener
        self.tf_buffer = Buffer()
        self.tf_buffer_floor = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_listener_floor = TransformListener(self.tf_buffer_floor, self)
        self.goals = {}
        self.bins_dict={}
        
        self.parts_complete =False
        
    ## ###################################### Ceiling Robot  ############################################################################### 
        self.ceiling_robot_info = Robot_Info("ceiling_Robot")
        self.Ceiling_robot_idle_time = self.get_clock().now().nanoseconds/1e9
     
        self.Ceiling_robot_arm_joint_state_subscriber = self.create_subscription(JointTrajectoryControllerState, '/ceiling_robot_controller/controller_state', self.ceiling_arm_joint_state_callback, qos_profile_sensor_data)
        self.gantry_state_subscriber = self.create_subscription(JointTrajectoryControllerState, '/gantry_controller/controller_state', self.gantry_state_callback, qos_profile_sensor_data)
        
        self.ceiling_action_client=ActionClient(self, FollowJointTrajectory, '/ceiling_robot_controller/follow_joint_trajectory')
        self.gantry_action_client=ActionClient(self, FollowJointTrajectory, '/gantry_controller/follow_joint_trajectory')

        self.ceiling_robot=False 
        
        self.ceiling_arm_joint_states=None
        
        self.ceiling_base_x = 0.0 
        self.ceiling_base_y = 0.0 
        self.ceiling_base_r = 0.0 

        self.pick_part_theta = 0.0 
        self.pick_torso_theta = 0.0 

        self.ceiling_torso_joint_names=[
            'gantry_x_axis_joint',  # x
            'gantry_rotation_joint', # r
            'gantry_y_axis_joint',  # y
            
           
        ]

        self.ceiling_torso_joints = None

        self.ceiling_arm_joint_names = [
            'ceiling_shoulder_pan_joint', 'ceiling_shoulder_lift_joint', 
            'ceiling_elbow_joint', 'ceiling_wrist_1_joint', 
            'ceiling_wrist_2_joint','ceiling_wrist_3_joint'
        ]
        self.ceiling_arm_normal_joints = None

        self.ceiling_init_position = [1.00,0.00, 0.00]

        self.ceiling_arm_init_position = [-3.179252150056005, -0.518062910410503, -1.9965621378670235, -0.630055723042239, 1.6116997462510554, -3.1415841107959963]

        self.init_matrix = numpy.matrix([[1.00,  0.00, 0.00, 0.017],
                                        [0.00,  -1.00, 0.00, -0.161],
                                        [0.00,  0.00, -1.00, 0.882],
                                        [0.00,  0.00, 0.00, 1.00]])

        self.ceiling_init_rotation = numpy.matrix([[1.00, 0.00, 0.00],
                                           [0.00, -1.00, 0.00],
                                           [0.00, 0.00, -1.00]]) 


        ##################gantry kitting姿态###############################
        self.ceiling_arm_kitting_position = [-0.019255781552382167, -2.11715423544646, 2.260508026291504,-0.14503290876189112, 1.5549932297936397, -3.141583271144321]

        self.ceiling_arm_kitting_matrix = numpy.matrix([[1.00,  0.00, 0.00, 0.343],
                                                    [0.00,  1.00, 0.00, 0.160],
                                                    [0.00,  0.00, 1.00, 0.453],
                                                    [0.00,  0.00, 0.00, 1.00]])

        self.ceiling_arm_kitting_rotation = numpy.matrix([[1.00, 0.00, 0.00],
                                                        [0.00, 1.00, 0.00],
                                                        [0.00, 0.00, 1.00]])
        self.rotate_rotation = numpy.matrix([[-1.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                                        [0.00000000e+00,  -1.00000000e+00,  0.00000000e+00],
                                        [0.00000000e+00, 0.00000000e+00,  1.00000000e+00]])

        self.rotate_Y_90= numpy.matrix([[0.00000000e+00, 0.00000000e+00, 1.00000000e+00],
                                        [0.00000000e+00,  1.00000000e+00,  0.00000000e+00],
                                        [-1.00000000e+00, 0.00000000e+00,  0.00000000e+00]])

        self.rotate_X_90= numpy.matrix([[1.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                                        [0.00000000e+00,  0.00000000e+00,  -1.00000000e+00],
                                        [0.00000000e+00, 1.00000000e+00,  0.00000000e+00]])

        self.flip_rotation = {
                'left_roll_0' :  np.matrix([[0.00000000e+00, 1.00000000e+00, 0.00000000e+00],
                                           [1.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                                           [0.00000000e+00, 0.00000000e+00,  -1.00000000e+00]]),
                'left_roll_pi' :  np.matrix([[0.00000000e+00, 1.00000000e+00, 0.00000000e+00],
                                           [-1.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                                           [0.00000000e+00, 0.00000000e+00,  1.00000000e+00]]),
                'right_roll_0' :  np.matrix([[0.00000000e+00, -1.00000000e+00, 0.00000000e+00],
                                           [-1.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                                           [0.00000000e+00, 0.00000000e+00,  -1.00000000e+00]]),        
                'right_roll_pi' :  np.matrix([[0.00000000e+00, -1.00000000e+00, 0.00000000e+00],
                                           [1.00000000e+00,  0.00000000e+00,  0.00000000e+00],
                                           [0.00000000e+00, 0.00000000e+00,  1.00000000e+00]]),        
        }
        ##############gantry 装配姿态#################
        # self.gantry_arm_assemble_position = [-1.9965621378670235, -0.518062910410503, -3.179252150056005, -0.630055723042239, 1.6116997462510554, 3.1415841107959963]

        # self.gantry_arm_assemble_matrix = numpy.matrix([[1.00,  0.00, 0.00, 0.017],
        #                                                 [0.00,  -1.00, 0.00, -0.161],
        #                                                 [0.00,  0.00, -1.00, 0.882],
        #                                                 [0.00,  0.00, 0.00, 1.00]])

        self.ceiling_arm_assemble_rotation = numpy.matrix([[1.00, 0.00, 0.00],
                                                        [0.00, -1.00, 0.00],
                                                        [0.00, 0.00, -1.00]])    
                                                         
        self.ceiling_arm_assemble_position = [-3.172421427905949, -1.664570820415968, -1.4746002176069055,0.3932066531825935, 1.6052786119583917, 0.000001]
        self.ceiling_arm_assemble_matrix = numpy.matrix([[1.00,  0.00, 0.00, 0.903],
                                                    [0.00,  -1.00, 0.00, -0.192],
                                                    [0.00,  0.00, -1.00, 0.562],
                                                    [0.00,  0.00, 0.00, 1.00]])    

                                                    
        self.ceiling_arm_flip_init_position = [-2.9508940961720214, -2.3241341588739353, -1.4187019275568016, -0.9695412285024076, 1.5707626896820814, -0.0000]   

        self.ceiling_robot_gripper_state = VacuumGripperState()

        self.ceiling_gripper_enable = self.create_client(VacuumGripperControl, 
                                                       "/ariac/ceiling_robot_enable_gripper")
        
        self.ceiling_robot_gripper_state_sub = self.create_subscription(VacuumGripperState, 
                                                                      '/ariac/ceiling_robot_gripper_state', 
                                                                      self.ceiling_robot_gripper_state_cb, 
                                                                      qos_profile_sensor_data)
        self.ceiling_robot_tool_changer_ = self.create_client(ChangeGripper,'/ariac/ceiling_robot_change_gripper')                                                            

    ## ###################################### AGV  ############################################################################### 

        self.agv1_status_subscriber = self.create_subscription(AGVStatus, '/ariac/agv1_status', self.agv1_status_callback, qos_profile_sensor_data)
        self.agv2_status_subscriber = self.create_subscription(AGVStatus, '/ariac/agv2_status', self.agv2_status_callback, qos_profile_sensor_data)
        self.agv3_status_subscriber = self.create_subscription(AGVStatus, '/ariac/agv3_status', self.agv3_status_callback, qos_profile_sensor_data)
        self.agv4_status_subscriber = self.create_subscription(AGVStatus, '/ariac/agv4_status', self.agv4_status_callback, qos_profile_sensor_data)
        self.agv1_position=0
        self.agv2_position=0
        self.agv3_position=0
        self.agv4_position=0

#region  ###################################### Sensor  ###############################################################################      
        self.bridge = CvBridge()
        self.rgbd_kts1_image = None
        self.rgbd_kts2_image = None
         
        self.break_beam_health=True
        self.proximity_health=True
        self.laser_profiler_health=True
        self.lidar_health=True
        self.camera_health=True
        self.logical_camera_health=True
        
        self.bin1_6_need_clean=False
        self.bin1_6_part=None
        self.bin6_6_need_clean=False
        self.bin6_6_part=None
        
        self.bin1_grid_status = {(i, j): False for i in range(3) for j in range(3)}
        self.bin6_grid_status = {(i, j): False for i in range(3) for j in range(3)}
        self.spacing = 0.18
       
        self.AGV_location={
            'agv1': 'agv1_ks1_tray',
            'agv2': 'agv2_ks2_tray',
            'agv3': 'agv3_ks3_tray',
            'agv4': 'agv4_ks4_tray',
        }
        self.logical_camera_conveyor_parts = []

        self.tray_slots={}

        self.tray_1_slots = {
            "slot1": None,
            "slot2": None,
            "slot3": None,
        }
        self.tray_2_slots = {
            "slot4": None,
            "slot5": None,
            "slot6": None
        }
        
        self.has_blocked = False
        self.has_blocked_for_check = False
        
        self.tray_table_1 = []
        self.tray_table_2 = []

        #logical_camera_0_parts
        self.bin1_parts = []
        self.bin4_parts = []
        self.agv1_ks1_tray_parts = []

        #logical_camera_1_parts
        self.bin2_parts = []
        self.bin3_parts = []
        self.agv2_ks2_tray_parts = []

        #logical_camera_2_parts
        self.bin6_parts = []
        self.bin7_parts = []
        self.agv3_ks3_tray_parts = []

        #logical_camera_3_parts
        self.bin5_parts = []
        self.bin8_parts = []
        self.agv4_ks4_tray_parts = []

        self.new_part_dict = {
            'agv1_ks1_tray':None,
            'agv2_ks2_tray':None,
            'agv3_ks3_tray':None,
            'agv4_ks4_tray':None,

            'agv1_as1_tray':None,
            'agv2_as1_tray':None,
            'agv3_as3_tray':None,
            'agv4_as3_tray':None,

            'agv1_as2_tray':None,
            'agv2_as2_tray':None,
            'agv3_as4_tray':None,
            'agv4_as4_tray':None,
            'bin1':None,
            'bin2':None,
            'bin3':None,
            'bin4':None,
            'bin5':None,
            'bin6':None,
            'bin7':None,
            'bin8':None,
            'conveyor':None,
        }

        self.new_part_flag_dict = {
            'agv1_ks1_tray':False,
            'agv2_ks2_tray':False,
            'agv3_ks3_tray':False,
            'agv4_ks4_tray':False,

            'agv1_as1_tray':False,
            'agv2_as1_tray':False,
            'agv3_as3_tray':False,
            'agv4_as3_tray':False,

            'agv1_as2_tray':False,
            'agv2_as2_tray':False,
            'agv3_as4_tray':False,
            'agv4_as4_tray':False,

            'bin1':False,
            'bin2':False,
            'bin3':False,
            'bin4':False,
            'bin5':False,
            'bin6':False,
            'bin7':False,
            'bin8':False,
            'conveyor':False,
        }

        self.rgb_bins_list={
                'bin1_1': None, 'bin1_2': None, 'bin1_3': None, 'bin1_4': None, 'bin1_5': None, 'bin1_6': None, 'bin1_7': None, 'bin1_8': None, 'bin1_9': None, 
                 'bin2_1': None, 'bin2_2': None, 'bin2_3': None, 'bin2_4': None, 'bin2_5': None, 'bin2_6': None, 'bin2_7': None, 'bin2_8': None, 'bin2_9': None, 
                 'bin3_1': None, 'bin3_2': None, 'bin3_3': None, 'bin3_4': None, 'bin3_5': None, 'bin3_6': None, 'bin3_7': None, 'bin3_8': None, 'bin3_9': None, 
                 'bin4_1': None, 'bin4_2': None, 'bin4_3': None, 'bin4_4': None, 'bin4_5': None, 'bin4_6': None, 'bin4_7': None, 'bin4_8': None, 'bin4_9': None, 
                 'bin5_1': None, 'bin5_2': None, 'bin5_3': None, 'bin5_4': None, 'bin5_5': None, 'bin5_6': None, 'bin5_7': None, 'bin5_8': None, 'bin5_9': None, 
                 'bin6_1': None, 'bin6_2': None, 'bin6_3': None, 'bin6_4': None, 'bin6_5': None, 'bin6_6': None, 'bin6_7': None, 'bin6_8': None, 'bin6_9': None, 
                 'bin7_1': None, 'bin7_2': None, 'bin7_3': None, 'bin7_4': None, 'bin7_5': None, 'bin7_6': None, 'bin7_7': None, 'bin7_8': None, 'bin7_9': None,
                 'bin8_1': None, 'bin8_2': None, 'bin8_3': None, 'bin8_4': None, 'bin8_5': None, 'bin8_6': None, 'bin8_7': None, 'bin8_8': None, 'bin8_9': None
        }

        self.logical_camera_as_11_parts = []
        self.logical_camera_as_12_parts = []
        self.logical_camera_as_21_parts = []
        self.logical_camera_as_22_parts = []
        self.logical_camera_as_33_parts = []
        self.logical_camera_as_34_parts = []
        self.logical_camera_as_43_parts = []
        self.logical_camera_as_44_parts = []

        self.logical_camera_conveyor_parts = []
        self.logical_camera_update_rate = 0.1
        
        self.heart_beat = self.get_clock().now()
        self.heart_beat = self.heart_beat.nanoseconds/1e9

        self.u_id_count = 0
        self.AGV_state={
            'agv1': 'init',
            'agv2': 'init',
            'agv3': 'init',
            'agv4': 'init',
        }


        self.assembly_parts = []
        self.conveyor_parts = []
        self.parts_type_dict={}
        self.parts_location_dict={}
        self.parts_on_conveyor_dict={}

        self.kts1_camera_flag = False
        self.kts2_camera_flag = False

        self.update_flag   = False
        self.camera_0_flag = False
        self.camera_1_flag = False
        self.camera_2_flag = False
        self.camera_3_flag = False
        self.camera_4_flag = False

        self.camera_as_11_flag = False
        self.camera_as_12_flag = False
        self.camera_as_21_flag = False
        self.camera_as_22_flag = False
        self.camera_as_33_flag = False
        self.camera_as_34_flag = False
        self.camera_as_43_flag = False
        self.camera_as_44_flag = False

        self.AGV1_location_flag = False
        self.AGV2_location_flag = False
        self.AGV3_location_flag = False
        self.AGV4_location_flag = False

        self.AGV1_state_flag = False
        self.AGV2_state_flag = False
        self.AGV3_state_flag = False
        self.AGV4_state_flag = False
       # Store RGB images from the right bins camera so they can be used for part detection(0403)
        self._right_bins_camera_image = None
        self._left_bins_camera_image = None
        # Turn on debug image publishing for part detection start(0403)
        self.display_bounding_boxes =  True
        # Turn on debug image publishing for part detection end(0403)
        # cv_bridge interface start(0403)
        self._bridge = CvBridge()
        # ---0412 start---
        # HSV colour bounds
        self.HSVcolors = {
            "red"    : {"hmin":   0, "smin":  10, "vmin": 115, "hmax":   4, "smax": 255, "vmax": 255},
            "green"  : {"hmin":  57, "smin":   0, "vmin":   0, "hmax":  80, "smax": 255, "vmax": 255},
            "blue"   : {"hmin": 116, "smin":   0, "vmin": 134, "hmax": 121, "smax": 255, "vmax": 255},
            "orange" : {"hmin":  14, "smin":   0, "vmin": 200, "hmax":  21, "smax": 255, "vmax": 255},
            "purple" : {"hmin": 130, "smin": 180, "vmin": 160, "hmax": 150, "smax": 255, "vmax": 255}
        }

        # BGR reference colours
        self.colors = {
            "red"    : (  0,   0, 255),
            "green"  : (  0, 255,   0),
            "blue"   : (255,   0,   0),
            "orange" : (100, 100, 255),
            "purple" : (255,   0, 100)
        }

        # Part Pose Reporting Object
        self.part_poses = {
            "red"    : {"battery": [], "pump": [], "sensor": [], "regulator": []},
            "green"  : {"battery": [], "pump": [], "sensor": [], "regulator": []},
            "blue"   : {"battery": [], "pump": [], "sensor": [], "regulator": []},
            "orange" : {"battery": [], "pump": [], "sensor": [], "regulator": []},
            "purple" : {"battery": [], "pump": [], "sensor": [], "regulator": []}
        }
        # Center of Part Poses
        self.centered_part_poses = {
            "red"    : {"battery": [], "pump": [], "sensor": [], "regulator": []},
            "green"  : {"battery": [], "pump": [], "sensor": [], "regulator": []},
            "blue"   : {"battery": [], "pump": [], "sensor": [], "regulator": []},
            "orange" : {"battery": [], "pump": [], "sensor": [], "regulator": []},
            "purple" : {"battery": [], "pump": [], "sensor": [], "regulator": []}
        }

        self.slot_mapping = {
            (1, 1): 1,
            (1, 2): 2,
            (1, 3): 3,
            (2, 1): 4,
            (2, 2): 5,
            (2, 3): 6,
            (3, 1): 7,
            (3, 2): 8,
            (3, 3): 9,
        }

        self.color_mapping = {
            "red"    : 0,
            "green"  : 1,
            "blue"   : 2,
            "orange" : 3,
            "purple" : 4
        }

        self.type_mapping = {
            "battery"   : 10,
            "pump"      : 11,
            "sensor"    : 12,
            "regulator" : 13
        }
        # ----0412 end---
                # read in part templates
        self.load_part_templates()
        self.parts_on_conveyor_dict={}

        self.threadLock = threading.Lock()
        
        self.task_threadLock = threading.Lock()
        
        qos_profile = qos_profile_sensor_data
        
        self.kts1_rgb_camera = self.create_subscription(Image,'/ariac/sensors/kts1_camera/rgb_image',self.rgbd_kts1_image_callback,qos_profile_sensor_data)
        self.kts2_rgb_camera = self.create_subscription(Image,'/ariac/sensors/kts2_camera/rgb_image',self.rgbd_kts2_image_callback,qos_profile_sensor_data)
        # 使用高级逻辑相机直接识别零件，保持与 competition_interface 对齐
        self.logical_camera_0_sub_ = self.create_subscription(AdvancedLogicalCameraImage,'/ariac/sensors/logical_camera_0/image',self.logical_camera_0_callback,qos_profile_sensor_data)
        self.logical_camera_1_sub_ = self.create_subscription(AdvancedLogicalCameraImage,'/ariac/sensors/logical_camera_1/image',self.logical_camera_1_callback,qos_profile_sensor_data)
        self.logical_camera_conveyor_sub_ = self.create_subscription(AdvancedLogicalCameraImage,'/ariac/sensors/logical_camera_conveyor/image',self.logical_camera_conveyor_callback,qos_profile_sensor_data)
        # 关闭基础逻辑相机与模板匹配 RGB 订阅，避免重复或冲突的识别方式
        # self.right_basic_logical_camera_sub_ = self.create_subscription(BasicLogicalCameraImage,'/ariac/sensors/right_basic_logical_camera/image',self.right_basic_logical_camera_callback,qos_profile_sensor_data)
        # self.left_basic_logical_camera_sub_ = self.create_subscription(BasicLogicalCameraImage,'/ariac/sensors/left_basic_logical_camera/image',self.left_basic_logical_camera_callback,qos_profile_sensor_data)
        # self.right_bins_RGB_camera_sub = self.create_subscription(ImageMsg,"/ariac/sensors/right_bins_RGB_camera/rgb_image",self._right_bins_RGB_camera_cb,qos_profile_sensor_data)
        # self.left_bins_RGB_camera_sub = self.create_subscription(ImageMsg,"/ariac/sensors/left_bins_RGB_camera/rgb_image",self._left_bins_RGB_camera_cb,qos_profile_sensor_data)
        # self.display_bounding_boxes_pub = self.create_publisher(ImageMsg,"/ariac/sensors/display_bounding_boxes",qos_profile_sensor_data)

        self.AGV1_status_sub_ = self.create_subscription(AGVStatus, "/ariac/agv1_status", self.AGV1_status_callback,qos_profile_sensor_data)
        self.AGV2_status_sub_ = self.create_subscription(AGVStatus, "/ariac/agv2_status", self.AGV2_status_callback,qos_profile_sensor_data)
        self.AGV3_status_sub_ = self.create_subscription(AGVStatus, "/ariac/agv3_status", self.AGV3_status_callback,qos_profile_sensor_data)
        self.AGV4_status_sub_ = self.create_subscription(AGVStatus, "/ariac/agv4_status", self.AGV4_status_callback,qos_profile_sensor_data)
        
        self.sensor_health_subscriber = self.create_subscription(Sensors, '/ariac/sensor_health', self.sensor_health_state_callback, qos_profile_sensor_data)

    #  basic logical camera callback 0413 
    def left_basic_logical_camera_callback(self,msg : BasicLogicalCameraImage):
        self.heart_beat = self.get_clock().now()
        self.heart_beat=self.heart_beat.nanoseconds/1e9      
        self.camera_1_flag = True
        if self.camera_1_flag:
            # print("识别到左边有:",len(msg.part_poses),"个零件")
            for model in msg.part_poses:
                part_pose = Pose()              # part to world
                part_pose.position.x = msg.sensor_pose.position.x-model.position.z
                part_pose.position.y = msg.sensor_pose.position.y-model.position.y
                # part_pose.position.x = msg.sensor_pose.position.x+model.position.z
                # part_pose.position.y = msg.sensor_pose.position.y+model.position.y
                part_pose.position.z = msg.sensor_pose.position.z-model.position.x
                p = quaternion_multiply( \
                    [msg.sensor_pose.orientation.x, msg.sensor_pose.orientation.y,msg.sensor_pose.orientation.z,msg.sensor_pose.orientation.w],  \
                    [model.orientation.x,model.orientation.y,model.orientation.z,model.orientation.w])
                part_pose.orientation.x = p[0]
                part_pose.orientation.y = p[1]
                part_pose.orientation.z = p[2]
                part_pose.orientation.w = p[3]  
                
                # print("model_pose",model.position)
                # print("sensor_pose",msg.sensor_pose)
                # print("part_pose",part_pose.position)
                bin_slot = self.find_closest_slot(part_pose.position,bin_slot_positions)
                # print("bin6有槽:",bin_slot)
                part= self.rgb_bins_list[bin_slot]
                if part:
                    part.pose= part_pose
                    if  Is_In_Effective_Range(part.type, part_pose.position, bins_ks_boundary["bin5_x"] ,bins_ks_boundary["bin5_y"], bins_product_height_flip):
                        self.parts_lsit_update(part, self.bin5_parts)

                    if  Is_In_Effective_Range(part.type, part_pose.position, bins_ks_boundary["bin6_x"] ,bins_ks_boundary["bin6_y"], bins_product_height_flip):
                        self.parts_lsit_update(part, self.bin6_parts)

                    if  Is_In_Effective_Range(part.type, part_pose.position, bins_ks_boundary["bin7_x"] ,bins_ks_boundary["bin7_y"], bins_product_height_flip):
                        self.parts_lsit_update(part, self.bin7_parts)

                    if  Is_In_Effective_Range(part.type, part_pose.position, bins_ks_boundary["bin8_x"] ,bins_ks_boundary["bin8_y"], bins_product_height_flip):
                        self.parts_lsit_update(part, self.bin8_parts)

            self.camera_1_flag =False
                    # print("part",part)
                    # print("part",part.location)
        
        # print('bin6 has:::',self.bin6_parts)
    
    def right_basic_logical_camera_callback(self,msg : BasicLogicalCameraImage):
        self.heart_beat = self.get_clock().now()
        self.heart_beat=self.heart_beat.nanoseconds/1e9      
        self.camera_0_flag = True
        if self.camera_0_flag:
            # print("识别到右边有:",len(msg.part_poses),"个零件")
            # print('此时bin2中有:',self.bin2_parts)
            for model in msg.part_poses:
                part_pose = Pose()              # part to world
                part_pose.position.x = msg.sensor_pose.position.x-model.position.z
                part_pose.position.y = msg.sensor_pose.position.y-model.position.y
                # part_pose.position.x = msg.sensor_pose.position.x+model.position.z
                # part_pose.position.y = msg.sensor_pose.position.y+model.position.y
                part_pose.position.z = msg.sensor_pose.position.z-model.position.x
                p = quaternion_multiply( \
                    [msg.sensor_pose.orientation.x, msg.sensor_pose.orientation.y,msg.sensor_pose.orientation.z,msg.sensor_pose.orientation.w],  \
                    [model.orientation.x,model.orientation.y,model.orientation.z,model.orientation.w])
                part_pose.orientation.x = p[0]
                part_pose.orientation.y = p[1]
                part_pose.orientation.z = p[2]
                part_pose.orientation.w = p[3]  
                
                # print("model_pose",model.position)
                # print("sensor_pose",msg.sensor_pose)
                # print("part_pose",part_pose.position)
                bin_slot = self.find_closest_slot(part_pose.position,bin_slot_positions)
                # print("有槽:",bin_slot)
                part= self.rgb_bins_list[bin_slot]
                if not part:        # RGB没有检测到这个，说明是pump
                    bin_key=self.find_closest_slot(part_pose.position,bin_position)
                    # print("应该是",bin_key)
                    part = sPart('pump',bin_key,part_pose,False)
                if part:
                    part.pose= part_pose
                    part.need_flip= determine_part_is_need_flip(part.type,part_pose)
                    if  Is_In_Effective_Range(part.type, part_pose.position, bins_ks_boundary["bin1_x"] ,bins_ks_boundary["bin1_y"], bins_product_height_flip):
                        self.parts_lsit_update(part, self.bin1_parts)

                    elif  Is_In_Effective_Range(part.type, part_pose.position, bins_ks_boundary["bin4_x"] ,bins_ks_boundary["bin4_y"], bins_product_height_flip):
                        self.parts_lsit_update(part, self.bin4_parts)

                    elif  Is_In_Effective_Range(part.type, part_pose.position, bins_ks_boundary["bin2_x"] ,bins_ks_boundary["bin2_y"], bins_product_height_flip):
                        self.parts_lsit_update(part, self.bin2_parts)

                    elif  Is_In_Effective_Range(part.type, part_pose.position, bins_ks_boundary["bin3_x"] ,bins_ks_boundary["bin3_y"], bins_product_height_flip):
                        self.parts_lsit_update(part, self.bin3_parts)

            self.camera_0_flag =False
        #             print("part",part)
        #             print("part",part.location)
        
        # print('bin2 has:::',self.bin2_parts)
        # for part in self.bin2_parts:
        #     print("零件的位置是:",part.pose.position)

    def find_closest_slot(self,part_position, slot_positions):
        from math import sqrt

        min_distance = float('inf')
        closest_slot = None

        # 遍历所有槽位和它们的坐标
        for slot, position in slot_positions.items():
            # 计算欧几里得距离
            distance = sqrt((position[0] - part_position.x) ** 2 +
                            (position[1] - part_position.y) ** 2 +
                            (position[2] - part_position.z) ** 2)

            # 检查这个距离是否是目前最小的
            if distance < min_distance:
                min_distance = distance
                closest_slot = slot

        return closest_slot
    
# default format used by the RGB cameras is rgb8 start（0403）
    def _left_bins_RGB_camera_cb(self, msg: ImageMsg):
        try:
            self._left_bins_camera_image = self._bridge.imgmsg_to_cv2(msg, "bgr8")
            for i in range(5, 9):  # 从1到8的二进制编号
                bin_key = f'bin{i}'
                self.bins_dict[bin_key] = self.get_bin_parts(i)
                if not self.bins_dict[bin_key]:
                    break
                
                # print(self.bins_dict[bin_key])
                for slot,slot_part in self.bins_dict[bin_key].items():
                    if slot_part.type and slot_part.color:
                        # print(self.bins_dict[bin_key])
                        # print("yanse+++",slot_part.type,slot_part.color)
                        part_color_type=self.determine_part_name(slot_part.type,slot_part.color)
                        part_pose = Pose() 
                        bin_slot_key = f'bin{i}_{slot}'
                        part_pose.position.x = bin_slot_positions[bin_slot_key][0]
                        part_pose.position.y = bin_slot_positions[bin_slot_key][1]
                        part_pose.position.z = bin_slot_positions[bin_slot_key][2]
                        part_pose.orientation.x = 0.0
                        part_pose.orientation.y = 0.0
                        part_pose.orientation.z = 0.0
                        part_pose.orientation.w = 1.0
                        # print("part_color_type:",part_color_type)
                        need_flip = determine_part_is_need_flip(part_color_type,part_pose)
                        part = sPart(part_color_type,bin_key,part_pose,need_flip)
                        self.rgb_bins_list[bin_slot_key]=part

            # print(self.rgb_bins_list)
        except CvBridgeError as e:
            print(e)

    def _right_bins_RGB_camera_cb(self, msg: ImageMsg):
        try:
            self._right_bins_camera_image = self._bridge.imgmsg_to_cv2(msg, "bgr8")
            for i in range(1, 5):  # 从1到8的二进制编号
                bin_key = f'bin{i}'
                self.bins_dict[bin_key] = self.get_bin_parts(i)
                if not self.bins_dict[bin_key]:
                    break
                
                # print(self.bins_dict[bin_key])
                for slot,slot_part in self.bins_dict[bin_key].items():
                    if slot_part.type and slot_part.color:
                        # print(self.bins_dict[bin_key])
                        # print("yanse+++",slot_part.type,slot_part.color)
                        part_color_type=self.determine_part_name(slot_part.type,slot_part.color)
                        part_pose = Pose() 
                        bin_slot_key = f'bin{i}_{slot}'
                        part_pose.position.x = bin_slot_positions[bin_slot_key][0]
                        part_pose.position.y = bin_slot_positions[bin_slot_key][1]
                        part_pose.position.z = bin_slot_positions[bin_slot_key][2]
                        part_pose.orientation.x = 0.0
                        part_pose.orientation.y = 0.0
                        part_pose.orientation.z = 0.0
                        part_pose.orientation.w = 1.0
                        # print("part_color_type:",part_color_type)
                        need_flip = determine_part_is_need_flip(part_color_type,part_pose)
                        part = sPart(part_color_type,bin_key,part_pose,need_flip)
                        self.rgb_bins_list[bin_slot_key]=part

        except CvBridgeError as e:
            print(e)

# The goal of this stage is to take in an RGB camera image and return a list of parts in the image start（0403）
    def get_bin_parts(self, bin_number: int):
        if type(self._left_bins_camera_image) == type(np.ndarray([])) and \
           type(self._right_bins_camera_image) == type(np.ndarray([])):
            if bin_number > 4:
                cv_img = self._left_bins_camera_image
            else:
                cv_img = self._right_bins_camera_image
            imgH, imgW = cv_img.shape[:2]
            
            # roi based on bin number
            if bin_number == 1 or bin_number == 6:
                # bottom left
                cv_img = cv_img[imgH//2:, (imgW//2)+20:imgW-100]
            if bin_number == 2 or bin_number == 5:
                # bottom right
                cv_img = cv_img[imgH//2:, 100:(imgW//2)-20]
            if bin_number == 3 or bin_number == 8:
                # top left
                cv_img = cv_img[:imgH//2, 100:(imgW//2)-20]
            if bin_number == 4 or bin_number == 7:
                # top right
                cv_img = cv_img[:imgH//2, (imgW//2)+20:imgW-100]
            # find parts by colour in image
            self.find_parts(cv_img)

            # debug image view on /ariac/sensors/display_bounding_boxes
            if self.display_bounding_boxes:            
                ros_img = self._bridge.cv2_to_imgmsg(cv_img, "bgr8")
                self.display_bounding_boxes_pub.publish(ros_img)

            # print results
            return self.output_by_slot()

        else:
            # Images are not received, return None
            return None
# The goal of this stage is to take in an RGB camera image and return a list of parts in the image 
    def determine_part_name(self,part_type,part_color):
        part_color_type=None
        if part_color =="red":
            if part_type == "battery":
                part_color_type="assembly_battery_red"
                #print("part_color_type",part_color_type)
            elif part_type=="pump":
                part_color_type="assembly_pump_red"
                #print("part_color_type",part_color_type)
            elif part_type=="sensor":
                part_color_type="assembly_sensor_red"
                #print("part_color_type",part_color_type)
            elif part_type=="regulator":
                part_color_type="assembly_regulator_red"
                #print("part_color_type",part_color_type)
        elif  part_color =="green":
            if part_type == "battery":
                part_color_type="assembly_battery_green"
                #print("part_color_type",part_color_type)
            elif part_type=="pump":
                part_color_type="assembly_pump_green"
                #print("part_color_type",part_color_type)
            elif part_type=="sensor":
                part_color_type="assembly_sensor_green"
                #print("part_color_type",part_color_type)
            elif part_type=="regulator":
                part_color_type="assembly_regulator_green"
                #print("part_color_type",part_color_type)
        elif  part_color =="blue":
            if part_type == "battery":
                part_color_type="assembly_battery_blue"
                #print("part_color_type",part_color_type)
            elif part_type=="pump":
                part_color_type="assembly_pump_blue"
                #print("part_color_type",part_color_type)
            elif part_type=="sensor":
                part_color_type="assembly_sensor_blue"
                #print("part_color_type",part_color_type)
            elif part_type=="regulator":
                part_color_type="assembly_regulator_blue"
                #print("part_color_type",part_color_type)
        elif  part_color =="orange":
            if part_type == "battery":
                part_color_type="assembly_battery_orange"
                #print("part_color_type",part_color_type)
            elif part_type=="pump":
                part_color_type="assembly_pump_orange"
                #print("part_color_type",part_color_type)
            elif part_type=="sensor":
                part_color_type="assembly_sensor_orange"
                #print("part_color_type",part_color_type)
            elif part_type=="regulator":
                part_color_type="assembly_regulator_orange"
                #print("part_color_type",part_color_type)
        elif  part_color =="purple":
            if part_type == "battery":
                part_color_type="assembly_battery_purple"
                #print("part_color_type",part_color_type)
            elif part_type=="pump":
                part_color_type="assembly_pump_purple"
                #print("part_color_type",part_color_type)
            elif part_type=="sensor":
                part_color_type="assembly_sensor_purple"
                #print("part_color_type",part_color_type)
            elif part_type=="regulator":
                part_color_type="assembly_regulator_purple"
                #print("part_color_type",part_color_type)
        elif  part_color is None:
            part_color_type="assembly_empty_empty"
        return part_color_type
# ---------------determine part type and part color 0411 end--------------------------- 
    # --------------------------------------find parts start（0403）----------------------------------------------------
    def find_parts(self, img):
        '''
        image processing
        ''' 
        # hsv masking
        # self.get_logger().info(f"Waiting for camera images ...")
        imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        for color in self.part_poses.keys():
            for type in self.part_poses[color].keys():
                # colour filtering
                imgMask = cv2.inRange(imgHSV, 
                                    self.colorBound(color, "lower"), 
                                    self.colorBound(color, "upper"))
                # template matching
                self.matchTemplate(imgMask, color, type)
                # display bounding boxes
                if self.display_bounding_boxes:
                    if len(self.part_poses[color][type]):
                        # sx, sy -> top left corner
                        # ex, ey -> bottom right corner
                        for (sx, sy, ex, ey) in self.part_poses[color][type]:
                            cv2.rectangle(img, (sx, sy), (ex, ey), self.colors[color], 3)

    def matchTemplate(self, imgMask, color, type):
        # template matching
        if type == "pump":
            tH, tW = self.pump_template.shape#[:2]
            matchField = cv2.matchTemplate(imgMask, self.pump_template, cv2.TM_CCOEFF_NORMED)
        elif type == "battery":
            tH, tW = self.battery_template.shape#[:2]
            matchField = cv2.matchTemplate(imgMask, self.battery_template, cv2.TM_CCOEFF_NORMED)
        elif type == "sensor":
            tH, tW = self.sensor_template.shape#[:2]
            matchField = cv2.matchTemplate(imgMask, self.sensor_template, cv2.TM_CCOEFF_NORMED)
        elif type == "regulator":
            tH, tW = self.regulator_template.shape#[:2]
            matchField = cv2.matchTemplate(imgMask, self.regulator_template, cv2.TM_CCOEFF_NORMED)

        # match many
        (yy, xx) = np.where(matchField >= 0.80)
        raw_matches = []
        for (x, y) in zip(xx, yy):
            raw_matches.append((x, y, x+tW, y+tH))

        # non-max suppression
        refined_matches = []
        refined_matches = non_max_suppression(np.array(raw_matches))

        # do this once to save divisions
        htH, htW = tH//2, tW//2
        centered_refined_matches = []
        for sx, sy, _, _ in refined_matches:
            centered_refined_matches.append((sx + htW, sy + htH))

        # store results
        self.part_poses[color][type] = refined_matches
        self.centered_part_poses[color][type] = centered_refined_matches

    def output_by_slot(self):
        bin = dict([(i, PartMsg(color=None, type=None)) for i in range(1, 10)])

        for color in self.centered_part_poses.keys():
            for type in self.centered_part_poses[color].keys():
               for (csx, csy) in self.centered_part_poses[color][type]:
                    row = 0
                    # slot 1, 2, 3
                    if csy <= 88:
                        row = 1                        
                    # slot 7, 8, 9
                    elif csy >= 151: 
                        row = 3
                    # slot 4, 5, 6
                    else: #csy > 88 and csy < 151:
                        row = 2
                    col = 0
                    if csx <= 68:
                        col = 1
                    elif csx >= 131:
                        col = 3
                    else: # csx > 68 and csx < 131:
                        col = 2
                    
                    bin[self.slot_mapping[(row, col)]] = PartMsg(color=color, type=type)
        return bin

    # Helper functions for part detection
    def colorBound(self, color, bound):
        if bound == "lower":
            return np.array([self.HSVcolors[color]["hmin"],
                             self.HSVcolors[color]["smin"],
                             self.HSVcolors[color]["vmin"]])
        # elif bound == "upper":
        return     np.array([self.HSVcolors[color]["hmax"],
                             self.HSVcolors[color]["smax"],
                             self.HSVcolors[color]["vmax"]])
    def load_part_templates(self):
        self.sensor_template = cv2.imread(
            path.join(get_package_share_directory("competition_tutorials"), "resources", "sensor.png"), cv2.IMREAD_GRAYSCALE)
        self.regulator_template = cv2.imread(
            path.join(get_package_share_directory("competition_tutorials"), "resources", "regulator.png"), cv2.IMREAD_GRAYSCALE)
        self.battery_template = cv2.imread(
            path.join(get_package_share_directory("competition_tutorials"), "resources", "battery.png"), cv2.IMREAD_GRAYSCALE)
        self.pump_template = cv2.imread(
            path.join(get_package_share_directory("competition_tutorials"), "resources", "pump.png"), cv2.IMREAD_GRAYSCALE)
        
        if (not self.sensor_template.shape[0] > 0) or \
           (not self.regulator_template.shape[0] > 0) or \
           (not self.battery_template.shape[0] > 0) or \
           (not self.pump_template.shape[0] > 0):
            return False
        return True
# --------------------------------------------find parts end (0403)---------------------------------------------------------
#endregion 

    def competition_state_cb(self, msg: CompetitionStateMsg):
        # Log if competition state has changed
        if self.competition_state != msg.competition_state:
            self.get_logger().debug(
                f'Competition state is: {self.states_[msg.competition_state]}',
                throttle_duration_sec=1.0)
        self.competition_state = msg.competition_state

    def start_competition(self):
        self.get_logger().debug('Waiting for competition to be ready')

        # Wait for competition to be ready
        while (self.competition_state != CompetitionStateMsg.READY):
            try:
                with self.spin_lock:
                    rclpy.spin_once(self)
            except KeyboardInterrupt:
                return
        
        # noisy terminal output; keep silent

        # Call ROS service to start competition
        while not self.starter.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /ariac/start_competition to be available...')

        # Create trigger request and call starter service
        request = Trigger.Request()
        future = self.starter.call_async(request)

        # Wait until the service call is completed
        with self.spin_lock:
            rclpy.spin_until_future_complete(self, future)

        if future.result().success:
            self.get_logger().info('Started competition.')
        else:
            self.get_logger().info('Unable to start competition')

            
    def end_competition(self):
        self.get_logger().info('Ending competition...')
        # Create client for /ariac/end_competition service
        end_competition_client = self.create_client(Trigger, '/ariac/end_competition')

        # Wait for /ariac/end_competition service to be available
        while not end_competition_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /ariac/end_competition service to be available...')

        # Create trigger request
        request = Trigger.Request()

        # Send request to end_competition service
        future = end_competition_client.call_async(request)

        # Wait until the service call is completed
        with self.spin_lock:
            rclpy.spin_until_future_complete(self, future)

        if future.result().success:
            self.get_logger().info('Competition ended successfully.')
            return True
        else:
            self.get_logger().info('Unable to end competition.')
            return False


    def use_spin(self):
        while True:
            with self.spin_lock:
                rclpy.spin_once(self)
                time.sleep(0.0001)
            

    def new_thread(self):
        new_thread1=Thread(target=self.use_spin)
        new_thread1.start()
        
    def wait(self, duration):
        sleep(duration)


    def sensor_health_state_callback(self,msg):                
        self.break_beam_health=msg.break_beam
        self.proximity_health=msg.proximity
        self.laser_profiler_health=msg.laser_profiler
        self.lidar_health=msg.lidar
        self.camera_health=msg.camera
        self.logical_camera_health=msg.logical_camera

    def agv1_status_callback(self,msg):                
        self.agv1_position=msg.location

    def agv2_status_callback(self,msg):
        self.agv2_position=msg.location

    def agv3_status_callback(self,msg):
        self.agv3_position=msg.location

    def agv4_status_callback(self,msg):
        self.agv4_position=msg.location
  
## ###################################### Order  ###############################################################################    
    def submit_order(self, order_id):
        

        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

        request = SubmitOrder.Request()
        request.order_id = order_id

        future = self.client.call_async(request)
        try:
            with self.spin_lock:
                rclpy.spin_until_future_complete(self, future)
        except KeyboardInterrupt:
            raise KeyboardInterrupt

        if future.result().success:
            self.get_logger().info(f'Submit order {order_id}')
        else:
            self.get_logger().warn('Unable Submit order')
        
        for order in self.orders_list:
            if order.id == order_id:
                self.orders_list.remove(order)
        
        

    def process_order_submission(self, order_id):
        
        for order in self.orders_list:
            if order.id == order_id:
                priority=order.priority
        
        if priority:
            self.submit_order(order_id)
            
            if self.orders_wait_list:
                for order_id in self.orders_wait_list:
                    self.submit_order(order_id)
        
        else:
            high_priority_order_exists = any(order.priority for order in self.orders_list)
            if high_priority_order_exists:
                
                self.orders_wait_list.append(order_id)
            
            else:
                self.submit_order(order_id)
        
        # print("当前状态：",self.competition_state,"订单数量:",len(self.orders_list))
        if self.competition_state == CompetitionState.ORDER_ANNOUNCEMENTS_DONE and len(self.orders_list)==0:

            self.end_competition()

    def timer_submit_callback(self):
        for order in self.orders_list:
            self.submit_order(order.id)
            sleep(1)
        self.end_competition()
        
            
                 

    def perform_quality_check(self, order_id: str) -> bool:
        while not self.quality_checker .wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service quality_check not available, waiting again...')
        # Send quality check request
        request = PerformQualityCheck.Request()
        request.order_id = order_id
        future = self.quality_checker.call_async(request)
        try:
            with self.spin_lock:
                rclpy.spin_until_future_complete(self, future)
        except KeyboardInterrupt:
            raise KeyboardInterrupt  
        
        if future.result().all_passed:
            self.get_logger().info(f'Check order {order_id} Well')
            return True,future.result()
        else:
            self.get_logger().warn('Check order False')   
            # print("future.result()------------",future.result()) 
            return False, future.result()

    def check_faluty_part(self,msg,quadrant_num):
        # 处理未通过质量检查的零件
        quadrants=[msg.quadrant1, msg.quadrant2, msg.quadrant3, msg.quadrant4]
        quadrant=quadrants[quadrant_num-1]
        if quadrant.faulty_part:
            # print(f"Faulty part in quadrant {quadrant_num}")
            return True
        else:
            return False


        
    def process_orders(self,order_msg):
        # print("order_msg",order_msg)
        
        # 解析订单信息
        order_id = order_msg.id
        order_type = order_msg.type
        order_priority = order_msg.priority
        order_kitting_task = order_msg.kitting_task
        order_assembly_task = order_msg.assembly_task
        order_combined_task = order_msg.combined_task
        
        self.order_recored_list.append(order_msg)
        self.orders_list.append(order_msg)
        # 将订单拆分成子任务
        if order_type == 0:
            
            self.ceiling_kitting_once=True
            self.oredr_length[order_id]=len(order_kitting_task.parts)
            self.co_tray_flag[order_id]=False
            quadrant_order = {4: 0, 2: 1, 3: 2, 1: 3}
            order_kitting_task.parts = sorted(order_kitting_task.parts, key=lambda x: quadrant_order.get(x.quadrant, float('inf')))

            if order_priority:
                for idx,part in enumerate(order_kitting_task.parts) :
                    product_type=determine_part_name(part.part.type,part.part.color)
                    product_quadrant=part.quadrant
                    subtask = KittingSubtask(order_id, order_kitting_task.agv_number, order_kitting_task.tray_id, order_kitting_task.destination, product_type,product_quadrant, idx==0) 
                    self.kitting_deque.appendleft(subtask)    
                   
            else:
                for idx,part in enumerate(order_kitting_task.parts) :
                    product_type=determine_part_name(part.part.type,part.part.color)
                    product_quadrant=part.quadrant
                    subtask = KittingSubtask(order_id, order_kitting_task.agv_number, order_kitting_task.tray_id, order_kitting_task.destination, product_type,product_quadrant, idx==len(order_kitting_task.parts)-1) 
                    self.kitting_deque.append(subtask)    


        elif order_type == 1:
            # 2.移动指定的agv到指定的station

            self.assembly_agvs=order_assembly_task.agv_numbers
            self.assembly_order=order_id
            self.assembly_destion=order_assembly_task.station
            self.order_pose[order_id]=None
            self.oredr_length[order_id]=len(order_assembly_task.parts)
            if order_priority:
                for idx,part in enumerate(order_assembly_task.parts) :
                    product_type=determine_part_name(part.part.type,part.part.color)
                    subtask = AssemblySubtask(order_id, order_assembly_task.agv_numbers, order_assembly_task.station,product_type, idx==0,part.assembled_pose,part.install_direction,grap_pose=None) 
                    self.assembly_deque.appendleft(subtask)

            
            else:
                for idx,part in enumerate(order_assembly_task.parts) :
                    product_type=determine_part_name(part.part.type,part.part.color)
                    subtask = AssemblySubtask(order_id, order_assembly_task.agv_numbers, order_assembly_task.station,product_type, idx==len(order_assembly_task.parts)-1,part.assembled_pose,part.install_direction,grap_pose=None) 
                    self.assembly_deque.append(subtask)  
                      
        elif order_type == 2:
            self.order_pose[order_id]=None
            self.combined_spilt.update({ order_id: { "main": False,"ci": False }})
            self.oredr_length[order_id]=len(order_combined_task.parts)

            if order_priority:
                for idx,part in enumerate(order_combined_task.parts) :
                    product_type=determine_part_name(part.part.type,part.part.color)  
                    subtask = CombinedSubtask(order_id, order_combined_task.station,product_type, idx==0,part.assembled_pose,part.install_direction,grap_pose=None) 
                    self.combined_products.appendleft(product_type)
                    self.combined_deque.appendleft(subtask)
            
            else:
                for idx,part in enumerate(order_combined_task.parts) :
                    product_type=determine_part_name(part.part.type,part.part.color)  
                    subtask = CombinedSubtask(order_id, order_combined_task.station,product_type, idx==len(order_combined_task.parts)-1,part.assembled_pose,part.install_direction,grap_pose=None) 
                    self.combined_products.append(product_type)  
                    self.combined_deque.append(subtask)

                
    def cmd_generate(self):
        floor_cmd = None
        ceiling_cmd = None
        floor_cmd_list=[]
        ceiling_cmd_list=[]

        if self.kitting_deque:
            
            # print("第一个订单是:",self.kitting_deque[0].product_type)
            if self.check_tray_exist(self.kitting_deque[0]) :
                self.co_tray_flag=False
                agv,tray=self.check_tray_exist(self.kitting_deque[0])
                floor_cmd = self.create_tray_command(agv,tray)
                
                kitting_subtask = self.kitting_deque.popleft() 
                kitting_part=self.floor_find_shortest_part(kitting_subtask.product_type, kitting_subtask.agv_number)
                
                if kitting_part== None or (kitting_part.location=="conveyor" or kitting_part.need_flip)  or  self.assembly_deque or  self.combined_deque :   # ceiling不做传送带上和翻转的任务
                    self.kitting_deque.appendleft(kitting_subtask)
                else:
                    ceiling_cmd = self.create_ceiling_command(kitting_part, "kitting",'agv'+str(kitting_subtask.agv_number)+'_ks'+str(kitting_subtask.agv_number)+'_tray',kitting_subtask)
  
            else:
                
                kitting_subtask = self.kitting_deque.popleft()
                kitting_part=self.floor_find_shortest_part(kitting_subtask.product_type, kitting_subtask.agv_number)
                if kitting_part==None : 
                    if kitting_subtask.is_last_subtask:                                                                                   # 零件不足
                        self.move_agv(kitting_subtask.agv_number, kitting_subtask.destination)
                        self.process_order_submission(kitting_subtask.order_id)    
                elif kitting_part.location in floor_bins : 
                    floor_cmd = self.create_floor_command(kitting_subtask,"kitting",kitting_part)
                
                elif kitting_part.location in ceiling_bins: 
                    ceiling_cmd = self.create_ceiling_command(kitting_part, "kitting",'agv'+str(kitting_subtask.agv_number)+'_ks'+str(kitting_subtask.agv_number)+'_tray',kitting_subtask)

                elif kitting_part.location in "conveyor": 
                    floor_cmd = self.create_floor_command(kitting_subtask,"kitting",kitting_part)

        if self.assembly_deque:
            
            if self.ceiling_robot_info.is_enabled and self.ceiling_robot_info.is_idle:
                assembly_subtask = self.assembly_deque.popleft()
                self.agv_to_as(assembly_subtask)
                part=None
                # print("应该是三个:",self.order_pose,self.order_pose[assembly_subtask.order_id],assembly_subtask.order_id)
                # print("得到的pose是:",self.order_pose,self.order_pose[assembly_subtask.order_id],assembly_subtask.order_id)
                if self.order_pose[assembly_subtask.order_id]:
                    for agv_part_pose in self.order_pose[assembly_subtask.order_id]:        
                        # print("subtask.product_type-----445",assembly_subtask.product_type,"   !!!  ",determine_part_name(agv_part_pose.part.type,agv_part_pose.part.color))
                        if determine_part_name(agv_part_pose.part.type,agv_part_pose.part.color)==assembly_subtask.product_type:
                            assembly_subtask.grap_pose=agv_part_pose.pose
                            agv=find_closest_agv(agv_part_pose.pose.position.y)
                            part = sPart(assembly_subtask.product_type,'agv'+str(agv)+'_as'+str(assembly_subtask.station)+'_tray',agv_part_pose.pose)  
                            break      
                if part:
                    ceiling_cmd = self.create_ceiling_command(part, "assembly","as"+str(assembly_subtask.station),assembly_subtask)
                else:                                                                                                                       # assembly不够,从bins上拿
                    target_part_type = assembly_subtask.product_type
                    target_part_list = self.search_part_on_bins(target_part_type)
                    target_part_list=[part for part in target_part_list if not part.need_flip]
                    if target_part_list:
                        min_part = min(target_part_list, key=lambda part: part.pose.position.x)
                        self.del_part_from_parts_list(min_part)
                        ceiling_cmd = self.create_ceiling_command(min_part, "combined_kitting","as"+str(assembly_subtask.station),assembly_subtask)
                    else:
                        if (not self.assembly_deque) and (not self.combined_deque):
                            self.process_order_submission(assembly_subtask.order_id)   
                        

        if self.combined_deque:
            
            combined_subtasks = list(self.combined_deque)                        # 先筛除所有需要翻转的任务
            flip_tasks = [task for task in combined_subtasks if all(part.need_flip for part in self.search_part_on_bins(task.product_type)) and self.search_part_on_bins(task.product_type)]

            print("长度分别是:",combined_subtasks,flip_tasks)
            if flip_tasks:
                for task in flip_tasks:
                    target_part_list = self.search_part_on_bins(task.product_type)
                    min_part = min(target_part_list, key=lambda part: part.pose.position.x)
                    self.del_part_from_parts_list(min_part)
                    agv_number = self.which_agv_to_as(task.station)
                    convert_task = self.combined_to_kitting(task, agv_number, len(self.combined_deque) )
                    floor_cmd = self.create_floor_command(convert_task, "combined_kitting_flip", min_part)
                    floor_cmd_list.append(floor_cmd)
                    self.combined_deque.remove(task) 
                    self.combine_ass_task.append(self.combined_to_assembly(task,agv_number))
                self.last_ass_kitting_cmd=floor_cmd_list[-1]


            if self.combined_deque:
                combined_subtask = self.combined_deque.popleft()
                target_part_list = self.search_part_on_bins(combined_subtask.product_type)
                target_part_list=[part for part in target_part_list if not part.need_flip]
                target_part_list_conveyor = self.search_part_on_conveyor(combined_subtask.product_type)
                target_part_list_conveyor = [part for part in target_part_list_conveyor if not part.need_flip]
                
                if target_part_list_conveyor:                        # 如果传送带上有满足的
                    part=target_part_list_conveyor[-1]
                    self.del_part_from_parts_list(part)
                    agv_number = self.which_agv_to_as(combined_subtask.station)
                    convert_task = self.combined_to_kitting(combined_subtask, agv_number, len(self.combined_deque) )
                    floor_cmd = self.create_floor_command(convert_task, "combined_kitting_conveyor", part)
                    self.combine_ass_task_con.append(self.combined_to_assembly(combined_subtask,agv_number))                 
                    self.last_ass_kitting_cmd_conveyor=floor_cmd

                if target_part_list:
                    min_part = min(target_part_list, key=lambda part: part.pose.position.x)
                    self.del_part_from_parts_list(min_part)
                
                    if ceiling_cmd:             # assembly已经把ceil给用了
                        ceiling_cmd_list.append(ceiling_cmd)
                        ceiling_cmd = self.create_ceiling_command(min_part, "combined_kitting", "as" + str(combined_subtask.station), combined_subtask)
                        ceiling_cmd_list.append(ceiling_cmd)
                    else:
                        ceiling_cmd = self.create_ceiling_command(min_part, "combined_kitting", "as" + str(combined_subtask.station), combined_subtask)
                else:
                    if not self.combined_deque:
                        self.process_order_submission(combined_subtask.order_id)   

        if floor_cmd:
            floor_cmd=[floor_cmd]
        if ceiling_cmd:
            ceiling_cmd=[ceiling_cmd]       
        
        if floor_cmd_list: 
            floor_cmd=floor_cmd_list
        if ceiling_cmd_list:
            ceiling_cmd=ceiling_cmd_list

        return floor_cmd, ceiling_cmd
    
    
            
    def ceiling_query(self):
        ceiling_cmd=None
        ceiling_cmd_list=[]

        if self.ceiling_robot_info.is_enabled and self.ceiling_robot_info.is_idle:
        
            if self.combined_deque:
                
                combined_subtasks = list(self.combined_deque)                        # 1、先筛除所有需要翻转的任务,需要翻转的任务要配合floor做
                flip_tasks = [task for task in combined_subtasks if all(part.need_flip for part in self.search_part_on_bins(task.product_type)) and self.search_part_on_bins(task.product_type)]

                print("长度分别是:",combined_subtasks,flip_tasks)
                
                if flip_tasks:
                    for task in flip_tasks:
                        target_part_list = self.search_part_on_bins(task.product_type)
                        min_part = min(target_part_list, key=lambda part: part.pose.position.x)
                        self.del_part_from_parts_list(min_part)
                        agv_number = self.which_agv_to_as(task.station)
                        convert_task = self.combined_to_kitting(task, agv_number, len(self.combined_deque) )
                        floor_cmd = self.create_floor_command(convert_task, "combined_kitting_flip", min_part)
                        self.combined_to_floor_cmd.append(floor_cmd)
                        self.combined_deque.remove(task) 
                        
                        self.combine_ass_task.append(self.combined_to_assembly(task,agv_number))

                    self.last_ass_kitting_cmd=self.combined_to_floor_cmd[-1]

                    print("看看",self.combined_to_floor_cmd)
            
                if self.combined_deque:                                           # 2、从传送带或者箱子上抓取物体
                    combined_subtask = self.combined_deque.popleft()
                    target_part_list = self.search_part_on_bins(combined_subtask.product_type)
                    target_part_list=[part for part in target_part_list if not part.need_flip]
                    target_part_list_conveyor = self.search_part_on_conveyor(combined_subtask.product_type)
                    target_part_list_conveyor = [part for part in target_part_list_conveyor if not part.need_flip]
                    
                    if target_part_list_conveyor:                        # 如果传送带上有满足的
                        part=target_part_list_conveyor[-1]
                        with self.convey_parts_lock :
                            self.convey_parts.remove(part)
                        agv_number = self.which_agv_to_as(combined_subtask.station)
                        convert_task = self.combined_to_kitting(combined_subtask, agv_number, len(self.combined_deque) )
                        floor_cmd = self.create_floor_command(convert_task, "combined_kitting_conveyor", part)
                        self.combine_ass_task_con.append(self.combined_to_assembly(combined_subtask,agv_number))                 
                        self.last_ass_kitting_cmd_conveyor=floor_cmd

                    if target_part_list:
                        min_part = min(target_part_list, key=lambda part: part.pose.position.x)
                        self.del_part_from_parts_list(min_part)
                    
                        if ceiling_cmd:             # assembly已经把ceil给用了
                            ceiling_cmd_list.append(ceiling_cmd)
                            ceiling_cmd = self.create_ceiling_command(min_part, "combined_kitting", "as" + str(combined_subtask.station), combined_subtask)
                            ceiling_cmd_list.append(ceiling_cmd)
                        else:
                            ceiling_cmd = self.create_ceiling_command(min_part, "combined_kitting", "as" + str(combined_subtask.station), combined_subtask)
                            ceiling_cmd_list.append(ceiling_cmd)
                    else:
                        self.oredr_length[combined_subtask.order_id]=self.oredr_length[combined_subtask.order_id]-1                # 这个命令执行不了
                        if self.oredr_length[combined_subtask.order_id]==0:   
                            self.process_order_submission(combined_subtask.order_id)   
        
            
            if self.assembly_deque:  #
               
                assembly_subtask = self.assembly_deque.popleft()
                self.agv_to_as(assembly_subtask)
                part=None
                if self.order_pose[assembly_subtask.order_id]:
                    print('1'*100)
                    print('self.order_pose',self.order_pose)
                    for agv_part_pose in self.order_pose[assembly_subtask.order_id]:      
                        # print("subtask.product_type-----445",assembly_subtask.product_type,"   !!!  ",determine_part_name(agv_part_pose.part.type,agv_part_pose.part.color))
                        if determine_part_name(agv_part_pose.part.type,agv_part_pose.part.color)==assembly_subtask.product_type:
                            assembly_subtask.grap_pose=agv_part_pose.pose
                            agv=find_closest_agv(agv_part_pose.pose.position.y)
                            part = sPart(assembly_subtask.product_type,'agv'+str(agv)+'_as'+str(assembly_subtask.station)+'_tray',agv_part_pose.pose)  
                            break      
                    if part:
                        self.order_pose[assembly_subtask.order_id].remove(agv_part_pose)  

                if part:
                    ceiling_cmd = self.create_ceiling_command(part, "assembly","as"+str(assembly_subtask.station),assembly_subtask)
                    ceiling_cmd_list.append(ceiling_cmd) 
                else:                                                                                                                       # assembly不够,从bins上拿
                    target_part_type = assembly_subtask.product_type
                    target_part_list = self.search_part_on_bins(target_part_type)
                    target_part_list=[part for part in target_part_list if not part.need_flip]
                    if target_part_list:
                        min_part = min(target_part_list, key=lambda part: part.pose.position.x)
                        self.del_part_from_parts_list(min_part)
                        ceiling_cmd = self.create_ceiling_command(min_part, "combined_kitting","as"+str(assembly_subtask.station),assembly_subtask)
                        ceiling_cmd_list.append(ceiling_cmd) 
                    else:
                        self.oredr_length[assembly_subtask.order_id]=self.oredr_length[assembly_subtask.order_id]-1                # 这个命令执行不了
                        if self.oredr_length[assembly_subtask.order_id]==0:   
                            self.process_order_submission(assembly_subtask.order_id)       
                
                    
            if self.kitting_deque:
                
                if self.check_tray_exist(self.kitting_deque[0])  and len(self.kitting_deque)>1:           # 只抓一次
                    
                    kitting_subtask = self.kitting_deque.popleft()
                    kitting_part=self.floor_find_shortest_part(kitting_subtask.product_type, kitting_subtask.agv_number)
                    print("走的是这一步 1",kitting_part) 
                    if not kitting_part : 
                        self.kitting_deque.appendleft(kitting_subtask)                          # 检测到没有给floor，防止convey

                    else:
                        if kitting_part.need_flip or kitting_part.location == "conveyor":     
                            self.kitting_deque.appendleft(kitting_subtask)  

                        elif kitting_part.location in (floor_bins + ceiling_bins): 
                            ceil_do_kit_cmd = self.create_ceiling_command(kitting_part, "kitting",'agv'+str(kitting_subtask.agv_number)+'_ks'+str(kitting_subtask.agv_number)+'_tray',kitting_subtask)
                            if ceiling_cmd :
                                self.kitting_deque.appendleft(kitting_subtask)                                                      # ceiling有命令了,归还任务
                            else:
                                ceiling_cmd_list.append(ceil_do_kit_cmd)

                # if not self.floor_robot_info.is_enabled :              # floor故障，kitting的命令给ceiling
                #     print("故障了，你不选吗 1")
                #     if self.kitting_deque:
                #         kitting_subtask = self.kitting_deque.popleft()
                #         kitting_part=self.floor_find_shortest_part(kitting_subtask.product_type, kitting_subtask.agv_number)
                        
                #         if kitting_part==None :                                                                 # 所有地方都没有需要的零件
                #             self.oredr_length[kitting_subtask.order_id]=self.oredr_length[kitting_subtask.order_id]-1
                #             print("floor现在的长度是:",self.oredr_length[kitting_subtask.order_id])  
                #             if self.oredr_length[kitting_subtask.order_id]==0:                                                                                 # 零件不足并且是最后一个
                #                 self.move_agv(kitting_subtask.agv_number, kitting_subtask.destination)
                #                 self.process_order_submission(kitting_subtask.order_id)   
                                
                #         elif kitting_part.need_flip or kitting_part.location == "conveyor":     
                #             self.kitting_deque.appendleft(kitting_subtask)  

                #         elif kitting_part.location in (floor_bins + ceiling_bins): 
                #             ceil_do_kit_cmd = self.create_ceiling_command(kitting_part, "kitting",'agv'+str(kitting_subtask.agv_number)+'_ks'+str(kitting_subtask.agv_number)+'_tray',kitting_subtask)
                #             if ceiling_cmd :
                #                 self.kitting_deque.appendleft(kitting_subtask)                                                      # ceiling有命令了,归还任务
                #             else:
                #                 ceiling_cmd_list.append(ceil_do_kit_cmd)
  

        if ceiling_cmd:
            ceiling_cmd=[ceiling_cmd]  
        else:
            ceiling_cmd=[]     
        if ceiling_cmd_list:
            ceiling_cmd=ceiling_cmd_list

        if self.floor_to_ceiling_cmd:
            ceiling_cmd.extend(self.floor_to_ceiling_cmd)
            self.floor_to_ceiling_cmd.clear()
        
        # print("目前ceiling命令的长度是:",len(ceiling_cmd))  
        return ceiling_cmd

    def floor_query(self):
        floor_cmd=None
        floor_cmd_list=[]
        if self.floor_robot_info.is_enabled and self.floor_robot_info.is_idle:
            if self.kitting_deque:
                if self.check_tray_exist(self.kitting_deque[0]) :                                                   # 1、先执行托盘命令 
                    # print('aaaaa00',self.check_tray_exist(self.kitting_deque[0]))
                    agv,tray=self.check_tray_exist(self.kitting_deque[0])
                    floor_cmd = self.create_tray_command(self.kitting_deque[0],agv,tray)
                
                else:
                    # print( "kitting_deque:",len(self.kitting_deque))
                    floor_subtasks = list(self.kitting_deque)                        # 1、先筛除所有需要翻转的任务,需要翻转的任务要配合floor做
                    conveyor_tasks = [task for task in floor_subtasks if all(part.location=="conveyor" for part in self.search_part_on_conveyor(task.product_type)) and self.search_part_on_conveyor(task.product_type)]
                    if conveyor_tasks:
                        for task in conveyor_tasks:
                            target_part_list = self.search_part_on_conveyor(task.product_type)
                            target_part_list_conveyor = [part for part in target_part_list if part.need_flip == False]  
                            print("不用翻转的有:",len(target_part_list_conveyor)) 
                            if target_part_list_conveyor:
                                min_part = target_part_list_conveyor[-1]
                            else:
                                min_part = target_part_list[-1]
                                
                            with self.convey_parts_lock :
                                self.convey_parts.remove(min_part)
                            floor_cmd = self.create_floor_command(task,"kitting",min_part)   
                            floor_cmd_list.append(floor_cmd)
                            self.kitting_deque.remove(task) 
                    
                    else:
                        kitting_subtask = self.kitting_deque.popleft()
                        print("任务类型是:",kitting_subtask.product_type,kitting_subtask.agv_number)  

                        print('kitting_subtask',kitting_subtask.product_type)
                    
                        kitting_part=self.floor_find_shortest_part(kitting_subtask.product_type, kitting_subtask.agv_number)
                        
                        if kitting_part==None :                                                                 # 所有地方都没有需要的零件
                            self.oredr_length[kitting_subtask.order_id]=self.oredr_length[kitting_subtask.order_id]-1
                            print("floor现在的长度是:",self.oredr_length[kitting_subtask.order_id])  
                            if self.oredr_length[kitting_subtask.order_id]==0:                                                                                 # 零件不足并且是最后一个
                                self.move_agv(kitting_subtask.agv_number, kitting_subtask.destination)
                                self.process_order_submission(kitting_subtask.order_id)   
                                
                        elif kitting_part.location in floor_bins : 
                            floor_cmd = self.create_floor_command(kitting_subtask,"kitting",kitting_part)
                            print('floor_cmd',floor_cmd)

                        elif kitting_part.location in ceiling_bins: 
                            ceiling_cmd = self.create_ceiling_command(kitting_part, "kitting",'agv'+str(kitting_subtask.agv_number)+'_ks'+str(kitting_subtask.agv_number)+'_tray',kitting_subtask)
                            self.floor_to_ceiling_cmd.append(ceiling_cmd)
                        elif kitting_part.location in "conveyor": 
                            floor_cmd = self.create_floor_command(kitting_subtask,"kitting",kitting_part)       
                        
                        if kitting_part==None : 
                            print("零件是:",kitting_part)     
                        else: 
                            print("零件位置是:",kitting_part.location)     
          
        # print("floor_cmd:",floor_cmd)
        if floor_cmd:
            floor_cmd=[floor_cmd]
        else:
            floor_cmd=[]
        if floor_cmd_list: 
            floor_cmd=floor_cmd_list
            
        # if self.combined_to_floor_cmd:
        #     floor_cmd.extend(self.combined_to_floor_cmd)
        #     self.combined_to_floor_cmd.clear()

        return floor_cmd
    

    def ceiling_execute(self):
        with self.task_threadLock:
            cmd= self.ceiling_query()
            # print('ceiling cmd',cmd)
        if cmd:
            self.ceiling_execute_cmd(cmd)
        else:
            sleep(0.1)

    def floor_execute(self):
        with self.task_threadLock:
            cmd= self.floor_query()
            # print('floor cmd',cmd)
        if cmd:
            self.floor_execute_cmd(cmd)
        else:
            sleep(0.1)


    def run(self):
        
        ceiling_init_thread = threading.Thread(target=self.gantry_robot_init)
        ceiling_init_thread.start()
        self.kitting_robot_init("bin_agv_insert_joint")     
        # self.move_to('bin2')
        # sleep(1)
        # self.grasp_part_on_bins('bin2',self.bin2_parts[0])
        # self.move_to('agv1_ks1_tray')
        # self.floor_execute()

        
        # if self.kitting_deque or self.combined_deque:                              # 为翻转留出空地
        #     self.determine_clean_part_bin1()
        #     self.determine_clean_part_bin6()      
            
        while True:
            
            # print("bin1:::",len(self.bin1_parts))
            # print("bin2:::",len(self.bin2_parts))
            # print("bin3:::",len(self.bin3_parts))
            # print("bin4:::",len(self.bin4_parts))
            # print("bin5:::",len(self.bin5_parts))
            # print("bin6:::",len(self.bin6_parts))
            # print("bin7:::",len(self.bin7_parts))
            # print("bin8:::",len(self.bin8_parts))
            # for part in self.bin2_parts:
            #     print(part.type)
            
            if self.ceiling_robot_info.is_idle:
                self.task_executor.submit(self.ceiling_execute)
            if self.floor_robot_info.is_idle:
                self.task_executor.submit(self.floor_execute)
            time.sleep(0.2)  # 可以适当调整线程之间的间隔时间

                             
    # def run(self):
        
    #     ceiling_init_thread = threading.Thread(target=self.gantry_robot_init)
    #     ceiling_init_thread.start()
    #     self.kitting_robot_init("bin_agv_insert_joint")      
        
    #     if self.kitting_deque or self.combined_deque:                              # 为翻转留出空地
    #         self.determine_clean_part_bin1()
    #         self.determine_clean_part_bin6()      

    #     while True:
    #         floor_cmd, ceiling_cmd = self.cmd_generate()
            
    #         if floor_cmd and ceiling_cmd:
    #             ceiling_thread=Thread(target=self.floor_execute_cmd,args=(floor_cmd,))
    #             ceiling_thread.start()
    #             self.ceiling_execute_cmd(ceiling_cmd)
                
    #             while ceiling_thread.is_alive():
    #                 sleep(0.1)
            
    #         elif floor_cmd and (not ceiling_cmd):

    #             self.floor_execute_cmd(floor_cmd) 

    #         elif ceiling_cmd and (not floor_cmd):
    #             self.ceiling_execute_cmd(ceiling_cmd)
        

    def floor_execute_cmd(self,cmds):
        self.threadLock.acquire()
        self.floor_robot_info.is_idle=False
        self.threadLock.release()

        for cmd in cmds:
            print("执行的命令是：",cmd,"类型为",cmd.type)
            if cmd.type=="tray":
                self.FloorRobotPickandPlaceTray(cmd,cmd.task,cmd.target_position)
                # print("打印co_tray_flag的状态",self.co_tray_flag,cmd.command_id.order_id) #tray的id比较特殊
        
                self.lock_agv_tray(cmd.target_position)
                self.agv_has_tray[cmd.target_position-1]=True
                
            if cmd.type=="kitting" :
                
                if cmd.pick_part.location=="conveyor":                                                            # 抓取
                    Flag=self.floor_grasp_on_conveyor(cmd.pick_part)
                    if not Flag:
                        self.oredr_length[cmd.task.order_id]=self.oredr_length[cmd.task.order_id]-1                   # 执行每一个命令都要减1，但是append需要加1
                        if self.oredr_length[cmd.task.order_id]==0:         ## 没有问题，提交订单
                            
                            self.move_agv(cmd.task.agv_number, cmd.task.destination)
                            
                            print("AGV has arrived :",agv_place[cmd.task.destination])   
                            if cmd.type=="kitting":
                                self.process_order_submission(cmd.task.order_id)    
                        return False
                    # if cmd.pick_part.need_flip:
                    #     self.filp_part_on_convey(cmd.pick_part,cmd.task.agv_number)

                else:
                    self.floor_grasp_on_bins(cmd.pick_part,cmd.task)           
                    
                if cmd.type=="kitting":
                    agv=cmd.task.agv_number
                                                
                location='agv'+str(cmd.task.agv_number)+'_ks'+str(cmd.task.agv_number)+'_tray'   
                self.move_to(location) 
                self.floor_robot_info.location=location
                
                if self.floor_robot_gripper_state.attached: 
                    
                    world_target=self.floor_place_part_on_agv(location,cmd.task,cmd.pick_part,cmd.type)                   # 放置

                    check_result,check_info=self.perform_quality_check(cmd.task.order_id)  
                    sleep(0.5)
                    check_result,check_info=self.perform_quality_check(cmd.task.order_id)    
                    
                    if  self.handle_faultu_part(check_info,cmd.task,world_target,cmd.pick_part):                ## 有问题
                        
                        self.kitting_deque.appendleft(cmd.task)
                    
                    else:
                        self.oredr_length[cmd.task.order_id]=self.oredr_length[cmd.task.order_id]-1                   # 执行每一个命令都要减1，但是append需要加1
                        print("做任务floor现在的长度是:",self.oredr_length[cmd.task.order_id])  
                        if self.oredr_length[cmd.task.order_id]==0:         ## 没有问题，提交订单
                            
                            self.move_agv(cmd.task.agv_number, cmd.task.destination)
                            
                            print("AGV has arrived :",agv_place[cmd.task.destination])   
                            if cmd.type=="kitting":
                                self.process_order_submission(cmd.task.order_id)    
                        
                else:                                                                                           ## 掉落

                    self.kitting_deque.appendleft(cmd.task)


            if cmd.type=="combined_kitting_flip":
            
                self.floor_grasp_on_bins(cmd.pick_part,cmd.task)   
                location='agv'+str(cmd.task.agv_number)+'_ks'+str(cmd.task.agv_number)+'_tray'   
                self.move_to(location) 
                self.floor_robot_info.location=location
                self.floor_place_part_on_agv(location,cmd.task,cmd.pick_part,cmd.type)              # 放置
                
                if self.last_ass_kitting_cmd==cmd:
                    for task in self.combine_ass_task:
                        self.assembly_deque.appendleft(task)
                print("给assembly任务的有",len(self.assembly_deque)) 

            if cmd.type=="combined_kitting_conveyor":

                self.floor_grasp_on_conveyor(cmd.pick_part)  
                location='agv'+str(cmd.task.agv_number)+'_ks'+str(cmd.task.agv_number)+'_tray'   
                self.move_to(location) 
                self.floor_robot_info.location=location
                self.floor_place_part_on_agv(location,cmd.task,cmd.pick_part,cmd.type)              # 放置
                if self.last_ass_kitting_cmd_conveyor==cmd:
                    for task in self.combine_ass_task_con:
                        self.assembly_deque.appendleft(task)
                print("给assembly任务的有",len(self.assembly_deque)) 
        
        self.threadLock.acquire()
        self.floor_robot_info.is_idle=True
        self.threadLock.release()
        
        
         
    def ceiling_execute_cmd(self,cmds):  
        
        self.threadLock.acquire()
        self.ceiling_robot_info.is_idle=False
        self.threadLock.release()
        for cmd in cmds:

            if cmd.type=="assembly":
                self.oredr_length[cmd.task.order_id]=self.oredr_length[cmd.task.order_id]-1
                self.move_to_ceiling(cmd.pick_part.location)
                self.ceiling_robot_info.work_state="move_to_agv"
                sleep(5)

                self.ceiling_pick_assembly_part(cmd.pick_part)  

                self.move_to_ceiling(cmd.target_position) 
                
                print("零件类型是:",cmd.pick_part.type)
                if 'battery' in cmd.pick_part.type:
                    self.ceiling_place_assembly_battery(cmd.task,cmd.type,cmd.pick_part)   
                if 'pump' in cmd.pick_part.type:
                    self.ceiling_place_assembly_pump(cmd.task,cmd.type,cmd.pick_part)   
                if 'sensor' in cmd.pick_part.type:
                    self.ceiling_place_assembly_sensor(cmd.task,cmd.type,cmd.pick_part)   
                if 'regulator' in cmd.pick_part.type:
                    self.ceiling_place_assembly_regulator(cmd.task,cmd.type,cmd.pick_part)   
                sleep(0.5)
                self.set_ceiling_robot_gripper_state(False) 
                self.ceiling_robot_info.work_state = "has_assembly"
                
                self.ceiling_arm_init()
                
                print(f"这个零件的状态是:{cmd.task.is_last_subtask}")
                
                if self.oredr_length[cmd.task.order_id]==0:  
                    self.process_order_submission(cmd.task.order_id)   
                    
            
            if cmd.type=="kitting":
                   
                # print("ceiling做kitting后order_length还有:",self.oredr_length[cmd.task.order_id],cmd.task.order_id) 
                self.move_to_ceiling(cmd.pick_part.location)
                while cmd.pick_part.location==self.floor_robot_info.location:       # Kitting在工作，去了不动
                    sleep(0.2)
                
                if cmd.pick_part.need_flip:
                    # self.pick_part_flip__on_bin_ceiling(cmd.pick_part.location,cmd.pick_part) 
                    # self.flip_part_on_ceiling(cmd.task.agv_number,cmd.pick_part)
                    print("Now Ceiling robot don't flip part")
                    
                else:
                    self.pick_part_on_bin_ceiling(cmd.pick_part.location,cmd.pick_part)            # 抓起来
                   

                self.move_to_ceiling(cmd.target_position)
                print("打印co_tray_flag的状态2",self.co_tray_flag[cmd.task.order_id])
                while cmd.pick_part.location==self.floor_robot_info.location or not self.co_tray_flag[cmd.task.order_id]:
                    # print("在等待")
                    sleep(0.2)
                
                if self.ceiling_robot_gripper_state.attached: 
                    self.ceiling_robot_place_part_on_kit_tray(cmd.task,cmd.pick_part,cmd.target_position)                   # 放下去
                else:
                    self.kitting_deque.appendleft(cmd.task)
                
                self.oredr_length[cmd.task.order_id]=self.oredr_length[cmd.task.order_id]-1  
                if self.oredr_length[cmd.task.order_id]==0:                                                         ## 没有问题，提交订单
                    
                    self.move_agv(cmd.task.agv_number, cmd.task.destination)
                    while self.AGV_location['agv'+str(cmd.task.agv_number)] != agv_place[cmd.task.destination]:
                        sleep(0.2)
                    
                    print("AGV has arrived :",agv_place[cmd.task.destination])   
                    self.process_order_submission(cmd.task.order_id)    
            
            if cmd.type=="combined_kitting":
                print('正在进行combined_kitting'*10)  
                self.oredr_length[cmd.task.order_id]=self.oredr_length[cmd.task.order_id]-1
                
                self.move_to_ceiling(cmd.pick_part.location)
                self.ceiling_robot_info.work_state="move_to_agv"
                
                if cmd.pick_part.need_flip:
                    
                    while (not self.floor_robot_info.is_idle) or (not self.floor_robot_info.is_enabled):
                        sleep(0.2)              # 等floor空闲
                    
                    agv_number=find_closest_agv(cmd.pick_part.pose.position.y)
                    self.move_to(cmd.pick_part.location)
                    self.grasp_flip_part_on_bins(cmd.pick_part,agv_number,type='combined_kitting')
                    self.move_to('can')
                    
                    
                    cmd.pick_part=self.pick_part_has_flip_ceiling(cmd.pick_part.location,cmd.pick_part,agv_number) 
                    
                else:
                    self.pick_part_on_bin_ceiling(cmd.pick_part.location,cmd.pick_part)            # 抓起来

                print('cmd.target_position',cmd.target_position)
                self.move_to_ceiling(cmd.target_position)
                

                print("零件类型是:",cmd.pick_part.type)
                if 'battery' in cmd.pick_part.type:
                    sleep(1)
                    self.ceiling_place_assembly_battery(cmd.task,cmd.type,cmd.pick_part)   
                if 'pump' in cmd.pick_part.type:
                    self.ceiling_place_assembly_pump(cmd.task,cmd.type,cmd.pick_part)   
                if 'sensor' in cmd.pick_part.type:
                    self.ceiling_place_assembly_sensor(cmd.task,cmd.type,cmd.pick_part)   
                if 'regulator' in cmd.pick_part.type:
                    self.ceiling_place_assembly_regulator(cmd.task,cmd.type,cmd.pick_part)   
                sleep(0.5)
                self.set_ceiling_robot_gripper_state(False) 
                
                self.ceiling_arm_init()
                
                if self.oredr_length[cmd.task.order_id]==0:  
                    self.process_order_submission(cmd.task.order_id) 


            if cmd.type=="combined_assembly":     
                print('正在进行combined_assembly'*10)  
                self.move_to_ceiling(cmd.pick_part.location)
                self.ceiling_robot_info.work_state="move_to_agv"
                
                self.ceiling_pick_assembly_part(cmd.pick_part)  

                self.move_to_ceiling(cmd.target_position) 
                
                print("零件类型是:",cmd.pick_part.type)
                if 'battery' in cmd.pick_part.type:
                    self.ceiling_place_assembly_battery(cmd.task,cmd.type,cmd.pick_part)   
                if 'pump' in cmd.pick_part.type:
                    self.ceiling_place_assembly_pump(cmd.task,cmd.type,cmd.pick_part)   
                if 'sensor' in cmd.pick_part.type:
                    self.ceiling_place_assembly_sensor(cmd.task,cmd.type,cmd.pick_part)   
                if 'regulator' in cmd.pick_part.type:
                    self.ceiling_place_assembly_regulator(cmd.task,cmd.type,cmd.pick_part)   
                sleep(0.5)
                self.set_ceiling_robot_gripper_state(False) 
                self.ceiling_robot_info.work_state = "has_assembly"
                
                self.ceiling_arm_init()
                
                print(f"这个零件的状态是:{cmd.task.is_last_subtask}")
                if cmd.task.is_last_subtask:                                                          ## 没有问题，提交订单
                    self.process_order_submission(cmd.task.order_id)   

    
        self.threadLock.acquire()
        self.ceiling_robot_info.is_idle=True
        self.threadLock.release()
                
            
    def combined_to_assembly(self, combined_subtask, agv_number):
        assembly_subtask = AssemblySubtask(
            order_id=combined_subtask.order_id,
            agv_numbers=agv_number,
            station=combined_subtask.station,
            product_type=combined_subtask.product_type,
            is_last_subtask=combined_subtask.is_last_subtask,
            assembled_pose=combined_subtask.assembled_pose,
            install_direction=combined_subtask.install_direction,
            grap_pose=combined_subtask.grap_pose
        )
        return assembly_subtask     

    def combined_to_kitting(self, combined_subtask, agv_number, product_quadrant):
        kitting_subtask = KittingSubtask(
            order_id=combined_subtask.order_id,
            agv_number=agv_number,
            tray_id=None,  # 根据需要设置 tray_id
            destination=combined_subtask.station,
            product_type=combined_subtask.product_type,
            product_quadrant=product_quadrant,
            is_last_subtask=combined_subtask.is_last_subtask
        )
        return kitting_subtask
   
        
    def create_floor_command(self,subtask, task_type,part):
        floor_cmd = Command()
        floor_cmd.command_id=self.get_clock().now() 
        floor_cmd.robot_name = "floor"
        floor_cmd.type = task_type
        floor_cmd.pick_part = part
        floor_cmd.target_position = subtask.destination

        floor_cmd.is_done = False
        floor_cmd.task=subtask
        return floor_cmd      

    def create_ceiling_command(self,pick_part, task_type,target_position,subtask):       
        ceiling_cmd = Command()
        ceiling_cmd.command_id=self.get_clock().now() 
        ceiling_cmd.robot_name = "ceiling"
        ceiling_cmd.type = task_type
        ceiling_cmd.pick_part = pick_part
        ceiling_cmd.target_position= target_position
        ceiling_cmd.is_done = False
        ceiling_cmd.task=subtask
        return ceiling_cmd    

    def create_tray_command(self,task_exe,agv,tray):
        floor_cmd = Command()
        floor_cmd.command_id=task_exe         # 借用一下这个变量，执行任务
        floor_cmd.robot_name = "floor"
        floor_cmd.type = "tray"
        floor_cmd.target_position = agv
        floor_cmd.task=tray
        
        floor_cmd.is_done = False
        return floor_cmd      

    def which_agv_to_as(self,station):
        if station == CombinedTask.AS1 or station == CombinedTask.AS2:    # 选择第一个或第二个空位
            if not self.agv_has_tray[0]:
                agv_number = 1
            elif not self.agv_has_tray[1]:
                agv_number = 2
            elif not self.agv_has_tray[2]:
                agv_number = 3
            elif not self.agv_has_tray[3]:
                agv_number = 4
        else:
            if not self.agv_has_tray[2]:
                agv_number = 3
            elif not self.agv_has_tray[3]:
                agv_number = 4
            elif not self.agv_has_tray[0]:
                agv_number = 1
            elif not self.agv_has_tray[1]:
                agv_number = 2  
        return agv_number     

        
##################  Floor   Robot   Function ###################################################################################

    def floor_robot_gripper_state_cb(self, msg: VacuumGripperState):
        self.floor_robot_gripper_state = msg
        

    def set_floor_robot_gripper_state(self, state) -> bool: 
        if self.floor_robot_gripper_state.enabled == state:
            return
        
        request = VacuumGripperControl.Request()
        request.enable = state
        
        future = self.floor_gripper_enable.call_async(request)
        
        try:
            with self.spin_lock:
                rclpy.spin_until_future_complete(self, future)
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        
        if future.result().success:
            # self.get_logger().info(f'Changed gripper state to {self.gripper_states_[state]}')
            pass
        else:
            self.get_logger().warn('Unable to change gripper state')


    def human_state_callback(self,msg):  
        self.human_distance=self.calculate_plane_distance(msg.human_position,msg.robot_position)
        k_H = math.sqrt(msg.human_velocity.x**2 + msg.human_velocity.y**2)
        k_R = math.sqrt(msg.robot_velocity.x**2 + msg.robot_velocity.y**2)
        
        t1 = 1
        t2 = 1.5
        delta = 2

        # 计算d_min
        d_min = k_H * (t1 + t2) + k_R * t1 + delta
        # self.min_distance=
        # print("机器人和人之间的距离是",self.human_distance,"d_min=",d_min)
               

    def kitting_arm_joint_state_callback(self, msg):
        # print("msg.position",msg)
        self.kitting_arm_joint_states =msg.reference.positions

        
    def linear_joint_state_callback(self, msg):
        self.kitting_base_y = msg.reference.positions[0]
        # print(self.kitting_base_y,"-----------------")
        
    def robot_health_state_callback(self, msg):
        self.floor_robot_info.is_enabled=msg.floor_robot
        self.ceiling_robot_info.is_enabled=msg.ceiling_robot


    def kitting_robot_init(self, state,Wait_flag = True): 
        while not self.kitting_arm_joint_states and rclpy.ok():
            sleep(0.1)
            # noisy terminal output; keep silent while waiting
        # 复位
        init_position = copy.deepcopy(self.kitting_typical_joints[state])
        self.MOV_A(init_position, eps =0.02,sleep_flag = Wait_flag)
        # print ('initialization success !')
        return True

    def STOP(self):
        q_begin = copy.deepcopy(self.kitting_arm_joint_states)
        q_end = self.kitting_arm_joint_states
        delta = [abs(q_begin[i] - q_end[i]) for i in range(0,len(q_begin))]
        time_from_start = max(delta)/kitting_angle_velocity
        #做运动插补
        traj = traj_generate(self.kitting_arm_joint_names,q_begin,q_end,time_from_start)
        self.move(self.floor_action_client,traj)

    # @floor_fault_detector
    def MOV_A(self, target_joint_angles,time_from_start = 0.0,eps = 0.01, sleep_flag=True):         #输入的是关节角度向量Angle，直接控制运动
        '''
        time_from_start,默认按照最快时间执行
        eps = 0.0025
        '''
        q_begin = copy.deepcopy(self.kitting_arm_joint_states)
        q_end = target_joint_angles
        delta = [abs(q_begin[i] - q_end[i]) for i in range(0,len(q_begin))]
        if time_from_start == 0.0:
            time_from_start = max(delta)/kitting_angle_velocity
        #做运动插补
        traj = traj_generate(self.kitting_arm_joint_names,q_begin,q_end,time_from_start)
        ##print traj

        self.move(self.floor_action_client,traj)
        sleep(time_from_start)

    @floor_fault_detector
    def move_to(self, location,eps = 0.02, flip = False,right=0.0,left=0.0):
        '''
        到达指定位置点 e.g., bin1
        '''
        # self.robot_info.next_park_location = location
        
        # self.robot_info.work_state= "moving"
        # if flip:
        #     self.robot_info.work_state= "flipping"
        #获取目标点的位置，主要用Y轴做移动
        # print(kitting_robot_park_location.keys())
        self.floor_robot_info.location = location
        end_point =-kitting_robot_park_location[location][1]-right+left
        q_begin = self.kitting_base_y 
        distance = abs(q_begin - end_point)
        move_time = distance/kitting_velocity
        q_begin = [q_begin]
        q_end= [end_point]   
        traj = traj_generate(self.linear_joint_names,q_begin,q_end,move_time)
        self.move(self.linear_action_client,traj)
        sleep(move_time)

        
        
    @floor_fault_detector
    def MOV_M(self, target_matrix,eps = 0.005,flip_flag=False,times=1.5,time_set=0.0):  #输入的是目标变换矩阵Mat,比MOV_A多求逆解，目标抓取
        q_begin = copy.deepcopy(self.kitting_arm_joint_states)

        #求逆解
        # print("MOV_M--target_matrix",target_matrix)
        target = IKinematic(target_matrix,q_begin,A_k)
        if target==None:
            print("解不出来,你看看target_matrix",target_matrix)
            return False
        # #print "IKinematic_result:",target
        if flip_flag:
            target[-1] = q_begin[-1]
        #求时间
        delta = [abs(q_begin[i] - target[i]) for i in range(0,len(q_begin))]
        ##print delta
        distance = max(delta)
        if time_set==0.0:
            time_from_start = distance/kitting_angle_velocity*times
        else:
            time_from_start = time_set
        traj = traj_generate(self.kitting_arm_joint_names,q_begin,target,time_from_start)
        self.move(self.floor_action_client,traj)
        sleep(time_from_start)
        # self.send_gantry_arm_to_state(q_end,time_from_start)

        return True

    def check_tray_exist(self, task):
        # print('aaaaaaaaaa',self.agv_has_tray[task.agv_number-1],self.find_tray_slot(task.tray_id))
        if not self.agv_has_tray[task.agv_number-1]:
            while not self.find_tray_slot(task.tray_id):
                sleep(0.2)
            tray=self.find_tray_slot(task.tray_id)
            return task.agv_number,tray
        else:
            return None
            
            
    def floor_find_shortest_part(self,target_part_type,agv):
        """
        返回距离最短的零件。输入参数是零件列表和机器人的纵坐标。
        """
        closest_part_no_flip = None
        closest_part_flip = None
        min_distance_no_flip = float('inf')
        min_distance_flip = float('inf')
        closest_part = None  # 初始化最近的零件为 None
        agv_location=location='agv'+str(agv)+'_ks'+str(agv)+'_tray'
        target_part_list_bins = self.search_part_on_bins(target_part_type)
        target_part_list_conveyor=self.search_part_on_conveyor(target_part_type)   
        
        # target_part_list_conveyor = [part for part in target_part_list_conveyor if part.need_flip == False]   #传送带上有需要翻转的，做不了

        print("类型为:",target_part_type,"识别出target_part_list_conveyor有:",len(target_part_list_conveyor))
        
        if target_part_list_bins:
            
            parts_in_floor,parts_in_ceiling=split_parts_by_location(target_part_list_bins)
            
            # if parts_in_ceiling:
            #     closest_part = parts_in_ceiling[-1]  
            #     self.del_part_from_parts_list(closest_part)
            
            if parts_in_floor:
                for part in parts_in_floor:
                    
                    part_location = kitting_robot_park_location[part.location]
                    distance = abs(self.kitting_base_y - part_location[1])  # 计算机器人和零件之间的纵坐标差值
                    dis2 = abs(part_location[1] - kitting_robot_park_location[agv_location][1])  # 计算AGV和零件之间的纵坐标差值
                    total_distance = distance + dis2

                    if part.need_flip:
                        if total_distance < min_distance_flip:  # If the distance is shorter
                            min_distance_flip = total_distance  # Update the shortest distance
                            closest_part_flip = part  # Update the nearest part
                    else:
                        if total_distance < min_distance_no_flip:  # If the distance is shorter
                            min_distance_no_flip = total_distance  # Update the shortest distance
                            closest_part_no_flip = part  # Update the nearest part

                # Select the closest part, prioritizing parts that do not need flipping
                closest_part = closest_part_no_flip if closest_part_no_flip else closest_part_flip
                self.del_part_from_parts_list(closest_part)

        
        if target_part_list_conveyor:
            closest_part = target_part_list_conveyor[-1]
            with self.convey_parts_lock :
                self.convey_parts.remove(closest_part)

            
        if not target_part_list_bins and not target_part_list_conveyor:
            return None
            
        return closest_part


    def filp_part_on_convey(self,part,agv_number,type='kitting'):
        target_bin=find_closest_bin(agv_number)
        self.move_to(target_bin)
        if type=='combined_kitting':
            self.move_to_ceiling(target_bin)
        ref_coord=[self.kitting_base_x,-self.kitting_base_y,self.kitting_base_z]
        curr_coord=[bin_position[target_bin][0],bin_position[target_bin][1]+0.3,bin_position[target_bin][2]+kitting_pick_part_heights_on_bin_agv[part.type]]
        world_target=self.relative_coordinate(ref_coord,curr_coord)
        p4=Rot2Matrix(self.init_rotation, world_target)

        p4[2,3]=p4[2,3] +0.2 
        self.MOV_M(p4,eps =0.01)  
        if 'battery' in part.type:
            p4[1,3]=p4[1,3]-0.01           
        if 'sensor' in part.type:
            p4[1,3]=p4[1,3]-0.02  
        if 'regulator' in part.type:
            p4[1,3]=p4[1,3]-0.02  
        if 'pump' in part.type:
            p4[1,3]=p4[1,3]-0.06   #  
        p4[2,3]=p4[2,3] -0.20
        self.MOV_M(p4,eps =0.01,times=5)  
        sleep(0.2)

        self.set_floor_robot_gripper_state(False)
        
        sleep(1)
        target_part=None
        while not target_part:
            part_list=self.search_part_on_bins(part.type)
            for part in part_list:
                distance = math.sqrt((part.pose.position.x - curr_coord[0]) ** 2 + (part.pose.position.y - curr_coord[1]) ** 2)
                print("距离是:",distance)
                if distance<0.1:
                    target_part=part
                    break
            sleep(0.2)
        if target_part:
            self.grasp_flip_part_on_bins(target_part,agv_number)
        


    def grasp_part_on_bins(self, location, part, flip = False, repick_callback_num = 0):
            target_matrix =self.Pose2Robot(part) 
            position,rpy = Matrix2Pos_rpy(target_matrix)
            target_matrix = Rot2Matrix(self.init_rotation, position)
    
            p1 = copy.deepcopy(target_matrix)
            p2 = copy.deepcopy(target_matrix)
            p2[2,3]=p2[2,3]+0.2


            
            self.set_floor_robot_gripper_state(True)
            self.MOV_M(p2,eps =0.01)  
            
            if 'pump' in part.type:

        
                p1[2,3] = target_matrix[2,3]+kitting_pick_part_heights_on_bin_agv[part.type]-0.039
                self.MOV_M(p1,eps =0.001,times=10)
                # print("抓取走这",p1)
                sleep(2)
            else:
                p1[2,3] = target_matrix[2,3]+kitting_pick_part_heights_on_bin_agv[part.type]-0.040 
                self.MOV_M(p1,eps =0.001,times=5)
                sleep(0.5)

            repick_nums = 0
            while not self.floor_robot_gripper_state.attached:
                if repick_nums >= 5:
                    break
                
                repick_nums = repick_nums + 1
                if 'pump' in part.type:
                    p1[2,3] = p1[2,3]- 0.001
                    print(f"这是第{repick_nums}次抓取,{p1[2,3]}")
                    sleep(1)
                else:
                    p1[2,3] = p1[2,3]- (repick_nums)*0.002
            
                self.MOV_M(p1,eps =0.001,times=10)
                sleep(0.5)

            p1[2,3]=p1[2,3]+0.4
            self.MOV_M(p1,eps =0.01)     
          
                       
    def Pose2Robot(self, part):
        '''
        输入是位置+四元数，输出是机器人直接可以使用的齐次旋转矩阵
        返回零件相对与机器人的位姿-齐次矩阵
        '''
        # print("self.kitting_base_y-----Pose2Robot",self.kitting_base_y)
        ref_coord=[self.kitting_base_x,-self.kitting_base_y,self.kitting_base_z]
        curr_coord=[part.pose.position.x,part.pose.position.y,part.pose.position.z]
        world_target=self.relative_coordinate(ref_coord,curr_coord)
        # print("ref_coord",ref_coord)
        # print("world_target",world_target)
        # print("curr_coord",curr_coord)
        base_target = Pose()
        base_target.position.x=world_target[0]
        base_target.position.y=world_target[1]
        base_target.position.z=world_target[2]
        # base_target.orientation = ee_target_tf.transform.rotation
        base_target.orientation = part.pose.orientation
        # #print base_target.orientation
        target = Pose2Matrix(base_target)
        
        return target

    def relative_coordinate(self,ref_coord, curr_coord):
        # 将参考系坐标和当前坐标转换为numpy数组
        ref_coord = np.array(ref_coord)
        curr_coord = np.array(curr_coord)

        # 计算相对于参考系的坐标
        rel_coord = curr_coord - ref_coord
        # 返回相对于参考系的坐标
        return rel_coord

    def floor_place_part_on_agv(self,location: str,subtask,target_part,cmd_type) -> bool:

        if target_part.need_flip and target_part.location!="conveyor":
            ref_coord=[self.kitting_base_x,-self.kitting_base_y,self.kitting_base_z]   
            curr_coord=[kitting_robot_park_location[location][0]+quad_offsets_[subtask.product_quadrant][0],kitting_robot_park_location[location][1]+quad_offsets_[subtask.product_quadrant][1],kitting_robot_park_location[location][2]]
            world_target=self.relative_coordinate(ref_coord,curr_coord)
            
            
            base_target1 = Pose()
            base_target1.position.x=0.0
            base_target1.position.y=0.0
            base_target1.position.z=0.0
            base_target1.orientation=QuaternionFromRPY(3.1415926,0,0)     #让夹爪旋转180
            part_to_gripper=Pose2Matrix(base_target1)

            # if subtask.product_quadrant==2 or subtask.product_quadrant==4:
            #     p1 = Rot2Matrix(self.flip_rotation['left_roll_0'], world_target)
            #     p1=p1@part_to_gripper
                
            # else:
            p1 = Rot2Matrix(self.flip_rotation['right_roll_0'], world_target)
            p1=p1@part_to_gripper

            p1[2,3] = p1[2,3]+0.05
            if subtask.product_quadrant==2 or subtask.product_quadrant==4:
                p1[1,3]=p1[1,3]+0.06 
                
            else:
                p1[1,3]=p1[1,3]+0.06   # 翻转需要向右移动 
            self.MOV_M(p1,eps =0.01) 
            
            if 'pump' in target_part.type:
                p1[2,3] = p1[2,3]
            else:
                p1[2,3] = p1[2,3]-0.05
            
            if cmd_type=="combined_kitting":
                p1[2,3]=p1[2,3]-0.02
            
            sleep(1)
                


            # p1=p1@part_to_gripper
            # self.MOV_M(p1,eps =0.01)   
            # p1[1,3]=p1[2,3]+0.03
            # p1[2,3]=p1[2,3]+kitting_pick_part_heights_on_bin_agv[target_part.type]+0.1
            # self.MOV_M(p1,eps =0.01,times=3)    
            self.set_floor_robot_gripper_state(False)
            p1[2,3]=p1[2,3]+0.3
            self.MOV_M(p1,eps =0.01) 
   
        else:
            ref_coord=[self.kitting_base_x,-self.kitting_base_y,self.kitting_base_z]   
            curr_coord=[kitting_robot_park_location[location][0]+quad_offsets_[subtask.product_quadrant][0],kitting_robot_park_location[location][1]+quad_offsets_[subtask.product_quadrant][1],kitting_robot_park_location[location][2]]
            world_target=self.relative_coordinate(ref_coord,curr_coord)
            # print("quadrant-----ref_coord",ref_coord)
            # print("world_target",world_target)
            # print("curr_coord",curr_coord)
            base_target = Pose()
            pose = Pose()
            base_target.position.x=world_target[0]
            base_target.position.y=world_target[1]
            base_target.position.z=world_target[2]-0.1
            q=quaternion_from_euler(-1.5707963267948966, 1.5707963267948966, 0)   
            base_target.orientation.x=q[0]
            base_target.orientation.y=q[1]
            base_target.orientation.z=q[2]
            base_target.orientation.w=q[3]
            target = Pose2Matrix(base_target)

            p1 = copy.deepcopy(target)
            p2 = copy.deepcopy(target)
            p2[2,3] = p2[2,3]+0.30
            self.MOV_M(p2,eps =0.01) 
            p1[2,3] = target[2,3]+0.20
            
            if 'pump' in target_part.type:
                p1[2,3] = p1[2,3]
            else:
                p1[2,3] = p1[2,3]-0.05
            
            if cmd_type=="combined_kitting":
                p1[2,3]=p1[2,3]-0.02
            q_begin = copy.deepcopy(self.kitting_arm_joint_states)
            # print("target",target,"-------------q_begin",q_begin)
            target = IKinematic(p1,q_begin,A_k)
            # print("target_Ik",target)
            delta = [abs(q_begin[i] - target[i]) for i in range(0,len(q_begin))]
            ##print delta
            distance = max(delta)
            time_from_start = distance/kitting_angle_velocity*2
            traj = traj_generate(self.kitting_arm_joint_names,q_begin,target,time_from_start)
            self.move(self.floor_action_client,traj)
            sleep(time_from_start)
            self.set_floor_robot_gripper_state(False)
            self.MOV_M(p2,eps =0.01) 
            
        self.kitting_robot_init("init_state")  
        return world_target
        

    def handle_faultu_part(self,check_info,subtask,world_target,target_part):
        if self.check_faluty_part(check_info,subtask.product_quadrant): 
            while self.floor_robot_gripper_state.attached:
                sleep(0.2)
                print("等待夹爪正常")
            self.set_floor_robot_gripper_state(True)
            p2=Rot2Matrix(self.init_rotation, world_target)
           
            if target_part.need_flip:                                          # 在agv上抓取
                
                if 'pump' in target_part.type:
                    p2[2,3] = p2[2,3]+kitting_pick_part_heights_on_bin_agv[target_part.type]-0.029         
                    self.MOV_M(p2,eps =0.01,times=3) 
                    sleep(0.5)
                else:
                    p2[2,3] = p2[2,3]+kitting_pick_part_heights_on_bin_agv[target_part.type]-0.026
                    self.MOV_M(p2,eps =0.01,times=3) 

                repick_nums = 0
                while not self.floor_robot_gripper_state.attached:
                    if repick_nums >= 5:
                        print("没有抓到")
                        break
                        
                    repick_nums = repick_nums + 1
                    if 'pump' in target_part.type:
                        p2[2,3] = p2[2,3]- (repick_nums)*0.001
                    else:
                        p2[2,3] = p2[2,3]- (repick_nums)*0.001
                
                    self.MOV_M(p2,eps =0.01,times=3)
                    sleep(0.5)
                    
            else:
                if 'pump' in target_part.type:
                    p2[2,3] = p2[2,3]+kitting_pick_part_heights_on_bin_agv[target_part.type]-0.029
                else:
                    p2[2,3] = p2[2,3]+kitting_pick_part_heights_on_bin_agv[target_part.type]-0.029

                self.MOV_M(p2,eps =0.01,times=10) 
                sleep(0.5)
                    
                repick_nums = 0
                while not self.floor_robot_gripper_state.attached:
                    if repick_nums >= 5:
                        print("没有抓到")
                        break
                        
                    repick_nums = repick_nums + 1
                    print(f"第{repick_nums}次抓取")
                    
                    if 'pump' in target_part.type:
                        p2[2,3] = p2[2,3]- (repick_nums)*0.001
                    else:
                        p2[2,3] = p2[2,3]- (repick_nums)*0.001
                
                    self.MOV_M(p2,eps =0.01,times=10)
                    sleep(0.5)
                    
            p2[2,3]=p2[2,3]+0.2 
            self.MOV_M(p2,eps =0.01)
            self.kitting_robot_init("standby" )  
            self.move_to("can")  
            self.kitting_robot_init("init_state")   
            self.set_floor_robot_gripper_state(False) 
            
            return True
            
        return False
    
    def combined_robot_place_part_on_kit_tray(self,location: str,quadrant,target_part) -> bool:

        ref_coord=[self.kitting_base_x,-self.kitting_base_y,self.kitting_base_z]   
        curr_coord=[kitting_robot_park_location[location][0]+quad_offsets_[quadrant][0],kitting_robot_park_location[location][1]+quad_offsets_[quadrant][1],kitting_robot_park_location[location][2]]
        world_target=self.relative_coordinate(ref_coord,curr_coord)
        # print("quadrant-----ref_coord",ref_coord)
        # print("world_target",world_target)
        # print("curr_coord",curr_coord)
        base_target = Pose()
        pose = Pose()
        base_target.position.x=world_target[0]
        base_target.position.y=world_target[1]
        base_target.position.z=world_target[2]-0.1
        # q=quaternion_from_euler(-1.5707963267948966, 1.5707963267948966, 0)   # x=-1.5707963267948966, y=1.5707963267948966 ,z=0,z旋转夹爪，正常抓取用这个
        #x=-1.5707963267948966, y=0 ,z=0,把物体立起来
        #x=0, 3.14159, 1.5707963267948966 ,z=0,把物体侧过来，相当于Flipped
        q=quaternion_from_euler(-1.5707963267948966, 1.5707963267948966, 0)   
        base_target.orientation.x=q[0]
        base_target.orientation.y=q[1]
        base_target.orientation.z=q[2]
        base_target.orientation.w=q[3]
        target = Pose2Matrix(base_target)

        p1 = copy.deepcopy(target)
        p2 = copy.deepcopy(target)
        
        if 'pump' in target_part.type:
            p1[2,3] = target[2,3]+0.20
        else:
            p1[2,3] = target[2,3]+0.15
            
        q_begin = copy.deepcopy(self.kitting_arm_joint_states)
        # print("target",target,"-------------q_begin",q_begin)
        target = IKinematic(p1,q_begin,A_k)
        # print("target_Ik",target)
        delta = [abs(q_begin[i] - target[i]) for i in range(0,len(q_begin))]
        ##print delta
        distance = max(delta)
        time_from_start = distance/kitting_angle_velocity*2
        traj = traj_generate(self.kitting_arm_joint_names,q_begin,target,time_from_start)
        self.move(self.floor_action_client,traj)
        sleep(time_from_start)
        self.set_floor_robot_gripper_state(False)


        self.MOV_M(self.init_matrix,eps =0.01) 
        
        return True

    def floor_grasp_on_bins(self,target_part,subtask):
        
        if self.floor_robot_gripper_state.type == "tray_gripper":
            if kitting_robot_park_location[target_part.location][1]>0:
                station="kts2"
            else:
                station="kts1"
            self.ChangeGripper(station, "parts")
            
        self.move_to(target_part.location)
        self.floor_robot_info.location=target_part.location
        
        if target_part.need_flip:
            self.grasp_flip_part_on_bins(target_part,subtask.agv_number)
            
        else:
            self.grasp_part_on_bins(target_part.location,target_part)

        self.kitting_robot_init("standby" )   


    def combinde_grasp_on_bins(self,target_part,quadrant,agv_number):
        
        # Change gripper to tray gripper
        if self.floor_robot_gripper_state.type != "part_gripper":
            if kitting_robot_park_location[target_part.location][1]>0:
                station="kts2"
            else:
                station="kts1"
            self.ChangeGripper(station, "parts")
            
        self.move_to(target_part.location)
        
        self.pick_part_on_bin_agv(target_part.location,target_part)
        location='agv'+str(agv_number)+'_ks'+str(agv_number)+'_tray'   
        self.move_to(location)
        if self.floor_robot_gripper_state.attached:
            return self.combined_robot_place_part_on_kit_tray(location,quadrant,target_part)
        else:
            return False

    def floor_grasp_on_conveyor(self,target_part):
        
        # Change gripper to tray gripper
        if self.floor_robot_gripper_state.type != "part_gripper":
            self.ChangeGripper("kts1", "parts")
        
        self.kitting_robot_init("conveyor_insert" )                                                                             # 1. 进入抓取姿态
        self.set_floor_robot_gripper_state(True)
                                                                                                            
        target_matrix = self.pose2robot(target_part)                                                                           # 2. 计算最佳抓取位置f_y，对应的时间t_best
        position,rpy = Matrix2Pos_rpy(target_matrix)
        target_matrix = Rot2Matrix(self.conveyor_insert_matrix, position)
        target_matrix[2,3]=target_matrix[2,3] + kitting_pick_part_heights_on_bin_agv[target_part.type] + vacuum_gripper_height-0.050
        
        q_begin = self.kitting_arm_joint_states
        target = IKinematic(target_matrix,q_begin,A_k)
        delta = [abs(q_begin[i] - target[i]) for i in range(0,len(q_begin))]
        distance = max(delta)
        angle_time = distance/kitting_angle_velocity
        target_part_time = target_part.time_stamp
        target_part_y = target_part.pose.position.y
        current_time = self.get_clock().now().nanoseconds/1e9

        c_y = -(current_time - target_part_time)*conveyor_vel+target_part_y
    
        if c_y <= self.kitting_base_y:
            t_best = (self.kitting_base_y- c_y)/(kitting_velocity +conveyor_vel)  
        else:
            t_best = (c_y - self.kitting_base_y)/(kitting_velocity - conveyor_vel) 
        print("t_best=",t_best,"angle_time=",angle_time,"检测时候零件的位置:",target_part_y)
        if angle_time >= t_best:
            t_best = angle_time
        
        f_y = c_y - t_best*conveyor_vel +0.03  
        # if c_y <= self.kitting_base_y:
        #     f_y = c_y - t_best*conveyor_vel +0.12       #0.03是      
        # else:                                                
        #     f_y = c_y - t_best*conveyor_vel +0.06       #0.03是   
            
        end_point =-f_y                                                                                                      # 3. 运动到对应位置，进行最佳抓取
        print("打印一下需要去的位置",end_point,"当前零件在",c_y,"当前机械臂在",self.kitting_base_y)
        if end_point>4.15:                                       # 超出范围了，不抓了
            self.kitting_robot_init("standby" )
            return False

        q_begin = self.kitting_base_y 
        distance = abs(q_begin - end_point)
        q_begin = [q_begin]
        q_end= [end_point]   
        #移动轨迹
        traj = traj_generate(self.linear_joint_names,q_begin,q_end,t_best)
        self.move(self.linear_action_client,traj)
        p1_mat = copy.deepcopy(target_matrix) 
        
        if target_part.need_flip:
            p1_mat[2,3]=p1_mat[2,3]-kitting_pick_part_heights_on_bin_agv[target_part.type]
        
        p1_mat[2,3]=p1_mat[2,3]+0.1
        self.MOV_M(p1_mat,time_set=t_best*0.9)
        p1_mat[2,3]=p1_mat[2,3]-0.00045-0.1
        
        q_begin = [self.kitting_base_y ]       #再定位一次
        current_time = self.get_clock().now().nanoseconds/1e9
        q_end= [ (current_time - target_part_time)*conveyor_vel-target_part_y] 
        if q_begin[0]>q_end[0]:

            t=abs(q_begin[0]-q_end[0])/1.0 
        else:
            t=abs(q_begin[0]-q_end[0])/0.6
        q_end[0]=q_end[0]+0.01 # 0.01是误差距离

        traj = traj_generate(self.linear_joint_names,q_begin,q_end,t)
        self.move(self.linear_action_client,traj)
        
        self.MOV_M(p1_mat,time_set=0.1*t_best)
        

        

        self.kitting_robot_init("standby")
                                                                            
        num_attempts = 0                                
        MAX_ATTEMPTS = 3
        while not self.floor_robot_gripper_state.attached and num_attempts < MAX_ATTEMPTS and rclpy.ok():            # 4. 保持和零件同速2s,进行二次抓取
            
            num_attempts += 1  
            print (f"第{num_attempts}次抓取")      
            current_time = self.get_clock().now().nanoseconds/1e9
            p_y = -(current_time- target_part_time)*conveyor_vel+target_part_y 
            #计算角度时间
            p1_mat = copy.deepcopy(target_matrix) 
            q_begin = self.kitting_arm_joint_states
            p1_mat[1,3] = p1_mat[1,3]+0.20
            if target_part.need_flip:
                p1_mat[2,3]=p1_mat[2,3]-kitting_pick_part_heights_on_bin_agv[target_part.type]
            p1_mat[2,3] = p1_mat[2,3]-0.001
            result = IKinematic(p1_mat,q_begin,A_k)
            delta = [abs(q_begin[i] - result[i]) for i in range(0,len(q_begin))]
            distance = max(delta)
            angle_time = distance/kitting_angle_velocity


            if p_y <= self.kitting_base_y:
                t_best = (self.kitting_base_y- p_y)/(kitting_velocity +conveyor_vel)  
            else:
                t_best = (p_y - self.kitting_base_y)/(kitting_velocity - conveyor_vel)
            if angle_time >= t_best:
                t_best = angle_time
            if t_best<=1:
                t_best = 1
            end_point =-(p_y-t_best*conveyor_vel-0.1)
            q_begin = self.kitting_base_y 
            distance = abs(q_begin - end_point)
            move_time = distance/kitting_velocity
            q_begin = [q_begin]
            q_end= [end_point]   
            traj = traj_generate(self.linear_joint_names,q_begin,q_end,t_best)
            self.move(self.linear_action_client,traj)
            
            q_begin = self.kitting_arm_joint_states
            traj = traj_generate(self.kitting_arm_joint_names,q_begin,result,t_best)
            self.move(self.floor_action_client,traj)
            sleep(t_best)        
            self.kitting_robot_init("standby" )
            
        return True

    # def floor_grasp_on_conveyor(self,target_part):
        
    #     # Change gripper to tray gripper
    #     if self.floor_robot_gripper_state.type != "part_gripper":
    #         self.ChangeGripper("kts1", "parts")
        
    #     self.kitting_robot_init("conveyor_insert" )
        
    #     target_matrix = self.pose2robot(target_part)
    #     position,rpy = Matrix2Pos_rpy(target_matrix)
    #     target_matrix = Rot2Matrix(self.conveyor_insert_matrix, position)
    #     target_matrix[2,3]=target_matrix[2,3] + kitting_pick_part_heights_on_bin_agv[target_part.type] + vacuum_gripper_height-0.0495
        
    #     p1_mat = copy.deepcopy(target_matrix)   
    #     p2_mat = copy.deepcopy(target_matrix)     
    #     self.set_floor_robot_gripper_state(True)
    #     q_begin = self.kitting_arm_joint_states
    #     target = IKinematic(target_matrix,q_begin,A_k)
    #     delta = [abs(q_begin[i] - target[i]) for i in range(0,len(q_begin))]
    #     distance = max(delta)
    #     angle_time = distance/kitting_angle_velocity
        
        
   
        
    #     current_time = self.get_clock().now().nanoseconds/1e9
    #     target_part_time = target_part.time_stamp
    #     target_part_y = target_part.pose.position.y

    #     c_y = -(current_time - target_part_time)*conveyor_vel+target_part_y
    
    #     if c_y <= self.kitting_base_y:
    #         t_best = (self.kitting_base_y- c_y)/(kitting_velocity - conveyor_vel)  
    #     else:
    #         t_best = (c_y - self.kitting_base_y)/(kitting_velocity + conveyor_vel) 
    #     if angle_time >= t_best:
    #         t_best = angle_time
    #     f_y = c_y - t_best*conveyor_vel +0.06

    #     # if f_y>=1.8:
    #     #     t_best = (c_y - 1.7-0.17)/conveyor_vel
    #     #     f_y = c_y - t_best*conveyor_vel + 0.1
            
    #     # if f_y < conveyor_end:
    #     #     return False
        
    #     print("      c_y=",c_y,"   t_best= ",t_best,"   f_y=",f_y)
        
    #     #抓取轨迹
    #     traj = traj_generate(self.kitting_arm_joint_names,q_begin,target,t_best)
    #     self.move(self.floor_action_client,traj)     

    #     end_point =-f_y
    #     q_begin = self.kitting_base_y 
    #     distance = abs(q_begin - end_point)
    #     q_begin = [q_begin]
    #     q_end= [end_point]   
    #     #移动轨迹
    #     traj = traj_generate(self.linear_joint_names,q_begin,q_end,t_best)
    #     self.move(self.linear_action_client,traj)
    #     sleep(t_best)
        
    #     # 运动时抓取
    #     q_begin = self.kitting_base_y 
    #     distance = 0.4
    #     end_point =q_begin+0.4
    #     q_begin = [q_begin]
    #     q_end= [end_point]   
    #     t=distance/conveyor_vel
    #     #移动轨迹
    #     traj = traj_generate(self.linear_joint_names,q_begin,q_end,t)
    #     self.move(self.linear_action_client,traj)
    #     p_pre=deepcopy(p1_mat)
    #     p_pre[2,3]=p_pre[2,3]+0.1
    #     self.MOV_M(p_pre,time_set=0.9*t)
        
    #     p1_mat[2,3]=p1_mat[2,3]-0.00045
    #     self.MOV_M(p1_mat,time_set=0.1*t)

        
        
    #     num_attempts = 0
    #     MAX_ATTEMPTS = 2
    #     while not self.floor_robot_gripper_state.attached and num_attempts < MAX_ATTEMPTS :
    #         p1_mat[2,3]=p1_mat[2,3]-0.00005
    #         self.MOV_M(p1_mat,times=10)
    #         num_attempts += 1  
    #         print (f"第{num_attempts}次抓取")   
    #         sleep(0.2)

    #     print ("最佳点抓取") 
    #     if  self.floor_robot_gripper_state.attached:
    #         p2_mat[2,3]=p2_mat[2,3]+0.4
    #         self.MOV_M(p2_mat,times=2)
            
    #     # sleep(5)
            
        
    #      #3. 最佳点抓取不成功， 多次尝试抓取
    #     num_attempts = 0
    #     MAX_ATTEMPTS = 3
    #     while not self.floor_robot_gripper_state.attached and num_attempts < MAX_ATTEMPTS and rclpy.ok():
            
    #         num_attempts += 1  
    #         print (f"第{num_attempts}次抓取")      
    #         current_time = self.get_clock().now().nanoseconds/1e9
    #         p_y = -(current_time- target_part_time)*conveyor_vel+target_part_y 
    #         #计算角度时间
    #         q_begin = self.kitting_arm_joint_states
    #         p1_mat[1,3] = -0.17
    #         p1_mat[2,3] = p1_mat[2,3]-0.001*num_attempts
    #         result = IKinematic(p1_mat,q_begin,A_k)
    #         # #print "IKinematic_result:",target
    #         delta = [abs(q_begin[i] - result[i]) for i in range(0,len(q_begin))]
    #         distance = max(delta)
    #         angle_time = distance/kitting_angle_velocity


    #         if p_y <= self.kitting_base_y:
    #             t_best = (self.kitting_base_y- p_y)/(kitting_velocity - conveyor_vel)  
    #         else:
    #             t_best = (p_y - self.kitting_base_y)/(kitting_velocity + conveyor_vel)
    #         if angle_time >= t_best:
    #             t_best = angle_time
    #         if t_best<=1:
    #             t_best = 1
    #         end_point =-(p_y-t_best*conveyor_vel-0.17)
    #         q_begin = self.kitting_base_y 
    #         distance = abs(q_begin - end_point)
    #         move_time = distance/kitting_velocity
    #         q_begin = [q_begin]
    #         q_end= [end_point]   
    #         traj = traj_generate(self.linear_joint_names,q_begin,q_end,t_best)
    #         self.move(self.linear_action_client,traj)
            
    #         q_begin = self.kitting_arm_joint_states
    #         traj = traj_generate(self.kitting_arm_joint_names,q_begin,result,t_best)
    #         self.move(self.floor_action_client,traj)
    #         sleep(t_best)        
    #         print("      c_y=",c_y,"   t_best= ",t_best,"   f_y=",f_y)
            
        # if  self.floor_robot_gripper_state.attached:
        #     self.kitting_robot_init("standby" )
            
        
          
    def pose2robot(self, part):
        '''
        输入是位置+四元数，输出是机器人直接可以使用的齐次旋转矩阵
        返回零件相对与机器人的位姿-齐次矩阵
        '''
        print("self.kitting_base_y-----Pose2Robot",self.kitting_base_y)
        ref_coord=[self.kitting_base_x,-self.kitting_base_y,self.kitting_base_z]
        curr_coord=[part.pose.position.x,part.pose.position.y,part.pose.position.z]
        world_target=self.relative_coordinate(ref_coord,curr_coord)        
        print("ref_coord",ref_coord)
        print("world_target",world_target)
        print("curr_coord",curr_coord)
        base_target = Pose()
        base_target.position.x=world_target[0]
        base_target.position.y=0.0  #已经同步
        base_target.position.z=world_target[2]
        # base_target.orientation = ee_target_tf.transform.rotation
        base_target.orientation = part.pose.orientation
        # #print base_target.orientation
        target = Pose2Matrix(base_target)
        
        return target
    
    
    def FrameWorldPose(self,frame_id):
        t =TransformStamped()
        pose = Pose()

        try:
            t= self.tf_buffer_floor .lookup_transform("slide_bar", frame_id,  rclpy.time.Time())
        except TransformException as ex:
            self.get_logger().info(f'Could not transform assembly station  to world: {ex}')
            return

        # print("---------",t)
        pose.position.x = t.transform.translation.x
        pose.position.y = t.transform.translation.y
        pose.position.z = t.transform.translation.z
        pose.orientation = t.transform.rotation

        target = Pose2Matrix(pose)
        return target
    
    def BuildPose(self, x, y, z, orientation):
        pose = Pose()
        pose.position.x = float(x)
        pose.position.y = float(y)
        pose.position.z = float(z)
        pose.orientation.x=orientation[1]
        pose.orientation.y=orientation[2]
        pose.orientation.z=orientation[3]
        pose.orientation.w=orientation[0]
        target = Pose2Matrix(pose)
        
        
        return target
    

    def FloorRobotPickandPlaceTray(self,cmd,tray, agv_num: int) -> bool:
        
        if self.floor_robot_gripper_state.type != "tray_gripper":
            if tray_slots_location[tray][1]>0:
                location='kts2'
            else:
                location='kts1'
             
            self.ChangeGripper(location, "trays")

        ref_coord=[self.kitting_base_x,-self.kitting_base_y,self.kitting_base_z]
        curr_coord=[tray_slots_location[tray][0],tray_slots_location[tray][1],tray_slots_location[tray][2]]
        world_target=self.relative_coordinate(ref_coord,curr_coord)  
        
        target_matrix = Rot2Matrix(self.init_rotation, world_target)
        self.set_floor_robot_gripper_state(True)  
        p1 = copy.deepcopy(target_matrix)
        p1[2,3]=p1[2,3]-0.036      
        self.MOV_M(p1,eps =0.01,times=3)


        pick_num=0
        while not self.floor_robot_gripper_state.attached:
            if pick_num>10:
                break
            pick_num+=1
            p1[2,3]=p1[2,3]-0.001
            self.MOV_M(p1,eps =0.01,times=3)
            sleep(0.1)
            
        p3=copy.deepcopy(p1)
        p3[2,3]=p3[2,3]+0.4
        self.MOV_M(p3,eps =0.01)
        self.move_to("agv"+str(agv_num)+"_ks"+str(agv_num)+"_tray")
        self.floor_robot_info.location="agv"+str(agv_num)+"_ks"+str(agv_num)+"_tray"

    # Move up slightly
        target_matrix = self.FrameWorldPose("agv" + str(agv_num) + "_tray")
        position,rpy = Matrix2Pos_rpy(target_matrix)
        pose=Pose()
        pose.position.x=-position[0]
        pose.position.y=-(position[1]-self.kitting_base_y)
        pose.position.z=position[2]+0.2
        q=quaternion_from_euler(-1.5707963267948966, 1.5707963267948966, 1.5707963267948966)
        pose.orientation.x=q[0]
        pose.orientation.y=q[1]
        pose.orientation.z=q[2]
        pose.orientation.w=q[3]
        target_=Pose2Matrix(pose)
        
        if tray in [ "slot4","slot5","slot6"]:
            need_yaw=3.1415888848633897
            base_target = Pose()
            base_target.position.x=0.0
            base_target.position.y=0.0
            base_target.position.z=0.0
            base_target.orientation=QuaternionFromRPY(need_yaw,0,0)     #让夹爪旋转
            part_to_gripper=Pose2Matrix(base_target)
            target_=target_@part_to_gripper 
    
        p1 = copy.deepcopy(target_)
        self.MOV_M(p1)
        p2 = copy.deepcopy(p1)
        p2[2,3]=p2[2,3]-0.22
        self.MOV_M(p2)
        sleep(0.2)
        
        self.set_floor_robot_gripper_state(False)
        # print("object_attached,self.gripper_enabled",self.object_attached,self.gripper_enabled)      
        # while self.object_attached:
        # p2[2,3]=p2[2,3]+0.40         
        # self.MOV_M(p2)
        sleep(0.2)
        
        self.ChangeGripper(location, "parts")
        self.kitting_robot_init("standby")
        self.co_tray_flag[cmd.command_id.order_id]=True
        sleep(0.2)
        
        
        


        
    def move(self,action_client,goal_trajectory): 
            
        goal_msg = FollowJointTrajectory.Goal()

        # Fill in the trajectory message
        goal_msg.trajectory = JointTrajectory()
        goal_msg.trajectory.joint_names = goal_trajectory.joint_names
        goal_msg.trajectory.points = goal_trajectory.points
        
        # future.add_done_callback(self.goal_response_callback) 
        action_client.wait_for_server()
        self._floor_robot_send_goal_future = action_client.send_goal_async(
            goal_msg)
        
        self._floor_robot_send_goal_future.add_done_callback(
            self.floor_robot_goal_response_callback)

    def floor_robot_goal_response_callback(self, future):
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        # self.get_logger().info('Goal accepted')

        self._floor_robot_get_result_future = goal_handle.get_result_async()
        self._floor_robot_get_result_future.add_done_callback(
            self.floor_robot_get_result_callback)
    
    def ceiling_robot_goal_response_callback(self, future):
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')

        self._ceiling_robot_get_result_future = goal_handle.get_result_async()
        self._ceiling_robot_get_result_future.add_done_callback(
            self.ceiling_robot_get_result_callback)

    def floor_robot_get_result_callback(self, future):
        result = future.result().result
        result: FollowJointTrajectory.Result

        if result.error_code == FollowJointTrajectory.Result.SUCCESSFUL:
            # self.get_logger().info("Move succeeded")
            pass
        else:
            self.get_logger().error(result.error_string)

        self.floor_robot_at_home = True

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info(f'Goal rejected')
            return
        self.get_logger().info(f'Goal  accepted')
    

    def ChangeGripper(self, station, gripper_type):
        self.kitting_robot_init("bin_agv_insert_joint")
        # 将夹爪移动到换爪器上
        
        
        self.move_to(station)   
        self.floor_robot_info.location=station


        target_matrix = self.FrameWorldPose(station +"_tool_changer_" + gripper_type + "_frame")
        position,rpy = Matrix2Pos_rpy(target_matrix)
        position[0]=-position[0]
        position[1]=-(position[1]+kitting_robot_park_location[station][1])
        position[2]=position[2]-0.035
        target_matrix = Rot2Matrix(self.init_rotation, position)

        p1 = copy.deepcopy(target_matrix)
        p2 = copy.deepcopy(target_matrix)
        p1[2,3]=p1[2,3]+0.1
        self.MOV_M(p1)
        self.MOV_M(p2)

        sleep(0.1)
        p3=copy.deepcopy(p2)
        p3[2,3]=p3[2,3]+0.4
        

        # 调用服务更换夹爪
        request = ChangeGripper.Request()

        if gripper_type == "trays":
            request.gripper_type =ChangeGripper.Request.TRAY_GRIPPER
        elif gripper_type == "parts":
            request.gripper_type = ChangeGripper.Request.PART_GRIPPER

        future = self.floor_robot_tool_changer_.call_async(request)
        with self.spin_lock:
            rclpy.spin_until_future_complete(self, future)

        if future.result() is None:
            self.get_logger().error("Error calling gripper change service")
            return False

        # if not future.result().success:
        #     self.get_logger().error("Gripper change service failed")
        #     return False
        
        while not future.result().success:
            sleep(0.1)
        else:
            self.MOV_M(p3,eps =0.01)

        return True
    
        
    def grasp_flip_part_on_bins(self,part,agv_number,type='kitting'):
        
        self.kitting_robot_init("bin_agv_insert_joint") 
        target_matrix =self.Pose2Robot(part) 
        position,rpy = Matrix2Pos_rpy(target_matrix)
        target_matrix = Rot2Matrix(self.init_rotation, position)
        self.set_floor_robot_gripper_state(True)
        
        p1 = copy.deepcopy(target_matrix)
        p2 = copy.deepcopy(target_matrix)
        p3 = copy.deepcopy(target_matrix)
        p1[2,3]=p1[2,3]+0.2
        p3[2,3]=p3[2,3]+0.3             #  提高夹爪，避免碰撞
        
        self.MOV_M(p1,eps =0.01)  
        

        # print(f"零件{part.type}翻转时,target_matrix高度为:",target_matrix[2,3])
        repick_nums = 0
        while not self.floor_robot_gripper_state.attached:
            if repick_nums >= 10:
                break
                
            repick_nums = repick_nums + 1
            if 'pump' in part.type:
                p1[2,3] = target_matrix[2,3]-0.040 - (repick_nums-1)*0.001
            else:
                p1[2,3] = target_matrix[2,3]-0.039 - (repick_nums-1)*0.001
            p2[2,3] = target_matrix[2,3]-0.035
        
            self.MOV_M(p1,eps =0.01,times=10)
            print(f"这是第{repick_nums}次尝试抓取")
            self.MOV_M(p2,eps =0.01)
            
            
            if self.floor_robot_gripper_state.attached:
                self.MOV_M(p3,eps =0.01)  
                break

        target_bin=find_closest_bin(agv_number)
        if part.location!=target_bin:
            self.move_to(target_bin)
            if type=='combined_kitting':
                self.move_to_ceiling(target_bin)
        ref_coord=[self.kitting_base_x,-self.kitting_base_y,self.kitting_base_z]
        curr_coord=[bin_position[target_bin][0],bin_position[target_bin][1]+0.3,bin_position[target_bin][2]+kitting_pick_part_heights_on_bin_agv[part.type]]
        world_target=self.relative_coordinate(ref_coord,curr_coord)
        p4=Rot2Matrix(self.init_rotation, world_target)

        print("                当前的角度是 :",GetYaw(part.pose.orientation))          # 放置      
        need_yaw=3.1415888848633897-abs(GetYaw(part.pose.orientation))
        base_target = Pose()
        base_target.position.x=0.0
        base_target.position.y=0.0
        base_target.position.z=0.0
        
        add_yaw=0
        if 'sensor' in part.type:
            add_yaw=-1.57  
        
        base_target.orientation=QuaternionFromRPY(need_yaw+add_yaw,0,0)     #让夹爪旋转
        part_to_gripper=Pose2Matrix(base_target)
        p4=p4@part_to_gripper 
        p4[2,3]=p4[2,3] +0.2 
        self.MOV_M(p4,eps =0.01)  
        if 'battery' in part.type:
            p4[1,3]=p4[1,3]-0.01           
        if 'sensor' in part.type:
            p4[1,3]=p4[1,3]-0.02  
        if 'regulator' in part.type:
            p4[1,3]=p4[1,3]-0.02  
        if 'pump' in part.type:
            p4[1,3]=p4[1,3]-0.06   #  
        p4[2,3]=p4[2,3] -0.20
        self.MOV_M(p4,eps =0.01,times=5)  
        sleep(0.2)

        self.set_floor_robot_gripper_state(False)
        
        sleep(0.5)      #夹爪切换需要等待
        
        p3 = Rot2Matrix(self.flip_rotation['right_roll_0'], world_target)
        p5=copy.deepcopy(p3)
        
        if 'battery' in part.type:

            p5[1,3]=p5[1,3]+0.08
            p5[2,3]=p5[2,3]+0.1
            self.MOV_M(p5,eps =0.01)  
            p5[2,3]=p5[2,3]-kitting_pick_part_heights_on_bin_agv[part.type]-0.11
            self.MOV_M(p5,eps =0.01)  
            self.set_floor_robot_gripper_state(True)
            p5[1,3]=p5[1,3]-0.054
            self.MOV_M(p5,eps =0.01,times=5) 

            pick_num=0
            while not self.floor_robot_gripper_state.attached:
                if pick_num>10:
                    break
                print(f"这是第{pick_num}次尝试抓取")
                pick_num+=1
                p5[1,3]=p5[1,3]-0.001
                self.MOV_M(p5,eps =0.01,times=5) 
                sleep(0.1) 

        if 'pump' in part.type:

            p5[1,3]=p5[1,3]+0.08
            p5[2,3]=p5[2,3]+0.15
            self.MOV_M(p5,eps =0.01,times=3)  
            p5[2,3]=p5[2,3]-kitting_pick_part_heights_on_bin_agv[part.type]-0.12
            self.MOV_M(p5,eps =0.01,times=3)  
            self.set_floor_robot_gripper_state(True)
            p5[1,3]=p5[1,3]-0.065
            self.MOV_M(p5,eps =0.01,times=10) 

            pick_num=0
            while not self.floor_robot_gripper_state.attached:
                if pick_num>10:
                    break
                pick_num+=1
                p5[1,3]=p5[1,3]-0.001*pick_num
                self.MOV_M(p5,eps =0.01,times=5) 
                sleep(0.1) 

        if 'regulator' in part.type:

            p5[1,3]=p5[1,3]+0.05
            p5[2,3]=p5[2,3]+0.1
            self.MOV_M(p5,eps =0.01)  
            p5[2,3]=p5[2,3]-kitting_pick_part_heights_on_bin_agv[part.type]-0.1
            self.MOV_M(p5,eps =0.01)  
            self.set_floor_robot_gripper_state(True)
            p5[1,3]=p5[1,3]-0.035
            self.MOV_M(p5,eps =0.01,times=5) 

            pick_num=0
            while not self.floor_robot_gripper_state.attached:
                if pick_num>10:
                    break
                pick_num+=1
                p5[1,3]=p5[1,3]-0.001*pick_num
                self.MOV_M(p5,eps =0.01,times=5) 
                sleep(0.1) 
            

        if 'sensor' in part.type:

            p5[1,3]=p5[1,3]+0.08
            p5[2,3]=p5[2,3]+0.1
            self.MOV_M(p5,eps =0.01)  
            p5[2,3]=p5[2,3]-kitting_pick_part_heights_on_bin_agv[part.type]-0.12
            self.MOV_M(p5,eps =0.01)  
            self.set_floor_robot_gripper_state(True)
            p5[1,3]=p5[1,3]-0.060
            self.MOV_M(p5,eps =0.01,times=5) 

            pick_num=0
            while not self.floor_robot_gripper_state.attached:
                if pick_num>10:
                    break
                pick_num+=1
                p5[1,3]=p5[1,3]-0.001*pick_num
                self.MOV_M(p5,eps =0.01,times=5) 
                sleep(0.1) 


        if type=="combined_kitting":
            p5[2,3]=p5[2,3]+0.1
            self.MOV_M(p5,eps =0.01)  
            base_target1 = Pose()
            base_target1.position.x=0.0
            base_target1.position.y=0.0
            base_target1.position.z=0.0
            base_target1.orientation=QuaternionFromRPY(3.1415926,0,0)     #让夹爪旋转180
            part_to_gripper=Pose2Matrix(base_target1)
            p5=p5@part_to_gripper
            self.MOV_M(p5,eps =0.01,times=3)  
            p5[2,3]=p5[2,3]-0.05
            self.MOV_M(p5,eps =0.01,times=5)  # 放慢速度别砸  
            sleep(0.2)
            self.set_floor_robot_gripper_state(False)
            
            p5[2,3]=p5[2,3]+0.2
            self.MOV_M(p5,eps =0.01)              
            self.kitting_robot_init("standby") 
            self.floor_has_flip=True
            
        else:
            p5[2,3]=p5[2,3]+0.2
            self.MOV_M(p5,eps =0.01)  
            self.kitting_robot_init("standby") 


    def adjust_part_on_bin_agv(self, part, x,y,flip = True):
        '''
        从当前的位置抓起来，放到目标位置
        part 原目标
        part_n 目标文件
        '''
        
        part_n=copy.deepcopy(part)
        part_n.pose.position.x=x
        part_n.pose.position.y=y
        target_matrix =self.Pose2Robot(part) 
        position,rpy = Matrix2Pos_rpy(target_matrix)
        target_matrix = Rot2Matrix(self.init_rotation, position)
  
        p1 = copy.deepcopy(target_matrix)
        p2 = copy.deepcopy(target_matrix)
        p3 = copy.deepcopy(target_matrix)
        p1[2,3]=p1[2,3]+0.2
        p3[2,3]=p3[2,3]+0.25             #  提高夹爪，避免碰撞
        
        self.MOV_M(p1,eps =0.01)  
        self.set_floor_robot_gripper_state(True)

        if part.need_flip:
            repick_nums = 0
            while not self.floor_robot_gripper_state.attached:
                if repick_nums >= 10:
                    return False
                    
                repick_nums = repick_nums + 1
                if 'pump' in part.type:
                    p1[2,3] = target_matrix[2,3]-0.040 - (repick_nums-1)*0.001
                else:
                    p1[2,3] = target_matrix[2,3]-0.041 - (repick_nums-1)*0.001
                self.MOV_M(p1,eps =0.01)
                p2[2,3] = target_matrix[2,3]-0.035
                self.MOV_M(p2,eps =0.01)
                sleep(0.1)
                
        else:
            repick_nums = 0
            while not self.floor_robot_gripper_state.attached:
                if repick_nums >= 5:
                    return False
                    
                repick_nums = repick_nums + 1
                p1[2,3] = target_matrix[2,3]+kitting_pick_part_heights_on_bin_agv[part.type]-0.041 - (repick_nums-1)*0.002
                p2[2,3] = target_matrix[2,3]+kitting_pick_part_heights_on_bin_agv[part.type]+0.1
            
                self.MOV_M(p1,eps =0.01)
                self.MOV_M(p2,eps =0.01)
                sleep(0.1)


        p2=copy.deepcopy(p1)
        p2[2,3]=p2[2,3]+0.15
        self.MOV_M(p2,eps =0.01,times=3)
        
        target_matrix_n =self.Pose2Robot(part_n) 
        position_n,rpy_n = Matrix2Pos_rpy(target_matrix_n)
        target_matrix_n = Rot2Matrix(self.init_rotation, position_n)
        
        p4 = copy.deepcopy(target_matrix_n)
        if 'pump' in part_n.type:
            p4[2,3] = p4[2,3]+0.10
        else:
            p4[2,3] = p4[2,3]    
        
        self.MOV_M(p4,eps =0.01,times=3)    


        self.set_floor_robot_gripper_state(False)
        
        self.kitting_robot_init('bin_agv_insert_joint')
        
    def pick_part_on_bin_agv(self, location, part, flip = False, repick_callback_num = 0):
            target_matrix =self.Pose2Robot(part) 
            position,rpy = Matrix2Pos_rpy(target_matrix)
            target_matrix = Rot2Matrix(self.init_rotation, position)
    
            p1 = copy.deepcopy(target_matrix)
            p2 = copy.deepcopy(target_matrix)
            p3 = copy.deepcopy(target_matrix)
            p2[2,3]=p2[2,3]+0.2
            p3[2,3]=p2[2,3]+0.3             #  提高夹爪，避免碰撞
            p3[0,3]=p2[0,3]-0.1  
            
            self.MOV_M(p2,eps =0.01)  
            self.set_floor_robot_gripper_state(True)
            if part.need_flip :
                repick_nums = 0
                while not self.floor_robot_gripper_state.attached:
                    if repick_nums >= 10:
                        return False
                        
                    repick_nums = repick_nums + 1
                    if 'pump' in part.type:
                        p1[2,3] = target_matrix[2,3]-0.040 - (repick_nums-1)*0.001
                    else:
                        p1[2,3] = target_matrix[2,3]-0.041 - (repick_nums-1)*0.001
                    self.MOV_M(p1,eps =0.01)
                    p2[2,3] = target_matrix[2,3]-0.035
                    self.MOV_M(p2,eps =0.01)
                    sleep(0.1)
                    
            else:
                repick_nums = 0
                while not self.floor_robot_gripper_state.attached:
                    if repick_nums >= 5:
                        return False
                    repick_nums = repick_nums + 1
                    if 'pump' in part.type:
                        p1[2,3] = target_matrix[2,3]+kitting_pick_part_heights_on_bin_agv[part.type]-0.04 - (repick_nums-1)*0.002
                    else:
                        p1[2,3] = target_matrix[2,3]+kitting_pick_part_heights_on_bin_agv[part.type]-0.041 - (repick_nums-1)*0.002
                
                    self.MOV_M(p1,eps =0.01)
                    sleep(0.1)

            self.MOV_M(p3,eps =0.01)     
        
#region ## ###################################### Ceiling Robot  Function   ###############################################################################   
    
    def agv_to_as(self,subtask):
            if isinstance(subtask.agv_numbers, int):
                subtask.agv_numbers = [subtask.agv_numbers]

            for agv in subtask.agv_numbers:
                if subtask.station==1 or subtask.station==3:        # 由 station 推出 agv_destination
                    agv_destination=1
                if subtask.station==2 or subtask.station==4:
                    agv_destination=2  
                
                agv_location=[self.agv1_position,self.agv2_position,self.agv3_position,self.agv4_position]
                print("agv的location是:",agv_location)
                if agv_location[agv - 1] != agv_destination:
                    self.move_agv(agv, agv_destination)
                    self.lock_agv_tray(agv)

            sleep(0.5)

            if  not self.order_pose[subtask.order_id] :

                self.order_pose[subtask.order_id] =  self.get_assembly_poses(subtask.order_id)
                print("打印一下获得的pose:",self.order_pose[subtask.order_id])
            
 
    def ceiling_robot_gripper_state_cb(self, msg: VacuumGripperState):
        self.ceiling_robot_gripper_state = msg

    def set_ceiling_robot_gripper_state(self, state) -> bool: 
        
        with self.spin_lock:
            rclpy.spin_once(self)
        if self.ceiling_robot_gripper_state.enabled == state:
            return
        
        request = VacuumGripperControl.Request()
        request.enable = state
        
        future = self.ceiling_gripper_enable.call_async(request)
        
        try:
            with self.spin_lock:
                rclpy.spin_until_future_complete(self, future)

        except KeyboardInterrupt:
            raise KeyboardInterrupt
        
        if future.result().success:
            # self.get_logger().info(f'Changed gripper state to {self.gripper_states_[state]}')
            pass
        else:
            self.get_logger().warn('Unable to change gripper state')
        
        
    
    
    def gantry_state_callback(self, msg):
        self.ceiling_base_x = msg.reference.positions[0]
        self.ceiling_base_y = msg.reference.positions[1]
        self.ceiling_base_r = msg.reference.positions[2]

        self.ceiling_torso_joints = [self.ceiling_base_x,self.ceiling_base_r,self.ceiling_base_y]
        
    def ceiling_arm_joint_state_callback(self,msg):
        self.ceiling_arm_joint_states = msg.reference.positions


    
    def gantry_robot_init(self):
        while not self.ceiling_arm_joint_states :
            sleep(0.5)
            #print 'waiting for initialization ...'
        # 复位
        print ("gantry_robot_initializing...")
        # print("gantry订阅到了",self.ceiling_arm_joint_states )
        self.MOV_A_CEILING(self.ceiling_arm_init_position, eps=0.01,sleep=False)

        
        eps = 0.02
        while rclpy.ok():
            q_begin = self.ceiling_torso_joints
            # q_begin = copy.deepcopy(self.ceiling_torso_joints)
            # print("q_began",q_begin)
            q_end = self.ceiling_init_position
            # print("q_end",q_end)
            delta = [abs(q_begin[i] - q_end[i]) for i in range(0,len(q_begin))]
            distance = max(delta)
            # print("distance",distance)
            if distance < eps:
                break
            move_time = distance/gantry_velocity
            # print("move_time",move_time)
            # print("-------")
            self.send_gantry_to_state(q_end, move_time)
            sleep(move_time)

        print ("gantry_robot_initialize success!")
        return True
    
    def send_gantry_to_state(self,positions,time_from_start):
        msg = JointTrajectory()
        msg.joint_names = self.ceiling_torso_joint_names
        point = JointTrajectoryPoint()
        tran_position = [positions[0], positions[1],-positions[2]]
        point.positions = tran_position
        point.time_from_start = Duration(seconds=time_from_start).to_msg()
        msg.points = [point] 
        
        self.move(self.gantry_action_client,msg)
        
    @ceiling_fault_detector
    def MOV_A_CEILING(self, target_joint_angles,time_from_start = 0.0,eps = 0.01,sleep = True):
        '''
        6-DOF，角度顺序是 [elbow,...]
        time_from_start,默认按照最快时间执行
        eps = 0.0025
        '''
        #当前关节角度
        q_begin = self.ceiling_arm_joint_states
        # q_begin = copy.deepcopy(self.ceiling_arm_joint_states)
        q_end = target_joint_angles
        delta = [abs(q_begin[i] - q_end[i]) for i in range(0,len(q_begin))]
        distance = max(delta)
        # 计算理想的运行时间
        runtime = distance/gantry_angle_velocity
        exe_time = 0.00
        start_time = self.get_clock().now().nanoseconds/1e9
        sleep_flag = True


        while  sleep_flag and exe_time < runtime*2:
            exe_time = self.get_clock().now().nanoseconds/1e9 - start_time
            #当前关节角度
            q_begin = self.ceiling_arm_joint_states
            # q_begin = copy.deepcopy(self.ceiling_arm_joint_states)
            q_end = target_joint_angles
            delta = [abs(q_begin[i] - q_end[i]) for i in range(0,len(q_begin))]
            distance = max(delta)
            # #print delta
            if distance < eps:
                break
            # 计算理想的运行时间
            if time_from_start == 0.0:
                time_from_start = distance/gantry_angle_velocity
            # #print time_from_start
            #做运动插补
            traj = traj_generate(self.ceiling_arm_joint_names,q_begin,q_end,time_from_start)
            ##print traj
            self.move(self.ceiling_action_client,traj)

            
            if sleep:
                time.sleep(time_from_start)
            else:
                sleep_flag = False   
                
    @ceiling_fault_detector
    def move_to_ceiling(self, location, right=0.0,left=0.0,forward=0.0,back=0.0,flip=False):
        '''
        location 表示驻点
        '''
        # print("开始移动") 
        # self.robot_info.next_park_location = location
        # if not flip:
        #     self.robot_info.work_state= "moving"
        # else:
        #     self.robot_info.work_state= "flipping"

        if "conveyor" in location:
            target_torso_point  = [0.3+5, -pi/2, self.ceiling_base_y]
        else:
            end_point = gantry_robot_park_location[location]
            target_torso_point = [end_point[0]+forward-back,end_point[2],end_point[1]+right-left]


        distance = max([abs(target_torso_point[0]-self.ceiling_base_x), abs(target_torso_point[1]- self.ceiling_base_r)\
            ,abs(-target_torso_point[2]-self.ceiling_base_y)])

        time_from_start = distance/gantry_velocity
        print("开始移动")
        self.send_gantry_to_state(target_torso_point,time_from_start)
        sleep(time_from_start)
        print("target_torso_point",target_torso_point)
        print("self.ceiling_torso_joints",self.ceiling_torso_joints)
        print("--------")
        print("已到达")   

    def Left(self, distance,eps = 0.001):
        q_begin = self.ceiling_torso_joints
        q_end = copy.deepcopy(q_begin)
        q_end[2] = q_end[2]-distance

        delta = [abs(q_begin[i] - q_end[i]) for i in range(0,len(q_begin))]
        dis = max(delta)
        move_time = dis/gantry_velocity
        self.send_gantry_to_state(q_end, move_time)
        sleep(move_time)

    def Right(self, distance,eps = 0.001):
        q_begin = self.ceiling_torso_joints
        q_end = copy.deepcopy(q_begin)
        q_end[2] = q_end[2]+distance

        delta = [abs(q_begin[i] - q_end[i]) for i in range(0,len(q_begin))]
        dis = max(delta)
        move_time = dis/gantry_velocity
        self.send_gantry_to_state(q_end, move_time)
        sleep(move_time)
        return True

    def Back(self, distance,eps = 0.001):
        q_begin = self.ceiling_torso_joints
        q_end = copy.deepcopy(q_begin)
        q_end[0] = q_end[0]-distance
        while rclpy.ok():
            q_begin = self.ceiling_torso_joints
            delta = [abs(q_begin[i] - q_end[i]) for i in range(0,len(q_begin))]
            dis = max(delta)
            if dis < eps:
                break
            move_time = dis/gantry_velocity
            self.send_gantry_to_state(q_end, move_time)
            sleep(move_time)

        return True

    def Forward(self, distance,eps = 0.001):
        q_begin = self.ceiling_torso_joints
        q_end = copy.deepcopy(q_begin)
        q_end[0] = q_end[0]+distance
        while rclpy.ok():
            q_begin = self.ceiling_torso_joints
            delta = [abs(q_begin[i] - q_end[i]) for i in range(0,len(q_begin))]
            dis = max(delta)
            if dis < eps:
                break
            move_time = dis/gantry_velocity
            self.send_gantry_to_state(q_end, move_time)
            sleep(move_time)
        return True


    def pick_part_on_bin_ceiling(self,location,part,distance = 0.5,repick_callback_num = 0):

        target_matrix =self.Tf_trans(part.pose) 

        position,rpy = Matrix2Pos_rpy(target_matrix)
        self.pick_part_theta = rpy[-1]
       
        insert_joint = [-3.18,-1.82,-1.83,-1.14,1.61,-3.14]#交换
        self.MOV_A_CEILING(insert_joint,eps=0.02)
        print("move to ==.")

        if 'battery' in part.type:

            need_yaw=1.574971916153666 -abs(GetYaw(part.pose.orientation))
            position[2]=position[2]-0.03*sin(need_yaw)
            position[1]=position[1]-0.03*cos(need_yaw)
            position[0]=position[0]-0.025
            print("在求证呢need_yaw:",need_yaw,"position:",position)
            
        target_matrix = Rot2Matrix(self.rotate_rotation, position)  

        p1 = copy.deepcopy(target_matrix)
        p2 = copy.deepcopy(target_matrix)

        p2[0,3] = target_matrix[0,3]+gantry_pick_part_heights_bin_agv[part.type]+vacuum_gripper_height+0.1
        #先到达P3
        print("move to p2 ...")
        if not self.MOV_M_ceiling(p2,eps =0.0025):
            return False
        print("move to p22 ...")
        #print 'open gripper'
        self.set_ceiling_robot_gripper_state(True)
        
        repick_nums = 0
        while not self.ceiling_robot_gripper_state.attached:
            if repick_nums >= 3:
                self.robot_arm_init(location, part)
                return False
                
            repick_nums = repick_nums + 1
            p1[0,3] = target_matrix[0,3]+gantry_pick_part_heights_bin_agv[part.type]+vacuum_gripper_height +0.07- (repick_nums-1)*0.003
            p2[0,3] = target_matrix[0,3]+gantry_pick_part_heights_bin_agv[part.type]+vacuum_gripper_height+0.15
        
            self.MOV_M_ceiling(p1,eps =0.01,time_factor=3)
            print("self.gripper_enabled",self.ceiling_robot_gripper_state.enabled)

            self.MOV_M_ceiling(p2,eps =0.01)
            
        p1[0,3]=p1[0,3]+0.4
        self.MOV_M_ceiling(p1,eps =0.01)
        self.robot_arm_init(location,part)
 
    def pick_part_has_flip_ceiling(self,location,part,agv_number,distance = 0.5,repick_callback_num = 0):

        target_bin=find_closest_bin(agv_number)
        target_part_list_bins = self.search_part_on_bins(part.type)
        
        if not target_part_list_bins:
            return False
            
        target_part=find_nearest_part(target_part_list_bins,bin_position[target_bin][0],bin_position[target_bin][1]+0.3)
        
        

        
        target_matrix =self.Tf_trans(target_part.pose) 
        position,rpy = Matrix2Pos_rpy(target_matrix)
        self.pick_part_theta = rpy[-1]


        insert_joint = [-3.18,-1.82,-1.83,-1.14,1.61,-3.14]#交换
        self.MOV_A_CEILING(insert_joint,eps=0.02)
        
        if 'battery' in part.type:
            position[2]=position[2]-0.03
        #     position[0]=position[0]-0.068
        # if 'pump' in part.type:
        #     position[1]=position[1]-0.03
        #     position[0]=position[0]-0.12
        # if 'sensor' in part.type:
        #     position[0]=position[0]-0.07
        # if 'regulator' in part.type:
        #     position[1]=position[1]+0.03
        #     position[0]=position[0]-0.07

            
        target_matrix = Rot2Matrix(self.rotate_rotation, position)  

        p1 = copy.deepcopy(target_matrix)
        p2 = copy.deepcopy(target_matrix)

        p2[0,3] = target_matrix[0,3]+gantry_pick_part_heights_bin_agv[part.type]+vacuum_gripper_height+0.1
        #先到达P3
        print("move to p2 ...")
        if not self.MOV_M_ceiling(p2,eps =0.0025):
            return False
        print("move to p22 ...")
        #print 'open gripper'
        self.set_ceiling_robot_gripper_state(True)
        


        repick_nums = 0
        while not self.ceiling_robot_gripper_state.attached:
            if repick_nums >= 3:
                self.robot_arm_init(location, part)
                return False
                
            repick_nums = repick_nums + 1
            p1[0,3] = target_matrix[0,3]+gantry_pick_part_heights_bin_agv[part.type]+vacuum_gripper_height +0.07- (repick_nums-1)*0.003
            p2[0,3] = target_matrix[0,3]+gantry_pick_part_heights_bin_agv[part.type]+vacuum_gripper_height+0.15
        
            self.MOV_M_ceiling(p1,eps =0.01)
            print("self.gripper_enabled",self.ceiling_robot_gripper_state.enabled)

            self.MOV_M_ceiling(p2,eps =0.01)
            
        p1[0,3]=p1[0,3]+0.2
        self.MOV_M_ceiling(p1,eps =0.01)
        self.robot_arm_init(location,part)
    
        return target_part


    def pick_part_flip__on_bin_ceiling(self,location,part,distance = 0.5,repick_callback_num = 0):

        target_matrix =self.Tf_trans(part.pose) 
        position,rpy = Matrix2Pos_rpy(target_matrix)
        self.pick_part_theta = rpy[-1]

        if 'battery' in part.type:
            
            need_yaw=1.574971916153666 -abs(GetYaw(part.pose.orientation))
            position[2]=position[2]-0.03*sin(need_yaw)
            position[1]=position[1]-0.03*cos(need_yaw)
            position[0]=position[0]+0.005

        insert_joint = [-3.18,-1.82,-1.83,-1.14,1.61,-3.14]#交换
        self.MOV_A_CEILING(insert_joint,eps=0.02)
        target_matrix = Rot2Matrix(self.rotate_rotation, position)  
        

        p1 = copy.deepcopy(target_matrix)
        p2 = copy.deepcopy(target_matrix)

        p2[0,3] = target_matrix[0,3]+gantry_pick_part_heights_bin_agv[part.type]+vacuum_gripper_height+0.1
        #先到达P3
        print("move to p2 ...")
        if not self.MOV_M_ceiling(p2,eps =0.0025):
            return False
        print("move to p22 ...")
        #print 'open gripper'
        self.set_ceiling_robot_gripper_state(True)
        repick_nums = 0
        
        print("来执行翻转任务了")
        if "pump" in part.type:
            while not self.ceiling_robot_gripper_state.attached:
                if repick_nums >= 3:
                    self.robot_arm_init(location, part)
                    return False
                    
                repick_nums = repick_nums + 1
                p1[0,3] = target_matrix[0,3]+gantry_pick_part_heights_bin_agv[part.type]+vacuum_gripper_height -0.049- (repick_nums-1)*0.001
                p2[0,3] = target_matrix[0,3]+gantry_pick_part_heights_bin_agv[part.type]+vacuum_gripper_height+0.08
            
                self.MOV_M_ceiling(p1,eps =0.01,time_factor=5)
                print("self.gripper_enabled",self.ceiling_robot_gripper_state.enabled)

                self.MOV_M_ceiling(p2,eps =0.01,time_factor=5)
        
        else:
            while not self.ceiling_robot_gripper_state.attached:
                if repick_nums >= 3:
                    self.robot_arm_init(location, part)
                    return False
                    
                repick_nums = repick_nums + 1
                p1[0,3] = target_matrix[0,3]+gantry_pick_part_heights_bin_agv[part.type]+vacuum_gripper_height -0.00- (repick_nums-1)*0.001
                p2[0,3] = target_matrix[0,3]+gantry_pick_part_heights_bin_agv[part.type]+vacuum_gripper_height+0.08
            
                self.MOV_M_ceiling(p1,eps =0.01,time_factor=5)
                print("self.gripper_enabled",self.ceiling_robot_gripper_state.enabled)

                self.MOV_M_ceiling(p2,eps =0.01,time_factor=5)
            
        p1[0,3]=p1[0,3]+0.3
        self.MOV_M_ceiling(p1,eps =0.01,time_factor=5)
        self.robot_arm_init(location,part)


    def flip_part_on_ceiling(self,agv_number,part):    
        
        target_bin=find_closest_bin_ceiling(agv_number)
        self.move_to_ceiling(target_bin)

        part_pose = Pose()             
        result = bin3_8_flip[target_bin]

        part_pose.position.x = result[0]
        part_pose.position.y = result[1]
        part_pose.position.z = result[2]
        part_pose.orientation.x = 0.0
        part_pose.orientation.y = 0.0
        part_pose.orientation.z = 0.0
        part_pose.orientation.w = 1.0 
        
        target_matrix = self.Tf_trans(part_pose)
        print("target",target_matrix)
        position,rpy = Matrix2Pos_rpy(target_matrix)

        target_matrix = Rot2Matrix(self.rotate_rotation, position)

        p1 = copy.deepcopy(target_matrix)
        p2 = copy.deepcopy(target_matrix)
        p2[0,3]=p2[0,3]+0.2
        self.MOV_M_ceiling(p2)
        p2[0,3]=p2[0,3]-0.06
        self.MOV_M_ceiling(p2,time_factor=3)

        self.set_ceiling_robot_gripper_state(False)
        

        p3 = Rot2Matrix(self.flip_rotation['right_roll_0'], position)
        p5=copy.deepcopy(p3)
        p5[0,3]=p5[0,3]+0.2
        p5[1,3]=p5[1,3]+0.1
        self.MOV_M_ceiling(p5,eps =0.01)  
        self.set_ceiling_robot_gripper_state(True)
        
        if 'pump' in part.type:

            p5[0,3]=p5[0,3]-0.12
            p5[1,3]=p5[1,3]-0.03
            self.MOV_M_ceiling(p5,eps =0.01,time_factor=5)  

            pick_num=0
            while not self.ceiling_robot_gripper_state.attached:
                if pick_num>10:
                    break
                pick_num+=1
                p5[1,3]=p5[1,3]-0.001
                self.MOV_M_ceiling(p5,eps =0.001,time_factor=20) 
                print(f"第{pick_num}次抓取")
                sleep(0.5) 
            
            p5[0,3]=p5[0,3]+0.2
            self.MOV_M_ceiling(p5,eps =0.01)   
            self.ceiling_arm_init()  
        





    def ceiling_arm_init(self):
        self.MOV_A_CEILING(self.ceiling_arm_init_position, eps=0.02)
        # self.robot_info.work_state= "standby"
        
    def robot_arm_init(self, target_agv_tray, target_part):
        # self.gripper.deactivate_gripper()

        if 'ks' in target_agv_tray and target_part.pose.position.x < -2.2828:
            p3 = self.ceiling_arm_kitting_position
            self.MOV_A_CEILING(p3,eps =0.02) 
            if "agv4" in target_agv_tray :
                self.MOV_A_CEILING(self.ceiling_arm_init_position,eps =0.05) 
            else:
                pass
        else:
            p3 = self.ceiling_arm_init_position
            self.MOV_A_CEILING(p3,eps =0.02) 
    
    
        
    def send_gantry_arm_to_state(self,positions,time_from_start):
        msg = JointTrajectory()
        msg.joint_names = self.ceiling_arm_joint_names
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start = Duration(seconds=time_from_start).to_msg()
        msg.points = [point]
        #rospy.loginfo("Sending command:\n" + str(msg))
        self.move(self.ceiling_action_client,msg)
    


    def STOP_ceiling(self):
        self.MOV_A_CEILING(self.ceiling_arm_joint_states)
        # self.send_gantry_to_state(self.ceiling_torso_joints,0.001)

    @ceiling_fault_detector
    def MOV_M_ceiling(self, target_matrix,time_from_start = 0.0,eps = 0.005,sleep = True,flip_flag=False,time_factor=1.0): 
        '''
        4*4 Matrix 
        主要用来做抓取,eps = 0.0025是机器人的最小误差，一般设置要大于此值
        '''
        q_begin = copy.deepcopy(self.ceiling_arm_joint_states)
        print("target_matrix",target_matrix)
        target = IKinematic(target_matrix,q_begin,A_k)
        if target==None:
            target = IKinematic(target_matrix,q_begin,A_k)
            if target==None:
                print("解不出来")
                return False
        # #print "IKinematic_result:",target
        if flip_flag:
            target[-1] = q_begin[-1]
        #求时间
        delta = [abs(q_begin[i] - target[i]) for i in range(0,len(q_begin))]
        ##print delta
        distance = max(delta)
        time_from_start = distance/gantry_angle_velocity*time_factor
        traj = traj_generate(self.ceiling_arm_joint_names,q_begin,target,time_from_start)
        self.move(self.ceiling_action_client,traj)

        time.sleep(time_from_start)
        return True
        


    def Tf_trans(self, pose):
        '''
        输入是位置+四元数，输出是机器人直接可以使用的齐次旋转矩阵
        返回零件相对与机器人的位姿-齐次矩阵
        ''' 
        sleep(0.5)   #机器人停稳了
        self.transforms = []
        tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)

        tf_msg = TransformStamped()
        tf_msg.header.stamp =self.get_clock().now().to_msg()
        tf_msg.header.frame_id = 'world'
        tf_msg.child_frame_id = 'gantry_target_frame'
        translation=Vector3()
        translation.x=pose.position.x
        translation.y=pose.position.y
        translation.z=pose.position.z
        tf_msg.transform.translation = translation
        tf_msg.transform.rotation = pose.orientation
        self.transforms.append(tf_msg)
        for _ in range(5):
            tf_broadcaster.sendTransform(self.transforms)

        print("tf_msg",tf_msg)  
        sleep(0.5)
        
        sleep(1.5)

        t =TransformStamped()

        pose = Pose()

        MAX_ATTEMPTS = 50
        attempts = 0
        while attempts < MAX_ATTEMPTS:
            attempts +=1

            try:
                # trans = self.tf_buffer.lookup_transform('world', 'ceiling_base', rclpy.time.Time())
                # print('ceiling的基坐标是',trans)
                t= self.tf_buffer.lookup_transform("ceiling_base", "gantry_target_frame",  rclpy.time.Time())
            except TransformException as ex:
                self.get_logger().info(f'Could not transform assembly station  to world: {ex}')
          

        print("-----t----",t) 
        pose.position.x = t.transform.translation.x
        pose.position.y = t.transform.translation.y
        pose.position.z = t.transform.translation.z
        pose.orientation = pose.orientation
        print("-----pose----",pose)
        target = Pose2Matrix(pose)
        return target
    
    def Tf_trans_pose(self, pose):
        '''
        输入是位置+四元数，输出是机器人直接可以使用的齐次旋转矩阵
        返回零件相对与机器人的位姿-齐次矩阵
        ''' 
        sleep(0.6)   #机器人停稳了
        self.transforms = []
        tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)

        tf_msg = TransformStamped()
        tf_msg.header.stamp =self.get_clock().now().to_msg()
        tf_msg.header.frame_id = 'world'
        tf_msg.child_frame_id = 'assembly_part_frame'
        translation=Vector3()
        translation.x=pose.position.x
        translation.y=pose.position.y
        translation.z=pose.position.z
        tf_msg.transform.translation = translation
        tf_msg.transform.rotation = pose.orientation
        self.transforms.append(tf_msg)
        for _ in range(5):
            tf_broadcaster.sendTransform(self.transforms)

        print("tf_msg",tf_msg)  
        sleep(0.5)
        
        
        t =TransformStamped()
        t1 =TransformStamped()
        pose = Pose()

        MAX_ATTEMPTS = 50
        attempts = 0
        while attempts < MAX_ATTEMPTS:
            attempts +=1

            try:
                t= self.tf_buffer.lookup_transform("ceiling_base", "assembly_part_frame",  rclpy.time.Time())
            except TransformException as ex:
                self.get_logger().info(f'Could not transform assembly station  to world: {ex}')           

        # print("-----t1----",t1) 
        pose.position.x = t.transform.translation.x
        pose.position.y = t.transform.translation.y
        pose.position.z = t.transform.translation.z
        pose.orientation = pose.orientation

        target = Pose2Matrix(pose)
        return target

    def FrameWorldPose_CEILING(self,frame_id):
        t =TransformStamped()
        pose = Pose()
        
        MAX_ATTEMPTS = 20
        attempts = 0
        while attempts < MAX_ATTEMPTS:
            attempts +=1

            try:
                t= self.tf_buffer.lookup_transform("ceiling_base", frame_id,  rclpy.time.Time())
            except TransformException as ex:
                self.get_logger().info(f'Could not transform assembly station  to world: {ex}')
                return

        print("-----与AGV的变换----",t)
        pose.position.x = t.transform.translation.x
        pose.position.y = t.transform.translation.y
        pose.position.z = t.transform.translation.z
        pose.orientation = t.transform.rotation

        target = Pose2Matrix(pose)

        try:
            t1= self.tf_buffer.lookup_transform("world", frame_id,  rclpy.time.Time())
        except TransformException as ex:
            self.get_logger().info(f'Could not transform assembly station  to world: {ex}')
            return
        print("-----我想看看{frame_id}在world中的坐标----",t1)
        return target
        
    
    

    def goal_response_callback(self, future):
        goal_handle = future.result()
        goal_trajectory = self.goals.pop(future)
        if not goal_handle.accepted:
            self.get_logger().info(f'Goal rejected')
            return
        self.get_logger().info(f'Goal  accepted')


    def ceiling_robot_place_part_on_kit_tray(self,subtask,target_part,dest) -> bool:
        
        
        part_pose = Pose()             
        result = agv_quadrant_position(subtask.agv_number, subtask.product_quadrant)

        part_pose.position.x = result[0]
        part_pose.position.y = result[1]
        part_pose.position.z = result[2]
        part_pose.orientation.x = 0.0
        part_pose.orientation.y = 0.0
        part_pose.orientation.z = 0.0
        part_pose.orientation.w = 1.0 
        
        # print("我要去的是象限",subtask.product_quadrant)

        target_matrix = self.Tf_trans(part_pose)
        print("target",target_matrix)
        position,rpy = Matrix2Pos_rpy(target_matrix)

        target_matrix = Rot2Matrix(self.rotate_rotation, position)

        p2 = copy.deepcopy(target_matrix)
        p2[0,3]=p2[0,3]+0.2
        
        self.MOV_M_ceiling(p2)

        self.set_ceiling_robot_gripper_state(False)

        self.ceiling_arm_init()
        
        check_result,check_info=self.perform_quality_check(subtask.order_id)   
        sleep(0.5)
        check_result,check_info=self.perform_quality_check(subtask.order_id)     #### 一个BUG，第二次才是真实的
         
        if self.check_faluty_part(check_info,subtask.product_quadrant): 
            
            self.set_ceiling_robot_gripper_state(True)
        
            print("开始抓取坏零件")
            p2 = copy.deepcopy(target_matrix)
            self.move_to_ceiling(dest,forward=0.2)
            p2[2,3]=p2[2,3]-0.2
            repick_nums = 0

            while not self.ceiling_robot_gripper_state.attached:
                if repick_nums >= 10:
                    self.ceiling_arm_init()
                    return False
                print(f"第{repick_nums}次抓取坏零件")
                repick_nums = repick_nums + 1
                p2[0,3] =target_matrix[0,3]+ceiling_faulty_part[target_part.type]-0.003*repick_nums
                # p2[0,3]=p2[0,3]- 0.01*repick_nums
                self.MOV_M_ceiling(p2)
                sleep(0.2)
                
            self.ceiling_arm_init()  
            self.move_to_ceiling("can")    
            self.set_ceiling_robot_gripper_state(False)
            self.kitting_deque.appendleft(subtask)
            self.oredr_length[subtask.order_id]=self.oredr_length[subtask.order_id]+1  
        
        return True 

    def ceiling_pick_assembly_part(self,subtask) -> bool:
        self.ceiling_arm_init()
        target_matrix = self.Tf_trans_pose(subtask.pose)
        print("target",target_matrix)
        position,rpy = Matrix2Pos_rpy(target_matrix)
        position[0]=position[0]
        if 'battery' in subtask.type:
            need_yaw=GetYaw(subtask.pose.orientation)-1.574971916153666
            position[2]=position[2]-0.03*sin(need_yaw)
            position[1]=position[1]-0.03*cos(need_yaw)
            position[0]=position[0]-0.027
        if 'pump' in subtask.type:
            position[0]=position[0]+0.003

        target_matrix = Rot2Matrix(self.rotate_rotation, position)

        p1 = copy.deepcopy(target_matrix)
        p2 = copy.deepcopy(target_matrix)
        p2[0,3]=p2[0,3]+0.2

        
        self.MOV_M_ceiling(p2)

        self.set_ceiling_robot_gripper_state(True)
        
        p1[0,3] = target_matrix[0,3]+gantry_pick_part_heights_bin_agv[subtask.type]+vacuum_gripper_height +0.07
        
        self.MOV_M_ceiling(p1,eps =0.005,time_factor=5)
        sleep(0.5)


        repick_nums = 0
        while not self.ceiling_robot_gripper_state.attached:
            if repick_nums >= 5:
                print("抓取失败")
                return False
            
            print(f"这是抓取第{repick_nums}次")
            repick_nums = repick_nums + 1
            p1[0,3] = p1[0,3]- 0.002
            self.MOV_M_ceiling(p1,eps =0.01,time_factor=5)
            sleep(0.5)

        # 
        print("self.attached",self.ceiling_robot_gripper_state.enabled)
        if self.ceiling_robot_gripper_state.attached:
            self.ceiling_robot_info.work_state = "has_grasped"

        
        p1[0,3]=p1[0,3]+0.5
        self.MOV_M_ceiling(p1,eps =0.01)
        
        return True 
    
    def TF_trans_two(self,as_station,pose):
        sleep(0.5)   #机器人停稳了
        sleep(0.5)

        self.transforms = []
        tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)

        tf_msg = TransformStamped()
        tf_msg.header.stamp =self.get_clock().now().to_msg()
        tf_msg.header.frame_id = as_station
        tf_msg.child_frame_id = 'put_assembly_part'
        translation=Vector3()
        translation.x=pose.pose.position.x
        translation.y=pose.pose.position.y
        translation.z=pose.pose.position.z
        tf_msg.transform.translation = translation
        tf_msg.transform.rotation = pose.pose.orientation
        self.transforms.append(tf_msg)
        for _ in range(5):
            tf_broadcaster.sendTransform(self.transforms)

        print("tf_msg",tf_msg)  
        
        sleep(0.5)
        
        
        t =TransformStamped()
        pose = Pose()

        MAX_ATTEMPTS = 50
        attempts = 0
        while attempts < MAX_ATTEMPTS:
            attempts +=1

            try:
                t= self.tf_buffer.lookup_transform("ceiling_base", "put_assembly_part",  rclpy.time.Time())
            except TransformException as ex:
                self.get_logger().info(f'Could not transform assembly station  to world: {ex}')           

        print("-----t----",t) 
        pose.position.x = t.transform.translation.x
        pose.position.y = t.transform.translation.y
        pose.position.z = t.transform.translation.z
        pose.orientation = pose.orientation
        print("-----pose----",pose)
        target = Pose2Matrix(pose)
        return target                     

    def ceiling_place_assembly_pump(self,subtask,cmd_type,part) -> bool:
        self.move_to_ceiling("as"+str(subtask.station))  
        self.ceiling_arm_init()
        sleep(0.5)
        target_matrix=self.TF_trans_two("as"+str(subtask.station)+"_insert_frame",subtask.assembled_pose)
        print("target--861--!!",target_matrix)
        position,rpy = Matrix2Pos_rpy(target_matrix)
        position[0]=position[0]
        position[1]=position[1]
        position[2]=position[2]
        # position[1]=-(position[1]+gantry_robot_park_location[station][1])
        # position[2]=position[2]-0.035
        target_matrix = Rot2Matrix(self.rotate_rotation, position)
        if cmd_type=="assembly":
            base_target = Pose()
            base_target.position.x=0.0
            base_target.position.y=0.0
            base_target.position.z=0.0
            need_yaw=GetYaw(subtask.grap_pose.orientation)-1.574971916153666
            yaw=GetYaw(subtask.assembled_pose.pose.orientation)
            if yaw>0:
                base_target.orientation=QuaternionFromRPY(need_yaw,0,0)     #让夹爪旋转
            else:
                base_target.orientation=QuaternionFromRPY(need_yaw+3.1415826,0,0)     #让夹爪旋转
            part_to_gripper=Pose2Matrix(base_target)
            target_matrix=target_matrix@part_to_gripper

        else:   
            base_target = Pose()
            base_target.position.x=0.0
            base_target.position.y=0.0
            base_target.position.z=0.0
            part_yaw=1.574971916153666 -abs(GetYaw(part.pose.orientation))
            if part_yaw>0:
                base_target.orientation=QuaternionFromRPY(part_yaw,0,0)     #让夹爪旋转
            else:
                base_target.orientation=QuaternionFromRPY(part_yaw+3.1415826,0,0)     #让夹爪旋转

            part_to_gripper=Pose2Matrix(base_target)
            target_matrix=target_matrix@part_to_gripper

        p1 = copy.deepcopy(target_matrix)
        p1[0,3]=p1[0,3]+0.3
        p2 = copy.deepcopy(target_matrix)
        p2[0,3]=p2[0,3]+0.15
        p2[2,3]=p2[2,3]-0.001
 
        
        self.MOV_M_ceiling(p1)
        
        if not self.ceiling_robot_gripper_state.attached:
            print("Faulty Gripper ")
            self.oredr_length[subtask.order_id]=self.oredr_length[subtask.order_id]+1
            self.assembly_deque.appendleft(subtask)
            return False
            
        sleep(0.2)
        self.MOV_M_ceiling(p2,time_factor=5)
        
        # self.MOV_M_ceiling(p1,time_factor=10)
        
        
    def ceiling_place_assembly_sensor(self,subtask,cmd_type,part) -> bool:
        
        self.move_to_ceiling("as"+str(subtask.station),right=0.1)  
        self.ceiling_arm_init()
        target_matrix=self.TF_trans_two("as"+str(subtask.station)+"_insert_frame",subtask.assembled_pose)
        print("target--861--!!",target_matrix)
        position,rpy = Matrix2Pos_rpy(target_matrix)
        position[0]=position[0]
        position[1]=position[1]
        position[2]=position[2]
        # position[1]=-(position[1]+gantry_robot_park_location[station][1])
        # position[2]=position[2]-0.035
        target_matrix1 = Rot2Matrix(self.rotate_Y_90, position)
        
        if cmd_type=="assembly":
            base_target = Pose()
            base_target.position.x=0.0
            base_target.position.y=0.0
            base_target.position.z=0.0
            need_yaw=GetYaw(subtask.grap_pose.orientation)-1.574971916153666        
            base_target.orientation=QuaternionFromRPY(need_yaw,1.574971916153666,0)     #让夹爪旋转
            part_to_gripper=Pose2Matrix(base_target)
        
        else:
            base_target = Pose()
            base_target.position.x=0.0
            base_target.position.y=0.0
            base_target.position.z=0.0
            part_yaw=1.574971916153666 -abs(GetYaw(part.pose.orientation))     
            base_target.orientation=QuaternionFromRPY(part_yaw,1.574971916153666,0)     #让夹爪旋转
            part_to_gripper=Pose2Matrix(base_target)
        

        target_matrix1[0,3]=target_matrix1[0,3]+0.1
        
        p1 = copy.deepcopy(target_matrix1)
        p1[0,3]=p1[0,3]+0.3

        p1=p1@part_to_gripper  
        self.MOV_M_ceiling(p1,time_factor=3)
        sleep(0.2)

        if not self.ceiling_robot_gripper_state.attached:
            print("Faulty Gripper ")
            self.oredr_length[subtask.order_id]=self.oredr_length[subtask.order_id]+1
            self.assembly_deque.appendleft(subtask)
            return False
 
        p3=copy.deepcopy(p1)
        p3[1,3]=p3[1,3]+0.050
        # p3[2,3]=p3[2,3]+0.045
        p3[0,3]=p3[0,3]-0.319
        self.MOV_M_ceiling(p3,time_factor=5)
        
        sleep(0.2)

        p3[1,3]=p3[1,3]-0.050
        self.MOV_M_ceiling(p3,time_factor=10)

    def ceiling_place_assembly_regulator(self,subtask,cmd_type,part) -> bool:
        #---------------------------------------------------------------------------------------------------------
        self.move_to_ceiling("as"+str(subtask.station))  
     
        self.ceiling_arm_init()
        
        target_matrix=self.TF_trans_two("as"+str(subtask.station)+"_insert_frame",subtask.assembled_pose)
        position,rpy = Matrix2Pos_rpy(target_matrix)
        position[0]=position[0]
        position[1]=position[1]
        position[2]=position[2]
        # position[1]=-(position[1]+gantry_robot_park_location[station][1])
        # position[2]=position[2]-0.035
        target_matrix = Rot2Matrix(self.rotate_Y_90, position)  
       
        if cmd_type=="assembly":  
            
            base_target = Pose()
            base_target.position.x=0.0
            base_target.position.y=0.0     
            base_target.position.z=0.0
            need_yaw=GetYaw(subtask.grap_pose.orientation)-1.574971916153666
            

            base_target.orientation=QuaternionFromRPY(need_yaw,0,0)     #让夹爪旋转
            part_to_gripper=Pose2Matrix(base_target)
            target_matrix=target_matrix@part_to_gripper

        else:   
            base_target = Pose()
            base_target.position.x=0.0
            base_target.position.y=0.0
            base_target.position.z=0.0
            
            if part.need_flip:
                part_yaw=1.574971916153666 -abs(GetYaw(part.pose.orientation))
            else:
                part_yaw=1.574971916153666 -abs(GetYaw(part.pose.orientation))
                print("part_yaw:::::",part_yaw)
            base_target.orientation=QuaternionFromRPY(part_yaw,0,0) 

            part_to_gripper=Pose2Matrix(base_target)
            target_matrix=target_matrix@part_to_gripper
            
        target_matrix[0,3]=target_matrix[0,3]+0.1
        self.MOV_M_ceiling(target_matrix,time_factor=2)
        
        if not self.ceiling_robot_gripper_state.attached:
            print("Faulty Gripper ")
            self.oredr_length[subtask.order_id]=self.oredr_length[subtask.order_id]+1
            self.assembly_deque.appendleft(subtask)
            return False
        
        p1=copy.deepcopy(target_matrix)
        p1[0,3]=p1[0,3]-0.07
        p1[2,3]=p1[2,3]+0.075
        sleep(0.2)
        self.MOV_M_ceiling(p1,time_factor=3)

        sleep(0.2)
        p1[0,3]=p1[0,3]-0.025
        self.MOV_M_ceiling(p1,time_factor=5)


   
    def ceiling_place_assembly_battery(self,subtask,cmd_type,part) -> bool:
        
        self.ceiling_arm_init()
        target_matrix=self.TF_trans_two("as"+str(subtask.station)+"_insert_frame",subtask.assembled_pose)
        print("target--861--!!",target_matrix)
        position,rpy = Matrix2Pos_rpy(target_matrix)
        position[0]=position[0]
        position[1]=position[1]
        position[2]=position[2]
        # position[1]=-(position[1]+gantry_robot_park_location[station][1])
        # position[2]=position[2]-0.035
        target_matrix = Rot2Matrix(self.rotate_rotation, position)
        
        if cmd_type=="assembly":
            base_target = Pose()
            base_target.position.x=0.0
            base_target.position.y=0.0
            base_target.position.z=0.0
            need_yaw=GetYaw(subtask.grap_pose.orientation)-1.574971916153666
            base_target.orientation=QuaternionFromRPY(need_yaw,0,0)     #让夹爪旋转
            part_to_gripper=Pose2Matrix(base_target)
            target_matrix=target_matrix@part_to_gripper    
        
        else:   
            base_target = Pose()
            base_target.position.x=0.0
            base_target.position.y=0.0
            base_target.position.z=0.0

            if part.need_flip:
                part_yaw=1.574971916153666
            else:
                part_yaw=1.574971916153666 -abs(GetYaw(part.pose.orientation))
            base_target.orientation=QuaternionFromRPY(part_yaw,0,0) 
            part_to_gripper=Pose2Matrix(base_target)
            target_matrix=target_matrix@part_to_gripper     
        
        target_matrix[0,3]=target_matrix[0,3]+0.3
        self.MOV_M_ceiling(target_matrix)
        
        if not self.ceiling_robot_gripper_state.attached:
            print("Faulty Gripper ")
            self.oredr_length[subtask.order_id]=self.oredr_length[subtask.order_id]+1
            self.assembly_deque.appendleft(subtask)
            return False
            

        p1=copy.deepcopy(target_matrix)
        p1[0,3]=p1[0,3]-0.2
        p1[1,3]=p1[1,3]-0.095
        self.MOV_M_ceiling(p1,time_factor=3)
        sleep(0.2)

        if self.ceiling_robot_fault_event.is_set():
            self.ceiling_robot_fault_event.clear()
            raise CeilingRobotFaultException("Ceiling robot is in fault state.")     
        
        p1[0,3]=p1[0,3]-0.045
        self.MOV_M_ceiling(p1,time_factor=5)
        sleep(0.2)
        
        p2=copy.deepcopy(p1)
        p2[1,3]=p2[1,3]+0.05
        self.MOV_M_ceiling(p2,time_factor=5)
        


#endregion 

#region ## ###################################### Sensor  ###############################################################################      

    def left_bins_camera_cb(self, msg):
        if not self.left_bins_camera_recieved_data:
            self.get_logger().info("Received data from left bins camera")
            self.left_bins_camera_recieved_data = True

        self.left_bins_parts_ = msg.part_poses
        self.left_bins_camera_pose_ = msg.sensor_pose

    def right_bins_camera_cb(self, msg):
        if not self.right_bins_camera_recieved_data:
            self.get_logger().info("Received data from right bins camera")
            self.right_bins_camera_recieved_data = True

        self.right_bins_parts_ = msg.part_poses
        self.right_bins_camera_pose_ = msg.sensor_pose

    #零件列表更新 输入：某个零件类 零件列表 零件位置误差
    def parts_lsit_update(self, part, parts_list, part_move_skew = normal_part_move_skew):
        current_time = self.get_clock().now() 
        current_time=current_time.nanoseconds/1e9
        # print("current_time",current_time) 
        part.set_time_stamp(current_time)
        # part.pose.position.z = self.part_position_z_limit(part)
        in_list_flag = 0
        if parts_list:
            parts_list_len = len(parts_list)                     
            for parts_list_i in range(parts_list_len):
                if Part_Compare(part, parts_list[parts_list_i], part_move_skew):      # 为真，是同一个零件
                    if part.location == 'conveyor':
                        if abs(current_time - parts_list[parts_list_i].time_stamp) < 0.3:
                            part.final_check = parts_list[parts_list_i].final_check
                            part.u_id = parts_list[parts_list_i].u_id                           
                            parts_list[parts_list_i] = part
                            in_list_flag = 1
                            break
                    else:
                        part.u_id = parts_list[parts_list_i].u_id
                        part.final_check = parts_list[parts_list_i].final_check
                        last_msg_weight = 0.15
                        
                        part.pose.position.x = (1-last_msg_weight) * part.pose.position.x + last_msg_weight * parts_list[parts_list_i].pose.position.x
                        part.pose.position.y = (1-last_msg_weight) * part.pose.position.y + last_msg_weight * parts_list[parts_list_i].pose.position.y
                        part.pose.position.z = (1-last_msg_weight) * part.pose.position.z + last_msg_weight * parts_list[parts_list_i].pose.position.z
                        
                        part.pose.orientation.x = (1-last_msg_weight) * part.pose.orientation.x + last_msg_weight * parts_list[parts_list_i].pose.orientation.x
                        part.pose.orientation.y = (1-last_msg_weight) * part.pose.orientation.y + last_msg_weight * parts_list[parts_list_i].pose.orientation.y
                        part.pose.orientation.z = (1-last_msg_weight) * part.pose.orientation.z + last_msg_weight * parts_list[parts_list_i].pose.orientation.z
                        part.pose.orientation.w = (1-last_msg_weight) * part.pose.orientation.w + last_msg_weight * parts_list[parts_list_i].pose.orientation.w
                        parts_list[parts_list_i] = part
                        in_list_flag = 1
                        break 

        if not in_list_flag:
            # print("不是同一个零件:",part.pose.position)
            if part.location == 'conveyor':
                # print("part.pose.position.y",part.pose.position.y)
                if part.pose.position.y >=3.9:#4.0
                    self.u_id_count = self.u_id_count +1
                    part.u_id = self.u_id_count
                    parts_list.append(part)
                    self.new_part_dict[part.location] = part
                    self.new_part_flag_dict[part.location] = True 
            else:
                self.u_id_count = self.u_id_count +1
                part.u_id = self.u_id_count
                parts_list.append(part)
                self.new_part_dict[part.location] = part
                self.new_part_flag_dict[part.location] = True 



    def rgbd_kts1_image_callback(self,msg):
        try:
            self.rgbd_kts1_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            top_left_corner = (17, 189)  # 示例坐标，您需要根据实际情况修改
            bottom_right_corner = (627, 440)  # 示例坐标，您需要根据实际情况修改
            cropped_table = self.rgbd_kts1_image[top_left_corner[1]:bottom_right_corner[1], top_left_corner[0]:bottom_right_corner[0]]
            
            height, width, _ = cropped_table.shape

            # 计算每个部分的高度
            part_height = width // 3

            # 分割图像为上、中、下三部分
            upper_part = cropped_table[:,0:part_height]
            middle_part = cropped_table[:,part_height:part_height*2]
            lower_part = cropped_table[:,part_height*2:]
            
            three_parts = [upper_part, middle_part, lower_part]
            selected_contour=None
            for i, image in enumerate(three_parts):

                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                _, threshold = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)


                contours, tree = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if 1000 < area < 8000:
                        selected_contour = contour
                        break
            
                # 在lower_part上绘制选定的轮廓
                if selected_contour is not None:

                    x, y, w, h = cv2.boundingRect(selected_contour)
                    extracted_area = image[y:y + h, x:x + w]
                    height, width, channels = extracted_area.shape

                    # Create a white border with a width of 20 units
                    border_width = 10
                    white_border = np.full((height + 2 * border_width, width + 2 * border_width, channels), 255, dtype=np.uint8)

                    # Combine the original image with the border
                    white_border[border_width:-border_width, border_width:-border_width] = extracted_area
                    # enlarged_area = cv2.resize(extracted_area, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

                    gray = cv2.cvtColor(white_border , cv2.COLOR_BGR2GRAY)

                    # OpenCV 4.7+ 去掉了 Dictionary_get；使用 getPredefinedDictionary 兼容新版
                    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
                    # OpenCV 4.7+ 使用 DetectorParameters() 构造函数替代 DetectorParameters_create
                    parameters = cv2.aruco.DetectorParameters()
                    # OpenCV 4.7+ 建议使用 ArucoDetector.detectMarkers；部分发行版已移除顶层 detectMarkers
                    if hasattr(cv2.aruco, "ArucoDetector"):
                        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
                        corners, ids, rejectedImgPoints = detector.detectMarkers(gray)
                    else:  # 兼容旧版
                        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
                    # print('abcd',ids)

                    now_slot='slot'+str(i+1)
                    if ids is not None:
                        self.tray_1_slots[now_slot]=ids
                    else:
                        # print(f"阈值为时，不行:")
                        pass
            # print(self.tray_1_slots)    

        except CvBridgeError as e:
            print(e)

    def rgbd_kts2_image_callback(self,msg):
        try:
            self.rgbd_kts2_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
            # top_left_corner = (280, 34)  # 示例坐标，您需要根据实际情况修改
            # bottom_right_corner = (690, 820)  # 示例坐标，您需要根据实际情况修改
            top_left_corner = (17, 189)  # 示例坐标，您需要根据实际情况修改
            bottom_right_corner = (627, 440)  # 示例坐标，您需要根据实际情况修改
            cropped_table = self.rgbd_kts2_image[top_left_corner[1]:bottom_right_corner[1], top_left_corner[0]:bottom_right_corner[0]]
            
            height, width, _ = cropped_table.shape

            # 计算每个部分的高度
            part_height = width // 3

            # 分割图像为上、中、下三部分
            upper_part = cropped_table[:,0:part_height]
            middle_part = cropped_table[:,part_height:part_height*2]
            lower_part = cropped_table[:,part_height*2:]
            
            three_parts = [upper_part, middle_part, lower_part]
            selected_contour=None
            for i, image in enumerate(three_parts):

                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                _, threshold = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)     

                contours, tree = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if 1000 < area < 8000:
                        selected_contour = contour
                        break
            
                # 在lower_part上绘制选定的轮廓
                if selected_contour is not None:

                    x, y, w, h = cv2.boundingRect(selected_contour)
                    extracted_area = image[y:y + h, x:x + w]
                    height, width, channels = extracted_area.shape

                    # Create a white border with a width of 20 units
                    border_width = 10
                    white_border = np.full((height + 2 * border_width, width + 2 * border_width, channels), 255, dtype=np.uint8)

                    # Combine the original image with the border
                    white_border[border_width:-border_width, border_width:-border_width] = extracted_area
                    # enlarged_area = cv2.resize(extracted_area, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

                    gray = cv2.cvtColor(white_border , cv2.COLOR_BGR2GRAY)

                    # OpenCV 4.7+ 去掉了 Dictionary_get；使用 getPredefinedDictionary 兼容新版
                    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
                    # OpenCV 4.7+ 使用 DetectorParameters() 构造函数替代 DetectorParameters_create
                    parameters = cv2.aruco.DetectorParameters()
                    # OpenCV 4.7+ 建议使用 ArucoDetector.detectMarkers；部分发行版已移除顶层 detectMarkers
                    if hasattr(cv2.aruco, "ArucoDetector"):
                        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
                        corners, ids, rejectedImgPoints = detector.detectMarkers(gray)
                    else:  # 兼容旧版
                        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
                    now_slot='slot'+str(i+4)
                    if ids is not None:
                        self.tray_2_slots[now_slot]=ids
                    else:
                        pass
            # print(self.tray_2_slots) 
        except CvBridgeError as e:
            print(e)
         
    def find_tray_slot(self,tray_id):
        sleep(0.1)
        self.tray_slots.update(self.tray_1_slots)
        self.tray_slots.update(self.tray_2_slots)
        # print("self.tray_slots槽里面有:",self.tray_slots)
        for slot, id in self.tray_slots.items():
            if id == tray_id:
                return slot
        return None       

    def kts1_camera_cb(self,msg):
        # print("kts1_camera_cb")
        self.heart_beat = self.get_clock().now()
        self.heart_beat=self.heart_beat.nanoseconds/1e9
        self.kts1_camera_flag = True
        if self.kts1_camera_flag:
            for model in msg.tray_poses:
                tray_id=model.id
                tray_name = "tray_"+str(tray_id)
                
                part_pose = Pose()              # part to world
                part_pose.position.x = msg.sensor_pose.position.x+model.pose.position.z
                part_pose.position.y = msg.sensor_pose.position.y+model.pose.position.y
                part_pose.position.z = msg.sensor_pose.position.z-model.pose.position.x
                p = quaternion_multiply( \
                    [msg.sensor_pose.orientation.x, msg.sensor_pose.orientation.y,msg.sensor_pose.orientation.z,msg.sensor_pose.orientation.w],  \
                    [model.pose.orientation.x,model.pose.orientation.y,model.pose.orientation.z,model.pose.orientation.w])
                part_pose.orientation.x = p[0]
                part_pose.orientation.y = p[1]
                part_pose.orientation.z = p[2]
                part_pose.orientation.w = p[3]  

                # print("tray_id",tray_id,"tray_pose",part_pose.position)

                #根据区域确定托盘是否在桌子上
                if  Define_tray_is_in_effective_table_range(tray_name, part_pose.position, tray_table_boundary['tray_table_1_x'],tray_table_boundary['tray_table_1_y'], tables_tray_hight):
                    part = sPart(tray_name,"kts1",part_pose)
                    self.parts_lsit_update(part, self.tray_table_1)
                    # print(tray_name,"on tray_table_1")
                    continue

        self.kts1_camera_flag = False
        # print("total number1:%d", len(self.tray_table_1))
        # print("---------")

    def kts2_camera_cb(self,msg):
        # print("kts2_camera_cb")
        self.heart_beat = self.get_clock().now()
        self.heart_beat=self.heart_beat.nanoseconds/1e9
        self.kts2_camera_flag = True
        if self.kts2_camera_flag:
            for model in msg.tray_poses:
                tray_id=model.id
                tray_name = "tray_"+str(tray_id)
                
                part_pose = Pose()              # part to world
                part_pose.position.x = msg.sensor_pose.position.x+model.pose.position.z
                part_pose.position.y = msg.sensor_pose.position.y+model.pose.position.y
                part_pose.position.z = msg.sensor_pose.position.z-model.pose.position.x
                p = quaternion_multiply( \
                    [msg.sensor_pose.orientation.x, msg.sensor_pose.orientation.y,msg.sensor_pose.orientation.z,msg.sensor_pose.orientation.w],  \
                    [model.pose.orientation.x,model.pose.orientation.y,model.pose.orientation.z,model.pose.orientation.w])
                part_pose.orientation.x = p[0]
                part_pose.orientation.y = p[1]
                part_pose.orientation.z = p[2]
                part_pose.orientation.w = p[3]  

                # print("tray_id",tray_id,"tray_pose",part_pose.position)

                #根据区域确定托盘是否在桌子上
                if  Define_tray_is_in_effective_table_range(tray_name, part_pose.position, tray_table_boundary['tray_table_2_x'],tray_table_boundary['tray_table_2_y'], tables_tray_hight):
                    part = sPart(tray_name,"kts2",part_pose)
                    self.parts_lsit_update(part, self.tray_table_2)
                    # print(tray_name,"on tray_table_2")
                    continue

        self.kts2_camera_flag = False
        # print("total number1:%d", len(self.tray_table_2))
        # print("---------")


    def AGV1_status_callback(self,msg):
        self.AGV1_location_flag = True
        if self.AGV1_location_flag:
            agv1_location = msg.location
            if agv1_location == 0:
                self.AGV_location['agv1'] = 'KITTING'
            if agv1_location == 1:
                self.AGV_location['agv1'] = 'ASSEMBLY_FRONT'
            if agv1_location == 2:
                self.AGV_location['agv1'] = 'ASSEMBLY_BACK'
            if agv1_location == 3:
                self.AGV_location['agv1'] = 'WAREHOUSE'
            if agv1_location == 99: #正在运动
                self.AGV_location['agv1'] = "moving"
                
             
            self.AGV1_location_flag = False 
    
    def AGV2_status_callback(self,msg):
        self.AGV2_location_flag = True
        if self.AGV2_location_flag:
            agv2_location = msg.location
            if agv2_location == 0:
                self.AGV_location['agv2'] = 'KITTING'
            if agv2_location == 1:
                self.AGV_location['agv2'] = 'ASSEMBLY_FRONT'
            if agv2_location == 2:
                self.AGV_location['agv2'] = 'ASSEMBLY_BACK'
            if agv2_location == 3:
                self.AGV_location['agv2'] = 'WAREHOUSE'
            if agv2_location == 99: #正在运动
                self.AGV_location['agv2'] = "moving"
             
            self.AGV1_location_flag = False 

    def AGV3_status_callback(self,msg):
        self.AGV3_location_flag = True
        if self.AGV3_location_flag:
            agv3_location = msg.location
            if agv3_location == 0:
                self.AGV_location['agv3'] = 'KITTING'
            if agv3_location == 1:
                self.AGV_location['agv3'] = 'ASSEMBLY_FRONT'
            if agv3_location == 2:
                self.AGV_location['agv3'] = 'ASSEMBLY_BACK'
            if agv3_location == 3:
                self.AGV_location['agv3'] = 'WAREHOUSE'
            if agv3_location == 99: #正在运动
                self.AGV_location['agv3'] = "moving"
             
            self.AGV1_location_flag = False 

    def AGV4_status_callback(self,msg):
        self.AGV4_location_flag = True
        if self.AGV4_location_flag:
            agv4_location = msg.location
            if agv4_location == 0:
                self.AGV_location['agv4'] = 'KITTING'
            if agv4_location == 1:
                self.AGV_location['agv4'] = 'ASSEMBLY_FRONT'
            if agv4_location == 2:
                self.AGV_location['agv4'] = 'ASSEMBLY_BACK'
            if agv4_location == 3:
                self.AGV_location['agv4'] = 'WAREHOUSE'
            if agv4_location == 99: #正在运动
                self.AGV_location['agv4'] = "moving"
             
            self.AGV1_location_flag = False 

        
            #  basic logical camera callback  0413 
    # def left_basic_logical_camera_callback(self,msg : BasicLogicalCameraImage):
    #     print("hu=====")
    #     n=msg.part_poses
    #     print(msg.part_poses)

    #     # msg.part_poses[0]
    #     # self.get_logger().info(f"5555 {len(msg.part_poses)}")
    #     # self.get_logger().info(f"5555 {msg.part_poses[0]}")
    #     self._left_basic_bins_parts = msg.part_poses


    #     self._left_basic_bins_camera_pose = msg.sensor_pose
    #     self.get_logger().info(f"9999 {self._left_basic_bins_camera_pose}")
    #     return self._left_basic_bins_parts, self._left_basic_bins_camera_pose
    
    # def right_basic_logical_camera_callback(self,msg : BasicLogicalCameraImage):
    #     self._right_basic_bins_parts = msg.part_poses
    #     self.get_logger().info(f"6666 {self._right_basic_bins_parts}")
    #     self._right_basic_bins_camera_pose = msg.sensor_pose
    #     self.get_logger().info(f"7777 {self._right_basic_bins_camera_pose}")

    def logical_camera_0_callback(self,msg):  
        self.heart_beat = self.get_clock().now()
        self.heart_beat=self.heart_beat.nanoseconds/1e9      
        self.camera_0_flag = True
        if self.camera_0_flag:
            for model in msg.part_poses:
                part_type=model.part.type
                part_color=model.part.color
                part_color_type=determine_part_name(part_type,part_color)
                part_pose = Pose()              # part to world
                part_pose.position.x = msg.sensor_pose.position.x+model.pose.position.z
                part_pose.position.y = msg.sensor_pose.position.y+model.pose.position.y
                part_pose.position.z = msg.sensor_pose.position.z-model.pose.position.x
                p = quaternion_multiply( \
                    [msg.sensor_pose.orientation.x, msg.sensor_pose.orientation.y,msg.sensor_pose.orientation.z,msg.sensor_pose.orientation.w],  \
                    [model.pose.orientation.x,model.pose.orientation.y,model.pose.orientation.z,model.pose.orientation.w])
                part_pose.orientation.x = p[0]
                part_pose.orientation.y = p[1]
                part_pose.orientation.z = p[2]
                part_pose.orientation.w = p[3]  
                # print('sensor pose',msg.sensor_pose)
                # print('model pose',model.pose)
                # print('part pose',part_pose)
                
                # print("part_type1",part_color_type,"part_pose",part_pose.position)

                
                
                # 根据区域决定物体在bin1或者bin4或agv1上  ？？？？？
                if  Is_In_Effective_Range(part_color_type, part_pose.position, bins_ks_boundary["bin1_x"] ,bins_ks_boundary["bin1_y"], bins_product_height_flip):
                    #决定是否需要翻转
                    need_flip = determine_part_is_need_flip(part_color_type,part_pose)
                    part = sPart(part_color_type,"bin1",part_pose,need_flip)
                    self.parts_lsit_update(part, self.bin1_parts)
                    if not self.bin1_6_need_clean:
                        self.bin1_6_need_clean=is_close_bin1(part.pose.position.x,part.pose.position.y)
                        self.bin1_6_part=part
 
                    # print(part_color_type,"on bin1_y")
                    continue
                if  Is_In_Effective_Range(part_color_type, part_pose.position, bins_ks_boundary["bin4_x"] ,bins_ks_boundary["bin4_y"], bins_product_height_flip):
                    #决定是否需要翻转
                    need_flip = determine_part_is_need_flip(part_color_type,part_pose)
                    part = sPart(part_color_type,"bin4",part_pose,need_flip)
                    self.parts_lsit_update(part, self.bin4_parts)
                    #print(part_color_type,"on bin4_y")
                    continue
                if  Is_In_Effective_Range(part_color_type, part_pose.position, bins_ks_boundary["bin2_x"] ,bins_ks_boundary["bin2_y"], bins_product_height_flip):
                    
                    #决定是否需要翻转
                    need_flip = determine_part_is_need_flip(part_color_type,part_pose)
                    part = sPart(part_color_type,"bin2",part_pose,need_flip)
                    self.parts_lsit_update(part, self.bin2_parts)
                    # print(part_color_type,"on bin2_y")
                    continue
                if  Is_In_Effective_Range(part_color_type, part_pose.position, bins_ks_boundary["bin3_x"] ,bins_ks_boundary["bin3_y"], bins_product_height_flip):
                    #决定是否需要翻转
                    need_flip = determine_part_is_need_flip(part_color_type,part_pose)
                    part = sPart(part_color_type,"bin3",part_pose,need_flip)
                    self.parts_lsit_update(part, self.bin3_parts)
                    #print(part_color_type,"on bin3_y")
                    continue
                # if  Is_In_Effective_Range(part_color_type, part_pose.position, agv_ks_boundary["agv1_x"] ,agv_ks_boundary["agv1_y"], agv_product_height_with_tray_flip) :
                # # and self.AGV_state['agv1'] == 'READY_TO_DELIVER':
                #     part = sPart(part_color_type,"agv1_ks1_tray",part_pose)
                #     self.parts_lsit_update(part, self.agv1_ks1_tray_parts)
                #     #print(part_color_type,"on agv1_y")
                #     continue
            self.camera_0_flag =False
            #print("bin1_parts:")
            # print("total number1:%d", len(self.bin1_parts))
            # print("total number4:%d", len(self.bin4_parts))
            # print("total number agv1:%d", len(self.agv1_ks1_tray_parts))

    def logical_camera_1_callback(self,msg):         
        self.heart_beat = self.get_clock().now()
        self.heart_beat=self.heart_beat.nanoseconds/1e9
        self.camera_1_flag = True
        if self.camera_1_flag:
            for model in msg.part_poses:
                part_type=model.part.type
                part_color=model.part.color
                part_color_type=determine_part_name(part_type,part_color)
                part_pose = Pose()              # part to world
                part_pose.position.x = msg.sensor_pose.position.x+model.pose.position.z
                part_pose.position.y = msg.sensor_pose.position.y+model.pose.position.y
                part_pose.position.z = msg.sensor_pose.position.z-model.pose.position.x
                p = quaternion_multiply( \
                    [msg.sensor_pose.orientation.x, msg.sensor_pose.orientation.y,msg.sensor_pose.orientation.z,msg.sensor_pose.orientation.w],  \
                    [model.pose.orientation.x,model.pose.orientation.y,model.pose.orientation.z,model.pose.orientation.w])
                part_pose.orientation.x = p[0]
                part_pose.orientation.y = p[1]
                part_pose.orientation.z = p[2]
                part_pose.orientation.w = p[3]  
                
                # print("part_type1",part_color_type,"part_pose",part_pose.position)

                
                
                # to determin the location
                if  Is_In_Effective_Range(part_color_type, part_pose.position, bins_ks_boundary["bin6_x"] ,bins_ks_boundary["bin6_y"], bins_product_height_flip):
                    #决定是否需要翻转
                    need_flip = determine_part_is_need_flip(part_color_type,part_pose)
                    part = sPart(part_color_type,"bin6",part_pose,need_flip)
                    self.parts_lsit_update(part, self.bin6_parts)
                    if not self.bin6_6_need_clean:
                        self.bin6_6_need_clean=is_close_bin6(part.pose.position.x,part.pose.position.y)
                        self.bin6_6_part=part
                        self.update_grid_status(part.pose.position.x,part.pose.position.y,-1.90, -2.625,self.bin6_grid_status )
                    #print(part_color_type,"on bin6_y")
                    continue
                if  Is_In_Effective_Range(part_color_type, part_pose.position, bins_ks_boundary["bin7_x"] ,bins_ks_boundary["bin7_y"], bins_product_height_flip):
                    #决定是否需要翻转
                    need_flip = determine_part_is_need_flip(part_color_type,part_pose)
                    part = sPart(part_color_type,"bin7",part_pose,need_flip)
                    self.parts_lsit_update(part, self.bin7_parts)
                    #print(part_color_type,"on bin7_y")
                    continue
                if  Is_In_Effective_Range(part_color_type, part_pose.position, bins_ks_boundary["bin5_x"] ,bins_ks_boundary["bin5_y"], bins_product_height_flip):
                    #决定是否需要翻转
                    need_flip = determine_part_is_need_flip(part_color_type,part_pose)
                    part = sPart(part_color_type,"bin5",part_pose,need_flip)
                    self.parts_lsit_update(part, self.bin5_parts)
                    # print(part_color_type,"on bin5_y")
                    continue
                if  Is_In_Effective_Range(part_color_type, part_pose.position, bins_ks_boundary["bin8_x"] ,bins_ks_boundary["bin8_y"], bins_product_height_flip):
                    #决定是否需要翻转
                    need_flip = determine_part_is_need_flip(part_color_type,part_pose)
                    part = sPart(part_color_type,"bin8",part_pose,need_flip)
                    self.parts_lsit_update(part, self.bin8_parts)
                    #print(part_color_type,"on bin8_y")
                    continue
                # if  Is_In_Effective_Range(part_color_type, part_pose.position, agv_ks_boundary["agv2_x"] ,agv_ks_boundary["agv2_y"], agv_product_height_with_tray_flip) :
                # # and self.AGV_state['agv1'] == 'READY_TO_DELIVER':
                #     part = sPart(part_color_type,"agv2_ks2_tray",part_pose)
                #     self.parts_lsit_update(part, self.agv2_ks2_tray_parts)
                #     #print(part_color_type,"on agv2_y")
                #     continue
            self.camera_1_flag =False



    def logical_camera_conveyor_callback(self,msg):
        self.heart_beat = self.get_clock().now()
        self.heart_beat=self.heart_beat.nanoseconds/1e9
        for model in msg.part_poses:
            part_type = model.part.type
            part_color=model.part.color           
            part_color_type=determine_part_name(part_type,part_color)
            
            # frame_to_word
            part_pose = Pose()
            part_pose.position.x = msg.sensor_pose.position.x+model.pose.position.z
            part_pose.position.y = msg.sensor_pose.position.y+model.pose.position.y
            part_pose.position.z = msg.sensor_pose.position.z-model.pose.position.x
            
            # part_pose.position.z = part_on_conveyor_z[part_color_type]

            p = quaternion_multiply( \
                [msg.sensor_pose.orientation.x, msg.sensor_pose.orientation.y,msg.sensor_pose.orientation.z,msg.sensor_pose.orientation.w],  \
                [model.pose.orientation.x,model.pose.orientation.y,model.pose.orientation.z,model.pose.orientation.w])
            part_pose.orientation.x = p[0]
            part_pose.orientation.y = p[1]
            part_pose.orientation.z = p[2]
            part_pose.orientation.w = p[3]
                # to determin the location
            if  Is_In_Effective_Range(part_color_type, part_pose.position, convey_boundary["convey_x"] ,convey_boundary["convey_y"], convey_product_height_flip):
                need_flip = determine_part_is_need_flip_on_convey(part_color_type,part_pose)
                part = sPart(part_color_type,"conveyor",part_pose,need_flip)
                part.time_stamp=self.get_clock().now().nanoseconds/1e9
                
                self.add_part(part)
        #     print("part_color_type",part_color_type,"part_pose",part_pose.position)
        if self.pre_length!=len(self.convey_parts):
            self.logical_camera_conveyor_parts=self.convey_parts
            self.pre_length=len(self.convey_parts)
            test=[pa for pa in self.convey_parts if pa.type=="assembly_sensor_green"]
            print("total number CONVEYOR:%d", len(self.convey_parts),"   ",len(test))
        # print("-------------------------")



    #通过零件的id删除该零件  输入：零件列表 零件类
    def search_del_part_use_id(self,parts_list, part):
        list_len = len(parts_list)
        for list_count in range(list_len):
            if parts_list[list_count].u_id == part.u_id:
                del parts_list[list_count]
                break
    
    #删除某个零件 输入：需要删除的零件类
    def del_part_from_parts_list(self, del_part):
        if del_part.location == 'bin1':
            self.search_del_part_use_id(self.bin1_parts, del_part)
        elif del_part.location == 'bin2':
            self.search_del_part_use_id(self.bin2_parts, del_part)
        elif del_part.location == 'bin3':
            self.search_del_part_use_id(self.bin3_parts, del_part)
        elif del_part.location == 'bin4':
            self.search_del_part_use_id(self.bin4_parts, del_part)            
        elif del_part.location == 'bin5':
            self.search_del_part_use_id(self.bin5_parts, del_part)            
        elif del_part.location == 'bin6':
            self.search_del_part_use_id(self.bin6_parts, del_part)            
        elif del_part.location == 'bin7':
            self.search_del_part_use_id(self.bin7_parts, del_part)            
        elif del_part.location == 'bin8':
            self.search_del_part_use_id(self.bin8_parts, del_part)
        
        # elif del_part.location == 'agv1_ks1_tray':
        #     self.search_del_part_use_id(self.agv1_ks1_tray_parts, del_part)
        # elif del_part.location == 'agv2_ks2_tray':
        #     self.search_del_part_use_id(self.agv2_ks2_tray_parts, del_part)                  
        # elif del_part.location == 'agv3_ks3_tray':
        #     self.search_del_part_use_id(self.agv3_ks3_tray_parts, del_part)            
        # elif del_part.location == 'agv4_ks4_tray':
        #     self.search_del_part_use_id(self.agv4_ks4_tray_parts, del_part) 

        # elif del_part.location == 'agv1_as1_tray':
        #     self.search_del_part_use_id(self.logical_camera_as_11_parts, del_part)
        # elif del_part.location == 'agv1_as2_tray':
        #     self.search_del_part_use_id(self.logical_camera_as_21_parts, del_part)
        # elif del_part.location == 'agv2_as1_tray':
        #     self.search_del_part_use_id(self.logical_camera_as_12_parts, del_part)
        # elif del_part.location == 'agv2_as2_tray':
        #     self.search_del_part_use_id(self.logical_camera_as_22_parts, del_part)            
        # elif del_part.location == 'agv3_as3_tray':
        #     self.search_del_part_use_id(self.logical_camera_as_33_parts, del_part)
        # elif del_part.location == 'agv3_as4_tray':
        #     self.search_del_part_use_id(self.logical_camera_as_43_parts, del_part)
        # elif del_part.location == 'agv4_as3_tray':
        #     self.search_del_part_use_id(self.logical_camera_as_34_parts, del_part)
        # elif del_part.location == 'agv4_as4_tray':
        #     self.search_del_part_use_id(self.logical_camera_as_44_parts, del_part)
        
        # elif del_part.location == 'conveyor':
        #     self.search_del_part_use_id(self.logical_camera_conveyor_parts, del_part)  
        
        elif del_part.location == 'tray_table_1':
            self.search_del_part_use_id(self.tray_table_1, del_part)
        elif del_part.location == 'tray_table_2':
            self.search_del_part_use_id(self.tray_table_2, del_part)

    #搜索在传送带上的零件通过颜色类型  输入：零件颜色类型 assembly_battery_red
    def search_part_on_conveyor(self,part_color_type):

        has_part_list=[]
        if len(self.logical_camera_conveyor_parts)>=1:
            print ("parts on conveyor:",len(self.logical_camera_conveyor_parts))

            # if time out then te
            current_time = self.get_clock().now()
            current_time=current_time.nanoseconds/1e9

            for part in self.logical_camera_conveyor_parts:
                if (current_time - part.time_stamp)*conveyor_vel > 8:
                    if part in self.convey_parts:
                        self.convey_parts.remove(part)
            
            for part in self.logical_camera_conveyor_parts:
                if part.type == part_color_type:
                    has_part_list.append(part)
            return has_part_list
        else:
            return []

    #搜索某个零件 输入：需要搜索的零件类
    def search_part_use_part(self, part):
        self.all_parts_list_old_update()

        parts_list = self.bin1_parts + self.bin4_parts + self.agv1_ks1_tray_parts + \
            self.bin2_parts + self.bin3_parts + self.agv2_ks2_tray_parts + \
            self.bin6_parts + self.bin7_parts + self.agv3_ks3_tray_parts + \
            self.bin5_parts + self.bin8_parts + self.agv4_ks4_tray_parts + \
            self.logical_camera_as_11_parts + self.logical_camera_as_12_parts + \
            self.logical_camera_as_21_parts + self.logical_camera_as_22_parts + \
            self.logical_camera_as_33_parts + self.logical_camera_as_34_parts + \
            self.logical_camera_as_43_parts + self.logical_camera_as_44_parts 
        list_len = len(parts_list)
        for list_count in range(list_len):
            if parts_list[list_count].u_id == part.u_id:
                return parts_list[list_count]

    #传感器是否故障 输出：false故障了   true正常
    def is_alive(self):
        current_time = self.get_clock().now()       
        current_time=current_time.nanoseconds/1e9
        if current_time - self.heart_beat > 0.5:
            self.has_blocked = True
            self.has_blocked_for_check = True
            return False
        else:
            return True

    #更新环境中零件列表
    def all_parts_list_old_update(self):
        self.parts_list_old_update(self.bin1_parts)
        self.parts_list_old_update(self.bin2_parts)
        self.parts_list_old_update(self.bin3_parts)
        self.parts_list_old_update(self.bin4_parts)
        self.parts_list_old_update(self.bin5_parts)
        self.parts_list_old_update(self.bin6_parts)
        self.parts_list_old_update(self.bin7_parts)
        self.parts_list_old_update(self.bin8_parts)
        self.parts_list_old_update(self.agv1_ks1_tray_parts)
        self.parts_list_old_update(self.agv2_ks2_tray_parts)
        self.parts_list_old_update(self.agv3_ks3_tray_parts)
        self.parts_list_old_update(self.agv4_ks4_tray_parts)
        self.parts_list_old_update(self.tray_table_1)  
        self.parts_list_old_update(self.tray_table_2) 

    #更新某个容器中的零件列表  输入：某个箱子 self.bin1_parts
    def parts_list_old_update(self, parts_list, old_time = 1):

        
        while self.has_blocked and self.is_alive():
            sleep(1)
            self.has_blocked = False

        if self.is_alive():
            current_time = self.get_clock().now()
            current_time=current_time.nanoseconds/1e9
            part_old_flag = True
            while part_old_flag:
                list_len = len(parts_list)
                part_old_flag = False
                self.threadLock.acquire()
                for list_count in range(0, list_len):
                    if list_count < len(parts_list) and parts_list[list_count].time_stamp:
                        if list_count < len(parts_list) and abs(current_time - parts_list[list_count].time_stamp) > old_time:
                            del parts_list[list_count]
                            part_old_flag = True
                            break
                self.threadLock.release()

    #零件类型分类 输出：字典key：零件颜色类型 vaule：零件
    def part_type_sort(self):
        self.all_parts_list_old_update()
        # clear 
        self.parts_type_dict.clear()
        # merge all_part

        all_parts = self.bin1_parts + self.bin4_parts + self.bin2_parts + self.bin3_parts +\
                    self.bin6_parts + self.bin7_parts + self.bin5_parts + self.bin8_parts + \
                    self.logical_camera_as_11_parts + self.logical_camera_as_12_parts + \
                    self.logical_camera_as_21_parts + self.logical_camera_as_22_parts + \
                    self.logical_camera_as_33_parts + self.logical_camera_as_34_parts + \
                    self.logical_camera_as_43_parts + self.logical_camera_as_44_parts + \
                    self.agv1_ks1_tray_parts + self.agv2_ks2_tray_parts + self.agv3_ks3_tray_parts + self.agv4_ks4_tray_parts
        # print("self.bin1_parts",len(self.bin1_parts))
        # print("len(all_parts)",len(all_parts))
        if len(all_parts)>=1:
            # build dict
            for part in all_parts:
                self.parts_type_dict.setdefault(part.type,[]).append(part)
        else:
            pass

    #通过零件颜色类型搜索零件
    def search_part_type(self, part_color_type):
        self.part_type_sort()
        # print("self.parts_type_dict",len(self.parts_type_dict))
        if part_color_type in self.parts_type_dict.keys():
            return self.parts_type_dict[part_color_type]
        else:
            return []
        
    #零件类型分类 输出：字典key：零件颜色类型 vaule：零件
    def part_type_sort_on_bins(self):
        self.all_parts_list_old_update()
        # clear 
        self.parts_type_dict.clear()
        # merge all_part

        all_parts_on_bins = self.bin1_parts + self.bin4_parts + self.bin2_parts + self.bin3_parts +\
                    self.bin6_parts + self.bin7_parts + self.bin5_parts + self.bin8_parts 
        # print("self.bin1_parts",len(self.bin1_parts))
        # print("len(all_parts)",len(all_parts))
        if len(all_parts_on_bins)>=1:
            # build dict
            for part in all_parts_on_bins:
                self.parts_type_dict.setdefault(part.type,[]).append(part)
        else:
            pass

    #通过零件颜色类型搜索零件
    def search_part_on_bins(self, part_color_type):
        self.part_type_sort_on_bins()
        # print("self.parts_type_dict",len(self.parts_type_dict))
        if part_color_type in self.parts_type_dict.keys():
            return self.parts_type_dict[part_color_type]
        else:

            return self.parts_type_dict['pump']
        
    #零件位置分类 输出：字典key：装零件的箱子 vaule：箱子上的零件
    def part_location_sort(self):
        self.all_parts_list_old_update()
        # clear 
        self.parts_location_dict.clear() 
        # merge all_part
        all_parts = self.bin1_parts + self.bin4_parts + self.bin2_parts + self.bin3_parts +\
                    self.bin6_parts + self.bin7_parts + self.bin5_parts + self.bin8_parts + \
                    self.logical_camera_as_11_parts + self.logical_camera_as_12_parts + \
                    self.logical_camera_as_21_parts + self.logical_camera_as_22_parts + \
                    self.logical_camera_as_33_parts + self.logical_camera_as_34_parts + \
                    self.logical_camera_as_43_parts + self.logical_camera_as_44_parts + \
                    self.agv1_ks1_tray_parts + self.agv2_ks2_tray_parts + self.agv3_ks3_tray_parts + self.agv4_ks4_tray_parts
                    
        if len(all_parts)>=1:
            # build dict
            for part in all_parts:
                self.parts_location_dict.setdefault(part.location,[]).append(part)

            #print test    
            # for key,value in self.parts_location_dict.items():
            #     print(key+": ")
            #     for p in value:
            #         print(p.location)
        else:
            pass

    #通过位置搜索零件  输入“bin1”
    def search_part_location(self,part_location):

        self.part_location_sort()
        if part_location in self.parts_location_dict.keys():
            return self.parts_location_dict[part_location]
        else:
            return False

    #通过位置和零件颜色类型搜索零件
    def search_part_location_type(self,part_location,part_color_type): 
        """
        robot_system 使用
        """  
        self.part_location_sort()
        part_type_list = []
        if part_location in self.parts_location_dict.keys():
            part_list = self.parts_location_dict[part_location]
            if part_list:
                for part in part_list:
                    if part.type == part_color_type:
                        part_type_list.append(part)
                return part_type_list
            else:
              return False  
        else:
            return False     
    
    #托盘id---------托盘信息
    def search_tray_by_tray_id(self,tray_id):
        tray_=None
        # Check table 1
        for tray in self.tray_table_1:
            if tray.type == "tray_"+str(tray_id):
                tray_=tray
                tray_.location="kts1"
                return tray_

        for tray in self.tray_table_2:
            if tray.type == "tray_"+str(tray_id):
                tray_=tray
                tray_.location="kts2"
                return tray_
        print("tray_table_1------------",self.tray_table_1,"self.tray_table_2---------",self.tray_table_2)

        return False   

    def determine_clean_part_bin1(self):
        parts_list = self.bin1_parts + self.bin2_parts 

        for part in self.bin1_parts:
            self.update_grid_status(part.pose.position.x,part.pose.position.y,-1.90, 3.375,self.bin1_grid_status )
        for part in parts_list:
            if part.need_flip:
                print('这里确实有需要翻转的')
                if self.bin1_6_need_clean :
                    print('bin1_6需要移走')
                    if self.find_nearest_empty_grid(-1.90, 3.375,self.bin1_grid_status):
                        x,y=self.find_nearest_empty_grid(-1.90, 3.375,self.bin1_grid_status)
                        self.move_to(self.bin1_6_part.location)
                        self.adjust_part_on_bin_agv(self.bin1_6_part,x,y)
                        self.del_part_from_parts_list(self.bin1_6_part)
                        
                        part_n=copy.deepcopy(self.bin1_6_part)
                        part_n.pose.position.x=x
                        part_n.pose.position.y=y
                        self.parts_lsit_update(part_n, self.bin1_parts)
                        break 
                    else:
                        self.move_to(self.bin1_6_part.location)
                        self.pick_part_on_bin_agv(self.bin1_6_part.location,self.bin1_6_part)
                        self.move_to('can')   
                        
                        self.set_floor_robot_gripper_state(False)   


    def determine_clean_part_bin6(self):
        parts_list =  self.bin6_parts +self.bin5_parts 
        for part in self.bin6_parts:
            self.update_grid_status(part.pose.position.x,part.pose.position.y,-1.90, -2.625,self.bin6_grid_status )
        for part in parts_list:
            if part.need_flip:
                if self.bin6_6_need_clean:
                    if self.find_nearest_empty_grid(-1.90, -2.625,self.bin6_grid_status):
                        x,y=self.find_nearest_empty_grid(-1.90, -2.625,self.bin6_grid_status)
                        self.move_to(self.bin6_6_part.location)
                        self.adjust_part_on_bin_agv(self.bin6_6_part,x,y)
                        self.del_part_from_parts_list(self.bin6_6_part)
                        
                        part_n=copy.deepcopy(self.bin6_6_part)
                        part_n.pose.position.x=x
                        part_n.pose.position.y=y
                        self.parts_lsit_update(part_n, self.bin6_parts)
                        break 
                    else:
                        self.move_to(self.bin6_6_part.location)
                        self.pick_part_on_bin_agv(self.bin6_6_part.location,self.bin6_6_part)
                        self.move_to('can') 
                        
                        self.set_floor_robot_gripper_state(False)      
                

#endregion 
             
    def move_agv(self, agv_num, destination):                              ## 这个destination在kitting 和assembly时有问题
        service_name = '/ariac/move_agv{}'.format(agv_num)
        self.move_agv_clients= self.create_client(MoveAGV, service_name)

        while not self.move_agv_clients.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

        request = MoveAGV.Request()
     
        request.location = destination

        future = self.move_agv_clients.call_async(request)
        
        with self.spin_lock:
            rclpy.spin_until_future_complete(self, future)
        
        while self.AGV_location['agv'+str(agv_num)] != agv_place[destination]:
            sleep(0.2)
                        
        print("AGV has arrived :",agv_place[destination])   
        
        if future.result() is not None:
            return future.result().success
        else:
            self.get_logger().error('Exception while calling service: %r' % future.exception())
            return False     
          

    def lock_agv_tray(self, agv_num: int) -> bool:
            srv_name = f'/ariac/agv{agv_num}_lock_tray'
            self.lock_agv_clients = self.create_client(Trigger, srv_name)

            request = Trigger.Request()
            # 异步发送请求
            future = self.lock_agv_clients .call_async(request)
            # 等待响应
            with self.spin_lock:
                rclpy.spin_until_future_complete(self, future)
            if future.result() is not None:
                return future.result().success
            else:
                self.get_logger().error('Exception while calling service: %r' % future.exception())
                return False              
    
    def get_assembly_poses(self, order_id: str) -> List[PartPose]:
        request = GetPreAssemblyPoses.Request()
        request.order_id = order_id

        future = self.pre_assembly_poses_getter.call_async(request)

        # 等待响应
        with self.spin_lock:
            rclpy.spin_until_future_complete(self, future)
        response = future.result()
        print("------1664----response--",response)

        agv_part_poses = []
        if response.valid_id:
            agv_part_poses = response.parts

            if len(agv_part_poses) == 0:
                self.get_logger().warn('No part poses received')
                return False
        else:
            self.get_logger().warn('Not a valid order ID')
            return False

        return agv_part_poses       
    
    def grid_center(self, row, col,center_x,center_y):
        x = center_x + (col - 1) * self.spacing
        y = center_y + (row - 1) * self.spacing
        return x, y

    def update_grid_status(self, x, y,center_x,center_y,grid_status):

        row = int(round((y - center_y) / self.spacing)) + 1
        col = int(round((x - center_x) / self.spacing)) + 1

        if (row, col) in grid_status:
            grid_status[(row, col)] = True
        else:
            # print("x,y的值是:",(x,y))
            # print("The input coordinates are out of the grid.")
            pass
            # print("grid_status是:::::",grid_status)
            # print("row,col:::::",row,col)

                
    def find_nearest_empty_grid(self,center_x,center_y,grid_status):
        target_row, target_col = 2, 3
        empty_grids = [(row, col) for (row, col), status in grid_status.items() if not status]
        
        print("空格子的状态:",grid_status)
        print("空格子的数量:",len(empty_grids))

        if not empty_grids:
            print("There are no empty grids.")
            return None

        min_distance = float("inf")
        nearest_empty_grid = None
        for row, col in empty_grids:
            distance = math.sqrt((row - target_row) ** 2 + (col - target_col) ** 2)
            if distance < min_distance:
                min_distance = distance
                nearest_empty_grid = (row, col)

        nearest_empty_grid_x = center_x + (nearest_empty_grid[1] - 1) * self.spacing
        nearest_empty_grid_y = center_y + (nearest_empty_grid[0] - 1) * self.spacing
        print("----x----",nearest_empty_grid_x)

        return nearest_empty_grid_x, nearest_empty_grid_y

    def collision_check(self):
        distance= math.sqrt((-1.3 - self.ceiling_base_x) ** 2 + (self.kitting_base_y - self.ceiling_base_y ) ** 2)
        while distance<0.3:
            if (not self.floor_robot_info.is_idle ) and self.ceiling_robot_info.is_idle:    # ceiling 机器人到初始状态躲避
                self.ceiling_arm_init()
            
            if self.floor_robot_info.is_idle and (not self.ceiling_robot_info.is_idle):    # floor 机器人到初始状态躲避
                self.kitting_robot_init('standby')

            if (not self.floor_robot_info.is_idle ) and (not self.ceiling_robot_info.is_idle):   # ceiling 机器人到初始状态并暂停躲避
                pass
                
                
    def calculate_plane_distance(self,human_position: Point, robot_position: Point) -> float:
        dx = human_position.x - robot_position.x
        dy = human_position.y - robot_position.y
        
        distance = math.sqrt(dx**2 + dy**2)
        return distance
    
    def get_order_type(self,order_id):
        for order_msg in self.order_recored_list:
            if order_msg.id == order_id:
                return order_msg.type
        return None  # 如果找不到匹配的 order_id，则返回 None 或适当的默认值

    def add_part(self, part):
        
        # 遍历已存在的零件列表
        for existing_part in self.convey_parts:
            # 检查零件类型和位置是否相同
            if existing_part.type == part.type :
                time_difference = part.time_stamp - existing_part.time_stamp
                distance = time_difference * 0.2  # 根据传送带速度计算距离阈值

                # 如果时间差乘以传送带速度接近零件位置差，则认为是同一个零件，不添加
                if abs(abs(existing_part.pose.position.y - part.pose.position.y) -distance)<0.1:
                    return

        # 如果未找到相同的零件或不满足条件，则将零件添加到列表中
        with self.convey_parts_lock :
            self.convey_parts.append(part)


def main(args=None):
    rclpy.init(args=args)


    interface = SIAInterface()
    
    interface.new_thread()
    
    interface.start_competition()
    
    # interface.wait(2)

    interface.run()
    


    interface.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
