
from geometry_msgs.msg import Point,Pose
from ariac_msgs.msg import KittingPart
import numpy 
from competition_tutorials.my_tf import *
import math
from math import cos 
from math import sin 
from math import atan2 
from math import acos
from math import asin 
from math import sqrt
from math import pi

#######已更新数据###########


#######等待更新数据####################################################################################################################################
class sPart:
    def __init__(self, name, location, pose,need_flip=False):
        self.type = name
        self.location = location 
        self.pose = pose
        self.need_flip=need_flip
        self.time_stamp = False
        self.u_id = -1
        self.final_check = False
    def set_time_stamp(self, time_stamp):
        self.time_stamp = time_stamp

def find_nearest_part(target_part_list_bins, x, y):
    min_distance = float("inf")
    nearest_part = None

    for part in target_part_list_bins:
        part_x = part.pose.position.x
        part_y = part.pose.position.y
        distance = math.sqrt((part_x - x) ** 2 + (part_y - y) ** 2)

        if distance < min_distance:
            min_distance = distance
            nearest_part = part

    return nearest_part
       
conveyor_vel = 0.2  # conveyor velocity                                    # 夹具台-3.7 6.26
conveyor_height = 0.875 
bin_position = {
    'bin1': [-1.9, 3.375, 0.72],
    'bin2': [-1.9, 2.625, 0.72],
    'bin3': [-2.65, 2.625, 0.72],
    'bin4': [-2.65, 3.375, 0.72],
    'bin5': [-1.9, -3.375, 0.72],
    'bin6': [-1.9, -2.625, 0.72],
    'bin7': [-2.65, -2.625, 0.72],
    'bin8': [-2.65, -3.375, 0.72], 
}

slot_reference_position = {
    1: [-0.18, -0.18, 0.0],
    2: [-0.18, -0.00, 0.0],
    3: [ 0.18, 0.18, 0.0],

    4: [-0.0, -0.18, 0.0],
    5: [-0.0, -0.00, 0.0],
    6: [ 0.0, 0.18, 0.0],

    7: [0.18, -0.18, 0.0],
    8: [0.0, -0.00, 0.0],
    9: [ 0.18, 0.18, 0.0],
}

bin_slot_positions={
'bin1_1': [-2.08, 3.195, 0.72],'bin1_2': [-2.08, 3.375, 0.72], 'bin1_3': [-2.08, 3.555, 0.72], 'bin1_4': [-1.9, 3.195, 0.72], 'bin1_5': [-1.9, 3.375, 0.72], 'bin1_6': [-1.9, 3.555, 0.72], 'bin1_7': [-1.72, 3.195, 0.72], 'bin1_8': [-1.72, 3.375, 0.72], 'bin1_9': [-1.72, 3.555, 0.72], 
 'bin2_1': [-2.08, 2.445, 0.72], 'bin2_2': [-2.08, 2.625, 0.72], 'bin2_3': [-2.08, 2.805, 0.72], 'bin2_4': [-1.9, 2.445, 0.72], 'bin2_5': [-1.9, 2.625, 0.72], 'bin2_6': [-1.9, 2.805, 0.72], 'bin2_7': [-1.72, 2.445, 0.72], 'bin2_8': [-1.72, 2.625, 0.72], 'bin2_9': [-1.72, 2.805, 0.72], 
 'bin3_1': [-2.83, 2.445, 0.72], 'bin3_2': [-2.83, 2.625, 0.72], 'bin3_3': [-2.83, 2.805, 0.72], 'bin3_4': [-2.65, 2.445, 0.72], 'bin3_5': [-2.65, 2.625, 0.72], 'bin3_6': [-2.65, 2.805, 0.72], 'bin3_7': [-2.4699999999999998, 2.445, 0.72], 'bin3_8': [-2.4699999999999998, 2.625, 0.72], 'bin3_9': [-2.4699999999999998, 2.805, 0.72],
'bin4_1': [-2.83, 3.195, 0.72], 'bin4_2': [-2.83, 3.375, 0.72], 'bin4_3': [-2.83, 3.555, 0.72], 'bin4_4': [-2.65, 3.195, 0.72], 'bin4_5': [-2.65, 3.375, 0.72], 'bin4_6': [-2.65, 3.555, 0.72], 'bin4_7': [-2.4699999999999998, 3.195, 0.72], 'bin4_8': [-2.4699999999999998, 3.375, 0.72], 'bin4_9': [-2.4699999999999998, 3.555, 0.72], 
'bin5_1': [-2.08, -3.555, 0.72], 'bin5_2': [-2.08, -3.375, 0.72], 'bin5_3': [-2.08, -3.195, 0.72], 'bin5_4': [-1.9, -3.555, 0.72], 'bin5_5': [-1.9, -3.375, 0.72], 'bin5_6': [-1.9, -3.195, 0.72], 'bin5_7': [-1.72, -3.555, 0.72], 'bin5_8': [-1.72, -3.375, 0.72], 'bin5_9': [-1.72, -3.195, 0.72], 
'bin6_1': [-2.08, -2.805, 0.72], 'bin6_2': [-2.08, -2.625, 0.72], 'bin6_3': [-2.08, -2.445, 0.72], 'bin6_4': [-1.9, -2.805, 0.72], 'bin6_5': [-1.9, -2.625, 0.72], 'bin6_6': [-1.9, -2.445, 0.72], 'bin6_7': [-1.72, -2.805, 0.72], 'bin6_8': [-1.72, -2.625, 0.72], 'bin6_9': [-1.72, -2.445, 0.72], 
'bin7_1': [-2.83, -2.805, 0.72], 'bin7_2': [-2.83, -2.625, 0.72], 'bin7_3': [-2.83, -2.445, 0.72], 'bin7_4': [-2.65, -2.805, 0.72], 'bin7_5': [-2.65, -2.625, 0.72], 'bin7_6': [-2.65, -2.445, 0.72], 'bin7_7': [-2.4699999999999998, -2.805, 0.72], 'bin7_8': [-2.4699999999999998, -2.625, 0.72], 'bin7_9': [-2.4699999999999998, -2.445, 0.72], 
'bin8_1': [-2.83, -3.555, 0.72], 'bin8_2': [-2.83, -3.375, 0.72], 'bin8_3': [-2.83, -3.195, 0.72], 'bin8_4': [-2.65, -3.555, 0.72], 'bin8_5': [-2.65, -3.375, 0.72], 'bin8_6': [-2.65, -3.195, 0.72], 'bin8_7': [-2.4699999999999998, -3.555, 0.72], 'bin8_8': [-2.4699999999999998, -3.375, 0.72], 'bin8_9': [-2.4699999999999998, -3.195, 0.72]
}

bin3_8_flip={
    'bin3': [-2.65, 2.625-0.27, 0.72],
    'bin8': [-2.65, -3.375-0.27, 0.72], 
}

agv_place=['KITTING','ASSEMBLY_FRONT','ASSEMBLY_BACK','WAREHOUSE']

floor_bins=['bin1','bin2','bin5','bin6']
ceiling_bins=['bin3','bin4','bin7','bin8']

def split_parts_by_location(target_part_list_bins):

    floor_parts = []
    ceiling_parts = []
    for part in target_part_list_bins:
        if part.location in floor_bins:
            floor_parts.append(part)
        elif part.location in ceiling_bins:
            ceiling_parts.append(part)
    return floor_parts, ceiling_parts

def agv_quadrant_position(agv_id, quadrant):

    agv1_quadrants = {
        1: [-1.94, 4.89],
        2: [-1.94, 4.707],
        3: [-2.196, 4.89],
        4: [-2.196, 4.707],
    }

    agv_position = agv_ks_position[f'agv{agv_id}']
    quadrant_offset = [agv1_quadrants[quadrant][i] - agv_ks_position['agv1'][i] for i in range(2)]
    position = [agv_position[i] + quadrant_offset[i] for i in range(2)]

    return position + [0.771]

ceiling_faulty_part= {

    'assembly_sensor_red': 0.09,    # 2023.3.18
    'assembly_sensor_blue':  0.09,
    'assembly_sensor_green':  0.09,
    'assembly_sensor_purple':  0.09,
    'assembly_sensor_orange':  0.09,

    'assembly_regulator_red': 0.09, 
    'assembly_regulator_blue': 0.09,
    'assembly_regulator_green': 0.09,
    'assembly_regulator_purple': 0.09,
    'assembly_regulator_orange': 0.09,

    'assembly_pump_red': 0.135,
    'assembly_pump_blue':0.135,
    'assembly_pump_green':0.135,
    'assembly_pump_purple':0.135,
    'assembly_pump_orange':0.135,


    'assembly_battery_red':0.075, #2021-04-15
    'assembly_battery_blue':0.075, 
    'assembly_battery_green':0.075,
    'assembly_battery_purple':0.075,
    'assembly_battery_orange':0.075,

    'can':0.15,
    'movable_tray_dark_wood':0.007,
    'movable_tray_light_wood':0.007,
    'movable_tray_metal_rusty':0.007,
    'movable_tray_metal_shiny':0.007,

}

## Bin_1 boundary
bin_1_x_max = bin_position["bin1"][0] + 0.3
bin_1_x_min = bin_position["bin1"][0] - 0.3
bin_1_y_max = bin_position["bin1"][1] + 0.3
bin_1_y_min = bin_position["bin1"][1] - 0.3
## Bin_4 boundary
bin_4_x_max = bin_position["bin4"][0] + 0.3
bin_4_x_min = bin_position["bin4"][0] - 0.3
bin_4_y_max = bin_position["bin4"][1] + 0.3
bin_4_y_min = bin_position["bin4"][1] - 0.3
                ## Bin_1 boundary
bin_2_x_max = bin_position["bin2"][0] + 0.3
bin_2_x_min = bin_position["bin2"][0] - 0.3
bin_2_y_max = bin_position["bin2"][1] + 0.3
bin_2_y_min = bin_position["bin2"][1] - 0.3
## Bin_4 boundary
bin_3_x_max = bin_position["bin3"][0] + 0.3
bin_3_x_min = bin_position["bin3"][0] - 0.3
bin_3_y_max = bin_position["bin3"][1] + 0.3
bin_3_y_min = bin_position["bin3"][1] - 0.3
                ## Bin_1 boundary
bin_6_x_max = bin_position["bin6"][0] + 0.3
bin_6_x_min = bin_position["bin6"][0] - 0.3
bin_6_y_max = bin_position["bin6"][1] + 0.3
bin_6_y_min = bin_position["bin6"][1] - 0.3
## Bin_4 boundary
bin_7_x_max = bin_position["bin7"][0] + 0.3
bin_7_x_min = bin_position["bin7"][0] - 0.3
bin_7_y_max = bin_position["bin7"][1] + 0.3
bin_7_y_min = bin_position["bin7"][1] - 0.3
                ## Bin_1 boundary
bin_5_x_max = bin_position["bin5"][0] + 0.3
bin_5_x_min = bin_position["bin5"][0] - 0.3
bin_5_y_max = bin_position["bin5"][1] + 0.3
bin_5_y_min = bin_position["bin5"][1] - 0.3
## Bin_4 boundary
bin_8_x_max = bin_position["bin8"][0] + 0.3
bin_8_x_min = bin_position["bin8"][0] - 0.3
bin_8_y_max = bin_position["bin8"][1] + 0.3
bin_8_y_min = bin_position["bin8"][1] - 0.3



agv_boundary = {
    'agv1_ks1_tray': [-2.45, -1.77, 4.44,  4.94],                                                            # ???
    'agv2_ks2_tray': [-2.45, -1.77, 1.12,  1.60],
    'agv3_ks3_tray': [-2.45, -1.77, -1.58, -1.10],
    'agv4_ks4_tray': [-2.45, -1.77, -4.91, -4.43],

    'agv1_as1_tray': [-5.785, -5.105, 4.44,  4.94],
    'agv2_as1_tray': [-5.785, -5.105, 1.12,  1.60],
    'agv3_as3_tray': [-5.785, -5.105, -1.58, -1.10],
    'agv4_as3_tray': [-5.785, -5.105, -4.91, -4.43],

    'agv1_as2_tray': [-10.775, -10.095, 4.44,  4.94],
    'agv2_as2_tray': [-10.775, -10.095, 1.12,  1.60],
    'agv3_as4_tray': [-10.775, -10.095, -1.58, -1.10],
    'agv4_as4_tray': [-10.775, -10.095, -4.91, -4.43],
}
# convey_product_height = {
#         'regulator': 0.875009 + logic_camera_height_error,
#         'pump':  0.874995 + logic_camera_height_error,
#         'battery': 0.875000 + logic_camera_height_error,
#         'sensor': 0.874990 + logic_camera_height_error,
#         'regulator_flipped': 0.944996 + logic_camera_height_error,
#         'pump_flipped':  0.995001 + logic_camera_height_error,
#         'battery_flipped': 0.915000 + logic_camera_height_error,
#         'sensor_flipped': 0.944991 + logic_camera_height_error,
#         }
part_on_conveyor_z = {
    'assembly_regulator_red': 0.875009,     # Z=0.945
    'assembly_regulator_blue': 0.875009,
    'assembly_regulator_green': 0.875009,
    'assembly_regulator_orange': 0.875009,
    'assembly_regulator_purple': 0.875009,

    'assembly_sensor_red': 0.874990,
    'assembly_sensor_blue': 0.874990,
    'assembly_sensor_green': 0.874990,  
    'assembly_sensor_orange': 0.874990,
    'assembly_sensor_purple': 0.874990,

    'assembly_pump_red': 0.874995,
    'assembly_pump_blue':0.874995,
    'assembly_pump_green':0.874995,
    'assembly_pump_orange':0.874995,
    'assembly_pump_purple':0.874995,

    'assembly_battery_red':0.875000,           # Z=0.875
    'assembly_battery_blue':0.875000,
    'assembly_battery_green':0.875000,
    'assembly_battery_orange':0.875000,
    'assembly_battery_purple':0.875000,

}




assembly_sensor_heigh = {
    
    'assembly_sensor_red': [0.032, 0.041],
    'assembly_sensor_blue': [0.032, 0.041],
    'assembly_sensor_green': [0.032, 0.041]

}
part_heigh= {
    'assembly_sensor_red': [0.032, 0.064],
    'assembly_sensor_blue': [0.032, 0.064],
    'assembly_sensor_green': [0.032, 0.064],
    'assembly_regulator_red': [0.032,0.064],
    'assembly_regulator_blue': [0.032,0.064],
    'assembly_regulator_green': [0.032,0.064],
    'assembly_pump_red': [0.056, 0.112],
    'assembly_pump_blue':[0.056, 0.112],
    'assembly_pump_green':[0.056, 0.112],
    'pump': [0.056, 0.112],
    'assembly_battery_red':[0.026, 0.053],
    'assembly_battery_blue':[0.026, 0.053],
    'assembly_battery_green':[0.026, 0.053]
}

part_heigh_R_90 = {
    'assembly_sensor_red': [0.041, 0.1032],
    'assembly_sensor_blue': [0.041, 0.1032],
    'assembly_sensor_green': [0.041, 0.1032]
}


bin_heigh = 0.725
delta_error = 0.05
tray_on_agv_height = 0.78
###############ZT
logic_camera_error = +0.05
logic_camera_height_error = 0.035

bins_size = 0.6

# agv_product_height = {
#         'regulator': 0.770283 + logic_camera_height_error,
#         'pump':  0.890968  + logic_camera_height_error,
#         'battery': 0.770283  + logic_camera_height_error,
#         'sensor': 0.770283 + logic_camera_height_error,
#         'sensor_90': 0.770283  + logic_camera_height_error,
#         }

bins_ks_boundary = {
        'bin1_x': [bin_position["bin1"][0] - bins_size/2 - logic_camera_error,\
                   bin_position["bin1"][0] + bins_size/2 + logic_camera_error],
        'bin1_y': [bin_position["bin1"][1] - bins_size/2 - logic_camera_error,\
                   bin_position["bin1"][1] + bins_size/2 + logic_camera_error],
        'bin2_x': [bin_position["bin2"][0] - bins_size/2 - logic_camera_error,\
                   bin_position["bin2"][0] + bins_size/2 + logic_camera_error],
        'bin2_y': [bin_position["bin2"][1] - bins_size/2 - logic_camera_error,\
                   bin_position["bin2"][1] + bins_size/2 + logic_camera_error],
        'bin3_x': [bin_position["bin3"][0] - bins_size/2 - logic_camera_error,\
                   bin_position["bin3"][0] + bins_size/2 + logic_camera_error],
        'bin3_y': [bin_position["bin3"][1] - bins_size/2 - logic_camera_error,\
                   bin_position["bin3"][1] + bins_size/2 + logic_camera_error],
        'bin4_x': [bin_position["bin4"][0] - bins_size/2 - logic_camera_error,\
                   bin_position["bin4"][0] + bins_size/2 + logic_camera_error],
        'bin4_y': [bin_position["bin4"][1] - bins_size/2 - logic_camera_error,\
                   bin_position["bin4"][1] + bins_size/2 + logic_camera_error],
        'bin5_x': [bin_position["bin5"][0] - bins_size/2 - logic_camera_error,\
                   bin_position["bin5"][0] + bins_size/2 + logic_camera_error],
        'bin5_y': [bin_position["bin5"][1] - bins_size/2 - logic_camera_error,\
                   bin_position["bin5"][1] + bins_size/2 + logic_camera_error],
        'bin6_x': [bin_position["bin6"][0] - bins_size/2 - logic_camera_error,\
                   bin_position["bin6"][0] + bins_size/2 + logic_camera_error],
        'bin6_y': [bin_position["bin6"][1] - bins_size/2 - logic_camera_error,\
                   bin_position["bin6"][1] + bins_size/2 + logic_camera_error],
        'bin7_x': [bin_position["bin7"][0] - bins_size/2 - logic_camera_error,\
                   bin_position["bin7"][0] + bins_size/2 + logic_camera_error],
        'bin7_y': [bin_position["bin7"][1] - bins_size/2 - logic_camera_error,\
                   bin_position["bin7"][1] + bins_size/2 + logic_camera_error],
        'bin8_x': [bin_position["bin8"][0] - bins_size/2 - logic_camera_error,\
                   bin_position["bin8"][0] + bins_size/2 + logic_camera_error],
        'bin8_y': [bin_position["bin8"][1] - bins_size/2 - logic_camera_error,\
                   bin_position["bin8"][1] + bins_size/2 + logic_camera_error],
               }

bins_product_height = {
        'regulator': 0.723480 + logic_camera_height_error,
        'pump':  0.723484 + logic_camera_height_error,
        'battery': 0.723488 + logic_camera_height_error,
        'sensor': 0.723485 + logic_camera_height_error,
        }

bins_product_height_flip = {
        'regulator': 0.793486 + logic_camera_height_error,
        'pump':  0.843483 + logic_camera_height_error,
        'battery': 0.763483 + logic_camera_height_error,
        'sensor': 0.793483 + logic_camera_height_error,
        }

agv_product_height_with_tray = {
        'regulator': 0.770992 + logic_camera_height_error,
        'pump':  0.770993 + logic_camera_height_error,
        'battery': 0.770969 + logic_camera_height_error,
        'sensor': 0.770979 + logic_camera_height_error,
        }

agv_product_height_with_tray_flip = {
        'regulator': 0.840986 + logic_camera_height_error,
        'pump':  0.890971 + logic_camera_height_error,
        'battery': 0.810976 + logic_camera_height_error,
        'sensor': 0.840975 + logic_camera_height_error,
        }

tray_table_position = {
    'tray_table_1' : [-1.300000,-5.840000,0],
    'tray_table_2' : [-1.300000,5.840000,0],

}
tray_table_size = {
    'length_y': 0.68,
    'width_x': 1.34,
}

tray_table_boundary = {
    'tray_table_1_x' : [tray_table_position['tray_table_1'][0] - tray_table_size['width_x']/2 - logic_camera_error,\
                        tray_table_position['tray_table_1'][0] + tray_table_size['width_x']/2 + logic_camera_error],
    'tray_table_1_y' : [tray_table_position['tray_table_1'][1] - tray_table_size['length_y']/2 - logic_camera_error,\
                        tray_table_position['tray_table_1'][1] + tray_table_size['length_y']/2 + logic_camera_error],  

    'tray_table_2_x' : [tray_table_position['tray_table_2'][0] - tray_table_size['width_x']/2 - logic_camera_error,\
                        tray_table_position['tray_table_2'][0] + tray_table_size['width_x']/2 + logic_camera_error],
    'tray_table_2_y' : [tray_table_position['tray_table_2'][1] - tray_table_size['length_y']/2 - logic_camera_error,\
                        tray_table_position['tray_table_2'][1] + tray_table_size['length_y']/2 + logic_camera_error],                   
}

tables_tray_hight = 0.73449 + logic_camera_height_error


#xin
agv_ks_position = {
    'agv1': [-2.069900, 4.800001, 0],
    'agv2': [-2.069900, 1.200001, 0],
    'agv3': [-2.069900, -1.199999, 0],
    'agv4': [-2.069900, -4.799999, 0],
    }



agv_tray_size = {
    'length_y': 0.38,#0.5
    'width_x': 0.52,#0.7
    }

agv_ks_boundary = {
        'agv1_x': [agv_ks_position["agv1"][0] - agv_tray_size["width_x"]/2 - logic_camera_error,\
                   agv_ks_position["agv1"][0] + agv_tray_size["width_x"]/2 + logic_camera_error],
        'agv1_y': [agv_ks_position["agv1"][1] - agv_tray_size["length_y"]/2 - logic_camera_error,\
                   agv_ks_position["agv1"][1] + agv_tray_size["length_y"]/2 + logic_camera_error],
        'agv2_x': [agv_ks_position["agv2"][0] - agv_tray_size["width_x"]/2 - logic_camera_error,\
                   agv_ks_position["agv2"][0] + agv_tray_size["width_x"]/2 + logic_camera_error],
        'agv2_y': [agv_ks_position["agv2"][1] - agv_tray_size["length_y"]/2 - logic_camera_error,\
                   agv_ks_position["agv2"][1] + agv_tray_size["length_y"]/2 + logic_camera_error],
        'agv3_x': [agv_ks_position["agv3"][0] - agv_tray_size["width_x"]/2 - logic_camera_error,\
                   agv_ks_position["agv3"][0] + agv_tray_size["width_x"]/2 + logic_camera_error],
        'agv3_y': [agv_ks_position["agv3"][1] - agv_tray_size["length_y"]/2 - logic_camera_error,\
                   agv_ks_position["agv3"][1] + agv_tray_size["length_y"]/2 + logic_camera_error],
        'agv4_x': [agv_ks_position["agv4"][0] - agv_tray_size["width_x"]/2 - logic_camera_error,\
                   agv_ks_position["agv4"][0] + agv_tray_size["width_x"]/2 + logic_camera_error],
        'agv4_y': [agv_ks_position["agv4"][1] - agv_tray_size["length_y"]/2 - logic_camera_error,\
                   agv_ks_position["agv4"][1] + agv_tray_size["length_y"]/2 + logic_camera_error],
        }

agv_tray_heigh = 0.771000
agv1_kitting_location={
    'agv1_ks1_tray':[-2.070051,4.799979,agv_tray_heigh],
    'agv1_as1_tray':[-7.669939, 4.799979,agv_tray_heigh],#-5.45
    'agv1_as2_tray':[-12.670208,4.799979,agv_tray_heigh],#-10.440274
}
agv2_kitting_location={
    'agv2_ks2_tray':[-2.070051, 1.199981,agv_tray_heigh],
    'agv2_as1_tray':[-7.670095, 1.199981,agv_tray_heigh],
    'agv2_as2_tray':[-12.670141, 1.199981,agv_tray_heigh],
}

agv3_kitting_location={
    'agv3_ks3_tray':[-2.070051,-1.199999,agv_tray_heigh],
    'agv3_as3_tray':[-7.669939,-1.199999,agv_tray_heigh],
    'agv3_as4_tray':[-12.670208,-1.199999,agv_tray_heigh],
}

agv4_kitting_location={
    'agv4_ks4_tray':[-2.070051,-4.799999,agv_tray_heigh],
    'agv4_as3_tray':[-7.669939,-4.799999,agv_tray_heigh],
    'agv4_as4_tray':[-12.670208,-4.799999,agv_tray_heigh],
}

####agv_as_boundary范围不正确####
agv_as_boundary = {
        'agv1_as1_x': [agv1_kitting_location['agv1_as1_tray'][0] - agv_tray_size["width_x"]/2 - logic_camera_error,\
                       agv1_kitting_location['agv1_as1_tray'][0] + agv_tray_size["width_x"]/2 + logic_camera_error],
        'agv1_as1_y': [agv1_kitting_location['agv1_as1_tray'][1] - agv_tray_size["length_y"]/2 - logic_camera_error,\
                       agv1_kitting_location['agv1_as1_tray'][1] + agv_tray_size["length_y"]/2 + logic_camera_error],
        
        'agv1_as2_x': [agv1_kitting_location['agv1_as2_tray'][0] - agv_tray_size["width_x"]/2 - logic_camera_error,\
                       agv1_kitting_location['agv1_as2_tray'][0] + agv_tray_size["width_x"]/2 + logic_camera_error],
        'agv1_as2_y': [agv1_kitting_location['agv1_as2_tray'][1] - agv_tray_size["length_y"]/2 - logic_camera_error,\
                       agv1_kitting_location['agv1_as2_tray'][1] + agv_tray_size["length_y"]/2 + logic_camera_error],
        
        'agv2_as1_x': [agv2_kitting_location['agv2_as1_tray'][0] - agv_tray_size["width_x"]/2 - logic_camera_error,\
                       agv2_kitting_location['agv2_as1_tray'][0] + agv_tray_size["width_x"]/2 + logic_camera_error],
        'agv2_as1_y': [agv2_kitting_location['agv2_as1_tray'][1] - agv_tray_size["length_y"]/2 - logic_camera_error,\
                       agv2_kitting_location['agv2_as1_tray'][1] + agv_tray_size["length_y"]/2 + logic_camera_error],
        
        'agv2_as2_x': [agv2_kitting_location['agv2_as2_tray'][0] - agv_tray_size["width_x"]/2 - logic_camera_error,\
                       agv2_kitting_location['agv2_as2_tray'][0] + agv_tray_size["width_x"]/2 + logic_camera_error],
        'agv2_as2_y': [agv2_kitting_location['agv2_as2_tray'][1] - agv_tray_size["length_y"]/2 - logic_camera_error,\
                       agv2_kitting_location['agv2_as2_tray'][1] + agv_tray_size["length_y"]/2 + logic_camera_error],
        
        'agv3_as3_x': [agv3_kitting_location['agv3_as3_tray'][0] - agv_tray_size["width_x"]/2 - logic_camera_error,\
                       agv3_kitting_location['agv3_as3_tray'][0] + agv_tray_size["width_x"]/2 + logic_camera_error],
        'agv3_as3_y': [agv3_kitting_location['agv3_as3_tray'][1] - agv_tray_size["length_y"]/2 - logic_camera_error,\
                       agv3_kitting_location['agv3_as3_tray'][1] + agv_tray_size["length_y"]/2 + logic_camera_error],
        
        'agv3_as4_x': [agv3_kitting_location['agv3_as4_tray'][0] - agv_tray_size["width_x"]/2 - logic_camera_error,\
                       agv3_kitting_location['agv3_as4_tray'][0] + agv_tray_size["width_x"]/2 + logic_camera_error],
        'agv3_as4_y': [agv3_kitting_location['agv3_as4_tray'][1] - agv_tray_size["length_y"]/2 - logic_camera_error,\
                       agv3_kitting_location['agv3_as4_tray'][1] + agv_tray_size["length_y"]/2 + logic_camera_error],
        
        'agv4_as3_x': [agv4_kitting_location['agv4_as3_tray'][0] - agv_tray_size["width_x"]/2 - logic_camera_error,\
                       agv4_kitting_location['agv4_as3_tray'][0] + agv_tray_size["width_x"]/2 + logic_camera_error],
        'agv4_as3_y': [agv4_kitting_location['agv4_as3_tray'][1] - agv_tray_size["length_y"]/2 - logic_camera_error,\
                       agv4_kitting_location['agv4_as3_tray'][1] + agv_tray_size["length_y"]/2 + logic_camera_error],
        
        'agv4_as4_x': [agv4_kitting_location['agv4_as4_tray'][0] - agv_tray_size["width_x"]/2 - logic_camera_error,\
                       agv4_kitting_location['agv4_as4_tray'][0] + agv_tray_size["width_x"]/2 + logic_camera_error],
        'agv4_as4_y': [agv4_kitting_location['agv4_as4_tray'][1] - agv_tray_size["length_y"]/2 - logic_camera_error,\
                       agv4_kitting_location['agv4_as4_tray'][1] + agv_tray_size["length_y"]/2 + logic_camera_error],      
        }

#convey_position = [-0.573076, 0.00]
convey_position = [-0.6000000, 0.00]

convey_size = {
    # 'length_y': 9 + 2,
    # 'width_x': 0.65,
    'length_y': 9,
    'width_x': 0.42,
    }

convey_boundary ={
        'convey_x': [convey_position[0] - convey_size["width_x"]/2 - logic_camera_error,\
                   convey_position[0] + convey_size["width_x"]/2 + logic_camera_error],
        'convey_y': [convey_position[1] - convey_size["length_y"]/2 - logic_camera_error,\
                   convey_position[1] + convey_size["length_y"]/2 + logic_camera_error],        
        }

convey_product_height = {
        'regulator': 0.875009 + logic_camera_height_error,
        'pump':  0.874995 + logic_camera_height_error,
        'battery': 0.875000 + logic_camera_height_error,
        'sensor': 0.874990 + logic_camera_height_error,      
        }
convey_product_height_flip = {
        'regulator': 0.944996 + logic_camera_height_error,
        'pump':  0.995001 + logic_camera_height_error,
        'battery': 0.915000 + logic_camera_height_error,
        'sensor': 0.944991 + logic_camera_height_error,
        }

#确定零件是否在有效的容器边界内
def Is_In_Effective_Range(model_type, position, x_range, y_range, product_height):
        
    if position.x > x_range[0] and position.x < x_range[1] \
    and position.y > y_range[0] and position.y < y_range[1]:        
        if ("regulator" in model_type  and position.z < product_height['regulator']) \
        or ("pump" in model_type  and position.z < product_height['pump']) \
        or ("battery" in model_type  and position.z < product_height['battery']) \
        or ("sensor" in model_type and position.z < product_height['sensor']) : 
            return True       
    return False

#确定托盘是否在桌子上
def Define_tray_is_in_effective_table_range(model_type, position, x_range, y_range, tables_tray_hight):
        
    if position.x > x_range[0] and position.x < x_range[1] and position.y > y_range[0] and position.y < y_range[1] and position.z <= tables_tray_hight:  
                   
        return True       
    return False


normal_part_move_skew = {
        "x": 0.05,
        "y": 0.05,
        "z": 1,
        }

conveyor_part_move_skew = {
        "x": 0.05,
        "y": 0.05 + conveyor_vel * 0.1,
        "z": 1,
        }
        
def Part_Compare(part_1, part_in_list, part_move_skew=normal_part_move_skew):
    if part_1.type == part_in_list.type \
    and abs(part_1.pose.position.x - part_in_list.pose.position.x) < part_move_skew["x"] \
    and abs(part_1.pose.position.y - part_in_list.pose.position.y) < part_move_skew["y"] \
    and abs(part_1.pose.position.z - part_in_list.pose.position.z) < part_move_skew["z"] :
        return True
        
    return False


######### Part 有关数据 ########
gantry_pick_part_heights_con= { #所有种类的零件高度都经过了测试，wgx 2021-04-17

    'assembly_sensor_red': 0.035,
    'assembly_sensor_blue': 0.035,
    'assembly_sensor_green': 0.035,

    'assembly_regulator_red': 0.0345, 
    'assembly_regulator_blue': 0.0345,
    'assembly_regulator_green': 0.0345,

    'pump': 0.0555,
    'assembly_pump_red': 0.0555,
    'assembly_pump_blue':0.0555,
    'assembly_pump_green':0.0555,

    'assembly_battery_red':0.030, 
    'assembly_battery_blue':0.030, 
    'assembly_battery_green':0.030,
    'can':0.15,
}

gantry_pick_part_heights_bin_agv= {

    'assembly_sensor_red': 0.00,    # 2023.3.18
    'assembly_sensor_blue': 0.00,
    'assembly_sensor_green': 0.00,
    'assembly_sensor_purple': 0.00,
    'assembly_sensor_orange': 0.000,

    'assembly_regulator_red': 0.00, 
    'assembly_regulator_blue': 0.00,
    'assembly_regulator_green': 0.00,
    'assembly_regulator_purple': 0.00,
    'assembly_regulator_orange': 0.00,

    'assembly_pump_red': 0.054-0.0035,
    'assembly_pump_blue':0.054-0.0035,
    'assembly_pump_green':0.054-0.0035,
    'assembly_pump_purple':0.054-0.0035,
    'assembly_pump_orange':0.054-0.0035,
    'pump': 0.054-0.0035,


    'assembly_battery_red':-0.006, #2021-04-15
    'assembly_battery_blue':-0.006, 
    'assembly_battery_green':-0.006,
    'assembly_battery_purple':-0.006,
    'assembly_battery_orange':-0.006,

    'can':0.15,
    'movable_tray_dark_wood':0.007,
    'movable_tray_light_wood':0.007,
    'movable_tray_metal_rusty':0.007,
    'movable_tray_metal_shiny':0.007,

}

kitting_pick_part_heights_con= {

    'assembly_sensor_red': 0.03638,
    'assembly_sensor_blue': 0.03638,
    'assembly_sensor_green': 0.03638,
    'assembly_sensor_orange': 0.03638,
    'assembly_sensor_purple': 0.03638,

    'assembly_regulator_red': 0.036, 
    'assembly_regulator_blue': 0.036,
    'assembly_regulator_green': 0.036,
    'assembly_regulator_orange': 0.036,
    'assembly_regulator_purple': 0.036,

    'assembly_pump_red': 0.0575,
    'assembly_pump_blue':0.0575,
    'assembly_pump_green':0.0575,
    'assembly_regulator_orange': 0.036,
    'assembly_regulator_purple': 0.036,

    'pump': 0.0575,

    'assembly_battery_red':0.0305, 
    'assembly_battery_blue':0.0305, 
    'assembly_battery_green':0.0305,
    'assembly_battery_orange':0.0305, 
    'assembly_battery_purple':0.0305,
    'can':0.15,
}

kitting_part_heights_on_bin_flip = {
    'assembly_sensor_red': 0.07,
    'assembly_sensor_blue': 0.07,
    'assembly_sensor_green': 0.07,
    'assembly_sensor_orange': 0.07,
    'assembly_sensor_purple': 0.07,

    'assembly_regulator_red': 0.07,
    'assembly_regulator_blue': 0.07,
    'assembly_regulator_green': 0.07,
    'assembly_regulator_orange': 0.07,
    'assembly_regulator_purple': 0.07,

    'assembly_pump_red': 0.12,
    'assembly_pump_blue':0.12,
    'assembly_pump_green':0.12,
    'assembly_pump_orange':0.12,
    'assembly_pump_purple':0.12,

    'assembly_battery_red':0.04, 
    'assembly_battery_blue':0.04, 
    'assembly_battery_green':0.04,
    'assembly_battery_orange':0.04, 
    'assembly_battery_purple':0.04,
    'can':0.15,
        'regulator': 0.793486 + logic_camera_height_error,
        'pump':  0.843483 + logic_camera_height_error,
        'battery': 0.763483 + logic_camera_height_error,
        'sensor': 0.793483 + logic_camera_height_error,
        }


kitting_pick_part_heights_on_bin_agv= {

    'assembly_sensor_red': 0.07,
    'assembly_sensor_blue': 0.07,
    'assembly_sensor_green': 0.07,
    'assembly_sensor_orange': 0.07,
    'assembly_sensor_purple': 0.07,

    'assembly_regulator_red': 0.07,
    'assembly_regulator_blue': 0.07,
    'assembly_regulator_green': 0.07,
    'assembly_regulator_orange': 0.07,
    'assembly_regulator_purple': 0.07,

    'pump': 0.12,
    'assembly_pump_red': 0.12,
    'assembly_pump_blue':0.12,
    'assembly_pump_green':0.12,
    'assembly_pump_orange':0.12,
    'assembly_pump_purple':0.12,

    'assembly_battery_red':0.04, 
    'assembly_battery_blue':0.04, 
    'assembly_battery_green':0.04,
    'assembly_battery_orange':0.04, 
    'assembly_battery_purple':0.04,
    'can':0.15,
}

######### Conveyor 有关数据 ################################################
conveyor_vel = 0.20
conveyor_begin = 4.26
conveyor_end = -4.15

kitting_robot_map_dic = {
    'agv1_ks1_tray':[-2.265685, 4.675404, 0],
    'bin1':[-1.898993, 3.379920, 0],
    'bin2':[-1.898993, 2.565006, 0],
    'agv2_ks2_tray':[-2.265685, 1.367643, 0],
    'can': [-2.188252, -0.014119, 0],
    'agv3_ks3_tray':[-2.265685, -1.333917, 0],
    'bin6':[-1.898993, -2.565006, 0],
    'bin5':[-1.898993, -3.379920, 0],
    'agv4_ks4_tray':[-2.265685, -4.696062, 0],
}

gantry_robot_map_dic = {
    'bin1': [-1.898993, 3.379920, 0],
    'bin2': [-1.898993, 2.565006, 0],
    'bin3': [-2.651690, 2.565006, 0],
    'bin4': [-2.651690, 3.379920, 0],
    'bin5': [-1.898993, -3.379920, 0],
    'bin6': [-1.898993, -2.565006, 0],
    'bin7': [-2.651690, -2.565006, 0],
    'bin8': [-2.651690, -3.379920, 0],

    'agv1_ks1_tray': [-2.265685,4.675404,0],
    'agv2_ks2_tray': [-2.265685,1.367643, 0],
    'agv3_ks3_tray': [-2.265685,-1.333917, 0],
    'agv4_ks4_tray': [-2.265685, -4.696062, 0],

    'agv1_as1_tray': [-5.60,    4.675404,0],
    'agv2_as1_tray': [-5.60,    1.367643, 0],
    'agv3_as3_tray': [-5.60,    -1.333917, 0],
    'agv4_as3_tray': [-5.60,    -4.696062, 0],

    'agv1_as2_tray': [-10.590274,    4.675404,0],
    'agv2_as2_tray': [-10.590274,    1.367643, 0],
    'agv3_as4_tray': [-10.590274,    -1.333917, 0],
    'agv4_as4_tray': [-10.590274,    -4.696062, 0],
}


agv_as_location_act_x1_0 = -5.45
agv_as_location_act_x1_1 = -6.115691
agv_as_location_act_x2_0 = -10.440274
agv_as_location_act_x2_1 = -11.115701






# 下面的数据主要用来target_pose_to_world做坐标转换########################################
# 下面的数据测量是tray的中心点位置，不是agv的
agv_tray_heigh = 0.750
agv1_kitting_location={
    'agv1_ks1_tray':[-2.115685,4.675936,agv_tray_heigh],
    'agv1_as1_tray':[-5.45, 4.675936,agv_tray_heigh],#-5.45
    'agv1_as2_tray':[-10.440274,4.675936,agv_tray_heigh],#-10.440274
}
agv2_kitting_location={
    'agv2_ks2_tray':[-2.115685, 1.368175,agv_tray_heigh],
    'agv2_as1_tray':[-5.45, 1.368175,agv_tray_heigh],
    'agv2_as2_tray':[-10.440274, 1.368175,agv_tray_heigh],
}

agv3_kitting_location={
    'agv3_ks3_tray':[-2.115685,-1.333385,agv_tray_heigh],
    'agv3_as3_tray':[-5.45,-1.333385,agv_tray_heigh],
    'agv3_as4_tray':[-10.440274,-1.333385,agv_tray_heigh],
}

agv4_kitting_location={
    'agv4_ks4_tray':[-2.115685,-4.69553,agv_tray_heigh],
    'agv4_as3_tray':[-5.45,-4.69553,agv_tray_heigh],
    'agv4_as4_tray':[-10.440274,-4.69553,agv_tray_heigh],
}


AGV_Kitting_location ={
    'agv1':agv1_kitting_location,
    'agv2':agv2_kitting_location,
    'agv3':agv3_kitting_location,
    'agv4':agv4_kitting_location,
}

AS_AGV = {
    'as1':['agv1','agv2'],
    'as2':['agv1','agv2'],
    'as3':['agv3','agv4'],
    'as4':['agv3','agv4'],
}
AS_AGV_location = {
    'as1':['agv1_as1_tray','agv2_as1_tray'],
    'as2':['agv1_as2_tray','agv2_as2_tray'],
    'as3':['agv3_as3_tray','agv4_as3_tray'],
    'as4':['agv3_as4_tray','agv4_as4_tray'],
}


container_position = {
    'bin1': [-1.898993, 3.379920, 0],
    'bin2': [-1.898993, 2.565006, 0],
    'bin3': [-2.651690, 2.565006, 0],
    'bin4': [-2.651690, 3.379920, 0],
    'bin5': [-1.898993, -3.379920, 0],
    'bin6': [-1.898993, -2.565006, 0],
    'bin7': [-2.651690, -2.565006, 0],
    'bin8': [-2.651690, -3.379920, 0], 
    'agv1_ks1_tray': [-2.115644, 4.675404, 0],
    'agv2_ks2_tray': [-2.115644, 1.367643, 0],
    'agv3_ks3_tray': [-2.115644, -1.333917, 0],
    'agv4_ks4_tray': [-2.115644, -4.696062, 0],
    'agv1_as1_tray': [-5.445, 4.675404, 0],#-5.445
    'agv1_as2_tray': [-10.435, 4.675404, 0],#-10.435
    'agv2_as1_tray': [-5.445, 1.367643, 0],
    'agv2_as2_tray': [-10.435, 1.367643, 0],
    'agv3_as3_tray': [-5.445, -1.333917, 0],
    'agv3_as4_tray': [-10.435, -1.333917, 0],   
    'agv4_as3_tray': [-5.445, -4.696062, 0],
    'agv4_as4_tray': [-10.435, -4.696062, 0],     
}


AGV_Orientation = quaternion_from_euler(0, 0, -pi/2)       


############装配相关数据#############################################

battery_offset_x = 0.003
battery_offset_y = 0.000
# regulator_offset_x = 0.043 
# regulator_offset_y = 0.115
regulator_offset_x = 0.0405#0.044
regulator_offset_y = -0.0025#0.109

sensor_offset_x = -0.1172#-0.11677
#sensor_offset_y = -0.0459
sensor_offset_y = -0.0430
sensor_offset_z = 0.0060
pump_offset_x = 0.003

as1_part = {
    # [x, y, z, r, p, y]
    'assembly_battery_red': [-7.252468+battery_offset_x, 3.270679+battery_offset_y, 1.287989, 0.0,0.0,0.0],
    'assembly_battery_blue':[-7.252468+battery_offset_x, 3.270679+battery_offset_y, 1.287989, 0.0,0.0,0.0],
    'assembly_battery_green':[-7.252468+battery_offset_x, 3.270679+battery_offset_y, 1.287989, 0.0,0.0,0.0],

    'assembly_regulator_red':  [-7.435080+regulator_offset_x, 2.930840+regulator_offset_y, 1.406471, 1.5*pi, pi/2, 1.5*pi],
    'assembly_regulator_blue': [-7.435080+regulator_offset_x, 2.930840+regulator_offset_y, 1.406471, 1.5*pi, pi/2, 1.5*pi],
    'assembly_regulator_green':[-7.435080+regulator_offset_x, 2.930840+regulator_offset_y, 1.406471, 1.5*pi, pi/2, 1.5*pi],
     
    'assembly_sensor_red':[-6.808+sensor_offset_x, 3.2597+sensor_offset_y,1.300+sensor_offset_z,pi/2,0,0],
    'assembly_sensor_blue':[-6.808+sensor_offset_x, 3.2597+sensor_offset_y,1.300+sensor_offset_z,pi/2,0,0],
    'assembly_sensor_green':[-6.808+sensor_offset_x, 3.2597+sensor_offset_y,1.300+sensor_offset_z,pi/2,0,0],

    'assembly_pump_red': [-7.187915+pump_offset_x, 2.943007, 1.274424, 0.0,0.0,0.0],
    'assembly_pump_blue':[-7.187915+pump_offset_x, 2.943007, 1.274424, 0.0,0.0,0.0],
    'assembly_pump_green':[-7.187915+pump_offset_x, 2.943007, 1.274424, 0.0,0.0,0.0],

}

as2_part = {
    # [x, y, z, r, p, y]
    'assembly_battery_red': [-12.252468+battery_offset_x, 3.270679+battery_offset_y, 1.287989, 0.0,0.0,0.0],
    'assembly_battery_blue':[-12.252468+battery_offset_x, 3.270679+battery_offset_y, 1.287989, 0.0,0.0,0.0],
    'assembly_battery_green':[-12.252468+battery_offset_x, 3.270679+battery_offset_y, 1.287989, 0.0,0.0,0.0],

    'assembly_regulator_red':  [-12.435080+regulator_offset_x, 2.930840+regulator_offset_y, 1.406471, 1.5*pi, pi/2, 1.5*pi],
    'assembly_regulator_blue': [-12.435080+regulator_offset_x, 2.930840+regulator_offset_y, 1.406471, 1.5*pi, pi/2, 1.5*pi],
    'assembly_regulator_green':[-12.435080+regulator_offset_x, 2.930840+regulator_offset_y, 1.406471, 1.5*pi, pi/2, 1.5*pi],

    'assembly_sensor_red':[-11.808+sensor_offset_x, 3.2597+sensor_offset_y,1.300+sensor_offset_z,pi/2,0,0],
    'assembly_sensor_blue':[-11.808+sensor_offset_x, 3.2597+sensor_offset_y,1.300+sensor_offset_z,pi/2,0,0],
    'assembly_sensor_green':[-11.808+sensor_offset_x, 3.2597+sensor_offset_y,1.300+sensor_offset_z,pi/2,0,0],

    'assembly_pump_red': [-12.187915+pump_offset_x, 2.943007, 1.274424, 0.0,0.0,0.0],
    'assembly_pump_blue':[-12.187915+pump_offset_x, 2.943007, 1.274424, 0.0,0.0,0.0],
    'assembly_pump_green':[-12.187915+pump_offset_x, 2.943007, 1.274424, 0.0,0.0,0.0],
}

as3_part = {
    # [x, y, z, r, p, y]
    'assembly_battery_red': [-7.252468+battery_offset_x, -2.729321+battery_offset_y, 1.287989, 0.0,0.0,0.0],
    'assembly_battery_blue':[-7.252468+battery_offset_x, -2.729321+battery_offset_y, 1.287989, 0.0,0.0,0.0],
    'assembly_battery_green':[-7.252468+battery_offset_x, -2.729321+battery_offset_y, 1.287989, 0.0,0.0,0.0],

    'assembly_regulator_red':  [-7.435080+regulator_offset_x, -3.06916+regulator_offset_y, 1.406471, 1.5*pi, pi/2, 1.5*pi],
    'assembly_regulator_blue': [-7.435080+regulator_offset_x, -3.06916+regulator_offset_y, 1.406471, 1.5*pi, pi/2, 1.5*pi],
    'assembly_regulator_green':[-7.435080+regulator_offset_x, -3.06916+regulator_offset_y, 1.406471, 1.5*pi, pi/2, 1.5*pi],

    'assembly_sensor_red':[-6.808+sensor_offset_x,  -2.7403+sensor_offset_y,1.300+sensor_offset_z,pi/2,0,0],
    'assembly_sensor_blue':[-6.808+sensor_offset_x, -2.7403+sensor_offset_y,1.300+sensor_offset_z,pi/2,0,0],
    'assembly_sensor_green':[-6.808+sensor_offset_x,-2.7403+sensor_offset_y,1.300+sensor_offset_z,pi/2,0,0],

    'assembly_pump_red': [-7.187915+pump_offset_x, -3.056993, 1.274424, 0.0,0.0,0.0],
    'assembly_pump_blue':[-7.187915+pump_offset_x, -3.056993, 1.274424, 0.0,0.0,0.0],
    'assembly_pump_green':[-7.187915+pump_offset_x, -3.056993, 1.274424, 0.0,0.0,0.0],
}

as4_part = {
    # [x, y, z, r, p, y]
    'assembly_battery_red': [-12.252468+battery_offset_x, -2.729321+battery_offset_y, 1.287989, 0.0,0.0,0.0],
    'assembly_battery_blue':[-12.252468+battery_offset_x, -2.729321+battery_offset_y, 1.287989, 0.0,0.0,0.0],
    'assembly_battery_green':[-12.252468+battery_offset_x, -2.729321+battery_offset_y,1.287989, 0.0,0.0,0.0],

    'assembly_regulator_red':  [-12.435080+regulator_offset_x, -3.06916+regulator_offset_y, 1.406471, 1.5*pi, pi/2, 1.5*pi],
    'assembly_regulator_blue': [-12.435080+regulator_offset_x, -3.06916+regulator_offset_y, 1.406471, 1.5*pi, pi/2, 1.5*pi],
    'assembly_regulator_green':[-12.435080+regulator_offset_x, -3.06916+regulator_offset_y, 1.406471, 1.5*pi, pi/2, 1.5*pi],

    'assembly_sensor_red':[-11.808+sensor_offset_x, -2.7403+sensor_offset_y,1.300+sensor_offset_z,pi/2,0,0],
    'assembly_sensor_blue':[-11.808+sensor_offset_x, -2.7403+sensor_offset_y,1.300+sensor_offset_z,pi/2,0,0],
    'assembly_sensor_green':[-11.808+sensor_offset_x, -2.7403+sensor_offset_y,1.300+sensor_offset_z,pi/2,0,0],

    'assembly_pump_red': [-12.187915+pump_offset_x, -3.056993, 1.274424, 0.0,0.0,0.0],
    'assembly_pump_blue':[-12.187915+pump_offset_x, -3.056993, 1.274424, 0.0,0.0,0.0],
    'assembly_pump_green':[-12.187915+pump_offset_x, -3.056993, 1.274424, 0.0,0.0,0.0],
}


as_data = {
    'as1':as1_part,
    'as2':as2_part,
    'as3':as3_part,
    'as4':as4_part,
}


delta_x = 2.05
delta_y = 0.05
as_location = {
    'as1': [-7.3+delta_x,   3.0, pi/2],
    'as2': [-12.3+delta_x,  3.0, pi/2],
    'as3': [-7.3+delta_x,  -3.0, pi/2],
    'as4': [-12.3+delta_x, -3.0, pi/2],
}

delta_x_regul = 1.64#0.35
delta_y_regul = -0.35#1.40
regulator_as_location = {
    'as1': [-7.3+ delta_x_regul,   3.0+delta_y_regul, pi/2],#pi
    'as2': [-12.3+delta_x_regul,  3.0+delta_y_regul, pi/2],
    'as3': [-7.3+ delta_x_regul,  -3.0+delta_y_regul, pi/2],
    'as4': [-12.3+delta_x_regul, -3.0+delta_y_regul, pi/2],
}

gantry_assembly_offset_y = -0.1543
gantry_assembly_offset_x = 1.15

bins = ['bin1', 'bin2','bin6','bin5','bin4','bin3','bin7','bin8']

kitting_bins = ['bin1', 'bin2','bin6','bin5']

bin5_safe_zone = ['bin1', 'bin2','agv1_ks1_tray','agv2_ks2_tray','agv3_ks3_tray','can']

bin6_safe_zone = ['bin1', 'bin2','agv1_ks1_tray','agv2_ks2_tray','agv3_ks3_tray','agv4_ks4_tray','can']

can_safe_zone =['bin1', 'bin2','bin5', 'bin6','agv1_ks1_tray','agv2_ks2_tray','agv4_ks4_tray']

bin2_safe_zone = ['bin5','bin6','agv1_ks1_tray','agv2_ks2_tray','agv3_ks3_tray','agv4_ks4_tray','can']

bin1_safe_zone = ['bin5','bin6','agv1_ks1_tray','agv2_ks2_tray','agv3_ks3_tray','agv4_ks4_tray','can']


agv_ks_location={
    "agv1":"agv1_ks1_tray",
    "agv2":"agv2_ks2_tray",
    "agv3":"agv3_ks3_tray",
    "agv4":"agv4_ks4_tray",
}

agv_as_location_act_x1_0 = -5.45
agv_as_location_act_x1_1 = -6.115691
agv_as_location_act_x2_0 = -10.440274
agv_as_location_act_x2_1 = -11.115701

    ######### Gantry_Robot 有关数据 ########################################
gantry_velocity = 0.8

gantry_angle_velocity = 2.00
# A_g是修正矩阵
A_g = numpy.matrix([[ 1.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                [0.00000000e+00, 1.00000000e+00, 0.00000000e+00,0.00000000e+00],
                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00],
                [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.000000000000]])


bin_location = {
    'bin1': [-1.898993, 3.379920, 0],
    'bin2': [-1.898993, 2.565006, 0],
    'bin3': [-2.651690, 2.565006, 0],
    'bin4': [-2.651690, 3.379920, 0],
    'bin5': [-1.898993, -3.379920, 0],
    'bin6': [-1.898993, -2.565006, 0],
    'bin7': [-2.651690, -2.565006, 0],
    'bin8': [-2.651690, -3.379920, 0],
}

agv_move=0.3
agv_loaction={
    
    'agv1_ks1_tray': [-2.065685,4.795404,0],     # 位置向前移动move=0.3
    'agv2_ks2_tray': [-2.065685,1.207643, 0],
    'agv3_ks3_tray': [-2.065685,-1.203917, 0],
    'agv4_ks4_tray': [-2.05685, -4.796062, 0],

    'agv1_as1_tray': [-7.67+agv_move,4.795404,0],
    'agv2_as1_tray': [-7.67+agv_move,1.207643, 0],
    'agv3_as3_tray': [-7.67+agv_move,-1.203917, 0],
    'agv4_as3_tray': [-7.67+agv_move, -4.796062, 0],

    'agv1_as2_tray': [-12.665274+agv_move,4.795404,0],
    'agv2_as2_tray': [-12.665274+agv_move,1.207643, 0],
    'agv3_as4_tray': [-12.665274+agv_move,-1.203917, 0],
    'agv4_as4_tray': [-12.665274+agv_move, -4.796062, 0],

}


bin_offset_x = -1.898993 +2.85-7
agv_offset_x = 0.75 -6.5#0.8
agv_offset_y = 0.23 # 0.23
agv_offset_x_ = -0.95 -6.5   # 原来是
agv_offset_y_ = 0.25 
as_offset_x = 2.05 +5.8

gantry_robot_park_location ={

    'as1': [-7.3+as_offset_x,   3.0, pi/2],
    'as2': [-12.3+as_offset_x,  3.0, pi/2],
    'as3': [-7.3+as_offset_x,  -3.0, pi/2],
    'as4': [-12.3+as_offset_x, -3.0, pi/2],

    'agv1_ks1_tray':[agv_loaction['agv1_ks1_tray'][0] - agv_offset_x+0.08, agv_loaction['agv1_ks1_tray'][1]-agv_offset_y,  -pi/2],
    'agv2_ks2_tray':[agv_loaction['agv2_ks2_tray'][0] - agv_offset_x, agv_loaction['agv2_ks2_tray'][1]-agv_offset_y,  -pi/2], 
    'agv3_ks3_tray':[agv_loaction['agv3_ks3_tray'][0] - agv_offset_x, agv_loaction['agv3_ks3_tray'][1]+agv_offset_y,  -pi/2], 
    'agv4_ks4_tray':[agv_loaction['agv4_ks4_tray'][0] - agv_offset_x+0.08, agv_loaction['agv4_ks4_tray'][1]+agv_offset_y,  -pi/2], 

    'agv1_as1_tray':[agv_loaction['agv1_as1_tray'][0] - agv_offset_x_, agv_loaction['agv1_as1_tray'][1]-agv_offset_y_, pi/2],
    'agv2_as1_tray':[agv_loaction['agv2_as1_tray'][0] - agv_offset_x_, agv_loaction['agv2_as1_tray'][1], pi/2], 
    'agv3_as3_tray':[agv_loaction['agv3_as3_tray'][0] - agv_offset_x_, agv_loaction['agv3_as3_tray'][1], pi/2], 
    'agv4_as3_tray':[agv_loaction['agv4_as3_tray'][0] - agv_offset_x_, agv_loaction['agv4_as3_tray'][1]+agv_offset_y_, pi/2],

    'agv1_as2_tray':[agv_loaction['agv1_as2_tray'][0] - agv_offset_x_, agv_loaction['agv1_as2_tray'][1]-agv_offset_y_, pi/2],
    'agv2_as2_tray':[agv_loaction['agv2_as2_tray'][0] - agv_offset_x_, agv_loaction['agv2_as2_tray'][1], pi/2], 
    'agv3_as4_tray':[agv_loaction['agv3_as4_tray'][0] - agv_offset_x_, agv_loaction['agv3_as4_tray'][1], pi/2], 
    'agv4_as4_tray':[agv_loaction['agv4_as4_tray'][0] - agv_offset_x_, agv_loaction['agv4_as4_tray'][1]+agv_offset_y_, pi/2], 

    # 位置有问题，我自己写新的  2023
    # 'agv1_as1':[agv_loaction['agv1_as1_tray'][0] , agv_loaction['agv1_as1_tray'][1], pi/2],
    # 'agv2_as1':[agv_loaction['agv2_as1_tray'][0] , agv_loaction['agv2_as1_tray'][1], pi/2], 

    'bin1': [-1.898993 - bin_offset_x, 3.379920, -pi/2],
    'bin2': [-1.898993 - bin_offset_x, 2.565006, -pi/2],
    'bin3': [-2.651690 - bin_offset_x, 2.565006, -pi/2],
    'bin4': [-2.651690 - bin_offset_x, 3.379920, -pi/2],
    'bin5': [-1.898993 - bin_offset_x, -3.379920, -pi/2],
    'bin6': [-1.898993 - bin_offset_x, -2.565006, -pi/2],
    'bin7': [-2.651690 - bin_offset_x, -2.565006, -pi/2],
    'bin8': [-2.651690+0.2 - bin_offset_x, -3.379920-0.3, -pi/2],

    'can':[-2.186829 -0.5- bin_offset_x,0.150000,-pi/2],
    "init_1":[-4.55,0.00, 0],
    "init_2":[-9.54,0.00, pi/2], 
    'gripper_cs' : [-3.90, 6.20, -pi/2],
    'tray_table' : [-6.09, 5.50, -pi/2],
    'kts1':[-2.265685- agv_offset_x+0.08+2,-5.9,0.00],
    'kts2':[-2.265685- agv_offset_x+0.08+2,6.089,-pi]
}


######### Kitting_Robot 有关数据 ################################

kitting_robot_park_location = {
    'agv1_ks1_tray':[-2.06, 4.8, 0.76],
    'bin1':[-1.30, 3.379920, 0],
    'bin2':[-1.30, 2.425006, 0],#'bin2':[-1.30, 2.565006, 0]
    'agv2_ks2_tray':[-2.06, 1.2, 0.76],
    'can': [-1.30, -0.014119, 0],
    'agv3_ks3_tray':[-2.06, -1.2, 0.76],
    'bin6':[-1.30, -2.565006, 0],
    'bin5':[-1.30, -3.379920, 0],
    'agv4_ks4_tray':[-2.06, -4.8, 0.76],
    'kts1':[-1.30,-4.789,0],
    'kts2':[-1.30,4.789,0],

}



quad_offsets_ = {
KittingPart.QUADRANT1: (0.10, 0.12),
KittingPart.QUADRANT2: (0.10, -0.12),
KittingPart.QUADRANT3: (-0.10, 0.12),
KittingPart.QUADRANT4: (-0.10, -0.12),
}

kitting_velocity = 0.8  #
kitting_angle_velocity = 2.00
controller_respond_time = 0.1
ASEND = 0.01 #机器人的指令周期
vacuum_gripper_height = 0.01
#kitting_robot一次指令最大可滑行距离
kitting_robot_slide_throld = 10.50

# A_m是修正矩阵
A_k = numpy.matrix([[-1.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                    [0.00000000e+00, -1.00000000e+00, 0.00000000e+00,0.00000000e+00],
                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00],
                    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.000000000000]])






######################################################################
class Robot_Info:
    '''
    work_state: assenbly: standby, move_to_agv, has_grasped, move_to_as, has_assembly
                kitting  floor: standby, move_to_bin, has_grasped, move_to_agv, has_placed
    '''
    def __init__(self,robot_name):
        self.name = robot_name
        self.position = Point()
        self.location = None
        self.next_park_location = None
        self.is_enabled = True
        self.is_idle = True
        self.work_state= "standby"
        self.has_been_disabled = False
        self.is_alive = True #place 避障时用
        self.pick_part = None





def limit_joint(angle, threshold):
    while angle>=threshold:
        angle = angle-threshold
    while angle<=-threshold:
        angle = angle+threshold
    return angle

def part_pose_after_flip(part):
    rpy =  euler_from_quaternion([part.pose.orientation.x,part.pose.orientation.y,\
            part.pose.orientation.z,part.pose.orientation.w])
    flip_rpy = [rpy[0]+pi,rpy[1],rpy[2]-pi]
    q = quaternion_from_euler(flip_rpy[0],flip_rpy[1],flip_rpy[2],"sxyz")
    part.pose.orientation.x = q[0]
    part.pose.orientation.y = q[1]
    part.pose.orientation.z = q[2]
    part.pose.orientation.w = q[3]
    part.pose.position.z = 0.779888
    return part
def kitting_close_park_location(position_y):
    l1 = abs(position_y -kitting_robot_park_location['agv1_ks1_tray'][1])
    l2 = abs(position_y -kitting_robot_park_location['bin1'][1])
    l3 = abs(position_y -kitting_robot_park_location['bin2'][1])
    l4 = abs(position_y -kitting_robot_park_location['agv2_ks2_tray'][1])
    l5 = abs(position_y -kitting_robot_park_location['can'][1])
    l6 = abs(position_y -kitting_robot_park_location['agv3_ks3_tray'][1])
    l7 = abs(position_y -kitting_robot_park_location['bin6'][1])
    l8 = abs(position_y -kitting_robot_park_location['bin5'][1])
    l9 = abs(position_y -kitting_robot_park_location['agv4_ks4_tray'][1])
    length = [l1,l2,l3,l4,l5,l6,l7,l8,l9]
    locations = ['agv1_ks1_tray','bin1','bin2','agv2_ks2_tray','can','agv3_ks3_tray','bin6','bin5','agv4_ks4_tray']
    short =length.index(min(length))
    location = locations[short]
    return location

def gantry_location_near_ks(ks_location):
    if ks_location == 'agv1_ks1_tray':
        return "bin4"
    if ks_location == "agv2_ks2_tray":
        return "can"
    if ks_location == "agv3_ks3_tray":
        return "can"
    if ks_location == "agv4_ks4_tray":
        return "bin8"

#决定箱子上是否需要翻转
def determine_part_is_need_flip(part_color_type,part_pose):
    
    if 'sensor' in part_color_type and part_pose.position.z>0.76:
        return True
    if 'regulator' in part_color_type and part_pose.position.z>0.76:
        return True
    if 'pump' in part_color_type and part_pose.position.z>0.76:
        return True
    if 'battery' in part_color_type and part_pose.position.z>0.75:
        return True
    
    return False
#决定传送带上是否需要翻转
def determine_part_is_need_flip_on_convey(part_color_type,part_pose):
    part_pose.position.z
    if 'sensor' in part_color_type and part_pose.position.z>0.91:
        return True
    if 'regulator' in part_color_type and part_pose.position.z>0.91:
        return True
    if 'pump' in part_color_type and part_pose.position.z>0.95:
        return True
    if 'battery' in part_color_type and part_pose.position.z>0.90:
        return True
    
    return False

def determine_part_name(part_type,part_color):
        part_color_type=None
        if part_color ==0:
            if part_type == 10:
                part_color_type="assembly_battery_red"
                #print("part_color_type",part_color_type)
            elif part_type==11:
                part_color_type="assembly_pump_red"
                #print("part_color_type",part_color_type)
            elif part_type==12:
                part_color_type="assembly_sensor_red"
                #print("part_color_type",part_color_type)
            elif part_type==13:
                part_color_type="assembly_regulator_red"
                #print("part_color_type",part_color_type)
        elif  part_color ==1:
            if part_type == 10:
                part_color_type="assembly_battery_green"
                #print("part_color_type",part_color_type)
            elif part_type==11:
                part_color_type="assembly_pump_green"
                #print("part_color_type",part_color_type)
            elif part_type==12:
                part_color_type="assembly_sensor_green"
                #print("part_color_type",part_color_type)
            elif part_type==13:
                part_color_type="assembly_regulator_green"
                #print("part_color_type",part_color_type)
        elif  part_color ==2:
            if part_type == 10:
                part_color_type="assembly_battery_blue"
                #print("part_color_type",part_color_type)
            elif part_type==11:
                part_color_type="assembly_pump_blue"
                #print("part_color_type",part_color_type)
            elif part_type==12:
                part_color_type="assembly_sensor_blue"
                #print("part_color_type",part_color_type)
            elif part_type==13:
                part_color_type="assembly_regulator_blue"
                #print("part_color_type",part_color_type)
        elif  part_color ==3:
            if part_type == 10:
                part_color_type="assembly_battery_orange"
                #print("part_color_type",part_color_type)
            elif part_type==11:
                part_color_type="assembly_pump_orange"
                #print("part_color_type",part_color_type)
            elif part_type==12:
                part_color_type="assembly_sensor_orange"
                #print("part_color_type",part_color_type)
            elif part_type==13:
                part_color_type="assembly_regulator_orange"
                #print("part_color_type",part_color_type)
        elif  part_color ==4:
            if part_type == 10:
                part_color_type="assembly_battery_purple"
                #print("part_color_type",part_color_type)
            elif part_type==11:
                part_color_type="assembly_pump_purple"
                #print("part_color_type",part_color_type)
            elif part_type==12:
                part_color_type="assembly_sensor_purple"
                #print("part_color_type",part_color_type)
            elif part_type==13:
                part_color_type="assembly_regulator_purple"
                #print("part_color_type",part_color_type)
        return part_color_type

def build_pose(x, y, z):
    pose = Pose()
    pose.position.x = x
    pose.position.y = y
    pose.position.z = z
    pose.orientation.x=0.0
    pose.orientation.y=0.0
    pose.orientation.z=0.0
    pose.orientation.w=1.0
    return pose

def frame_world_pose(frame_id):
    pose = Pose()
    pose.position.x = kitting_robot_park_location[frame_id][0]
    pose.position.y = kitting_robot_park_location[frame_id][1]
    pose.position.z = kitting_robot_park_location[frame_id][2]
    pose.orientation.x=0.0
    pose.orientation.y=0.0
    pose.orientation.z=0.0
    pose.orientation.w=1.0
    return pose

def find_closest_location(x, y):     # 确定gantry做assembly的停靠点
    closest_location = None
    closest_distance = float('inf')

    for location, coordinates in gantry_robot_park_location.items():
        distance = sqrt((x - coordinates[0]) ** 2 + (y - coordinates[1]) ** 2)
        if distance < closest_distance:
            closest_distance = distance
            closest_location = location

    return closest_location

def find_closest_agv(obj_y):

    closest_agv = None
    min_distance = float('inf')

    for agv, position in agv_ks_position.items():
        distance = abs(obj_y - position[1])
        if distance < min_distance:
            min_distance = distance
            closest_agv = agv

    return int(closest_agv[3:])  # 提取AGV编号并将其转换为整数


def find_closest_bin(agv_num):
    agv_key = f'agv{agv_num}'
    if agv_key not in agv_ks_position:
        raise ValueError(f"Invalid AGV number: {agv_num}")

    agv_y = agv_ks_position[agv_key][1]
    bin1_y = bin_position['bin1'][1]
    bin6_y = bin_position['bin6'][1]

    if abs(agv_y - bin1_y) < abs(agv_y - bin6_y):
        return 'bin1'
    else:
        return 'bin6'
    
def find_closest_bin_ceiling(agv_num):
    agv_key = f'agv{agv_num}'
    if agv_key not in agv_ks_position:
        raise ValueError(f"Invalid AGV number: {agv_num}")

    agv_y = agv_ks_position[agv_key][1]
    bin3_y = bin_position['bin3'][1]
    bin8_y = bin_position['bin8'][1]

    if abs(agv_y - bin3_y) < abs(agv_y - bin8_y):
        return 'bin3'
    else:
        return 'bin8'

def is_close_bin1(x, y):
    point_x, point_y = -1.90, 3.55
    threshold = 0.05

    distance = math.sqrt((x - point_x) ** 2 + (y - point_y) ** 2)
    return distance < threshold

def is_close_bin6(x, y):
    point_x, point_y = -1.90, -2.445
    threshold = 0.05

    distance = math.sqrt((x - point_x) ** 2 + (y - point_y) ** 2)
    return distance < threshold

def custom_mapping(input_value):  # 识别tray——num
    if input_value >= 0 and input_value <= 5:
        return input_value
    elif input_value == 9:
        return 6
    elif input_value == 8:
        return 7
    elif input_value == 7:
        return 8
    elif input_value == 6:
        return 9
    else:
        return None  # 如果输入值无效，返回 None
    
tray_slots_location={
    "slot1": [-0.87,-5.84,0.735],
    "slot2": [-1.30,-5.84,0.735],
    "slot3": [-1.73,-5.84,0.735],
    "slot4": [-1.73,5.84,0.735],
    "slot5": [-1.30,5.84,0.735],
    "slot6": [-0.87,5.84,0.735]
}

# def quaternion_multiply(q0, q1):
#     """
#     Multiplies two quaternions.

#     Input
#     :param q0: A 4 element array containing the first quaternion (q01, q11, q21, q31)
#     :param q1: A 4 element array containing the second quaternion (q02, q12, q22, q32)

#     Output
#     :return: A 4 element array containing the final quaternion (q03,q13,q23,q33)

#     """
#     # Extract the values from q0
#     w0 = q0[0]
#     x0 = q0[1]
#     y0 = q0[2]
#     z0 = q0[3]

#     # Extract the values from q1
#     w1 = q1[0]
#     x1 = q1[1]
#     y1 = q1[2]
#     z1 = q1[3]

#     # Computer the product of the two quaternions, term by term
#     q0q1_w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
#     q0q1_x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
#     q0q1_y = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
#     q0q1_z = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1

#     # Create a 4 element array containing the final quaternion
#     final_quaternion =numpy.array([q0q1_w, q0q1_x, q0q1_y, q0q1_z])

#     # Return a 4 element array containing the final quaternion (q02,q12,q22,q32)
#     return final_quaternion

