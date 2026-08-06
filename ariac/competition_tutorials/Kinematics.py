#!/usr/bin/python2.7
#coding:utf-8


import numpy
import time
from math import cos 
from math import sin 
from math import atan2 
from math import acos
from math import asin 
from math import sqrt
from math import pi
import copy

from tf2_ros import *
from competition_tutorials.my_tf import *
import geometry_msgs
from geometry_msgs.msg import *


from geometry_msgs.msg import Pose
from tf2_geometry_msgs import PoseStamped, from_msg_msg, to_msg_msg
from competition_tutorials.data import *

def MultiplyPose(self, p1: Pose, p2: Pose) -> Pose:
    f1 = from_msg_msg(PoseStamped(header=p1.header, pose=p1))
    f2 = from_msg_msg(PoseStamped(header=p2.header, pose=p2))
    f3 = f1 * f2
    return to_msg_msg(f3.p)  # 只返回位置信息，不包括方向




def GetYaw(orientation):
    q = (orientation.x, orientation.y, orientation.z, orientation.w)
    roll, pitch, yaw = euler_from_quaternion(q)
    return yaw

def QuaternionFromRPY( r, p, y):
    q = quaternion_from_euler(r, p, y)
    q_msg = Quaternion()
    q_msg.x = q[0]
    q_msg.y = q[1]
    q_msg.z = q[2]
    q_msg.w = q[3]

    return q_msg

def target_pose_to_world(part,agv_id, agv_present_location):
    '''
    订单中的零件位置转换成，要放置在agv托盘上的世界坐标位置
    '''
    new_part = copy.deepcopy(part)
    new_part.pose.position.x = AGV_Kitting_location[agv_id][agv_present_location][0] + part.pose.position.y
    new_part.pose.position.y = AGV_Kitting_location[agv_id][agv_present_location][1] - part.pose.position.x
    #   part_roll = atan2(2 * (part.pose.orientation.x * part.pose.orientation.w + part.pose.orientation.y * part.pose.orientation.z), \
    #                   1.0 - 2 * (part.pose.orientation.x ** 2 + part.pose.orientation.y **2))
    
    new_part.pose.position.z = AGV_Kitting_location[agv_id][agv_present_location][2] + kitting_pick_part_heights_on_bin_agv[part.type]
    p = euler_from_quaternion([part.pose.orientation.x,part.pose.orientation.y,part.pose.orientation.z,part.pose.orientation.w])

    if abs(abs(p[0]) - pi)<=0.3 and ("pump" in part.type):
        new_part.is_flip = True
    q = quaternion_from_euler(p[0],p[1],p[2],"sxyz")
    part.pose.orientation.x = q[0]
    part.pose.orientation.y = q[1]
    part.pose.orientation.z = q[2]
    part.pose.orientation.w = q[3]

    q = quaternion_multiply(AGV_Orientation,  \
                    [part.pose.orientation.x,part.pose.orientation.y,part.pose.orientation.z,part.pose.orientation.w])
    new_part.pose.orientation.x = q[0]
    new_part.pose.orientation.y = q[1]
    new_part.pose.orientation.z = q[2]
    new_part.pose.orientation.w = q[3]
    
    return new_part

def angle_limit(angle):
    angle = angle%(2*pi)
    return angle

def rpy_check(rpy, rpy_1):
    error_0 = abs((angle_limit(rpy[0]) - angle_limit(rpy_1[0])))
    error_1 = abs((angle_limit(rpy[1]) - angle_limit(rpy_1[1])))
    error_2 = abs((angle_limit(rpy[2]) - angle_limit(rpy_1[2])))



    if (error_0 <= 0.1 or error_0 >= 2*pi-0.1) and (error_1 <= 0.1 or error_1 >= 2*pi-0.1)\
        and (error_2 <= 0.1 or error_2 >= 2*pi-0.1):
        return True
    else:
        return False

max_delta = 0.00
def xyz_check(part_1, part_2):
    if  abs(part_1.pose.position.x - part_2.pose.position.x) < 0.03 + max_delta \
    and abs(part_1.pose.position.y - part_2.pose.position.y) < 0.03 + max_delta \
    and abs(part_1.pose.position.z - part_2.pose.position.z) < 0.03 + max_delta :
        return True
    else:
        return False

#UR10 parmas
d1 =  0.1807
a2 = -0.6127
a3 = -0.57155
d4 =  0.17415
d5 =  0.11985
d6 =  0.11655
PI = pi
ZERO_THRESH = 0.00000001

# q = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']


def Forward(q,A_x):
    
    T=[0.00]*16
    #00
    s1 = math.sin(q[0])
    c1 = math.cos(q[0])
    ##01
    q23 = q[1]
    q234 = q[1]
    s2 = math.sin(q[1])
    c2 = math.cos(q[1])
    ##02
    s3 = math.sin(q[2])
    c3 = math.cos(q[2])
    q23 += q[2]
    q234 += q[2]
    ##03
    s4 = math.sin(q[3])
    c4 = math.cos(q[3])
    q234 += q[3]
    ##04
    s5 = math.sin(q[4])
    c5 = math.cos(q[4])
    ##05
    s6 = math.sin(q[5])
    c6 = math.cos(q[5])
    s23 = math.sin(q23)
    c23 = math.cos(q23)
    s234 = math.sin(q234)
    c234 = math.cos(q234)
    #4*4 roate arrary
    #00 parameter
    T[0] = c234*c1*s5 - c5*s1
    # 01 parameter
    T[1] = c6*(s1*s5 + c234*c1*c5) - s234*c1*s6
    # 02 parameter
    T[2] = -s6*(s1*s5 + c234*c1*c5) - s234*c1*c6
    # 03 parameter
    T[3] = d6*c234*c1*s5 - a3*c23*c1 - a2*c1*c2 - d6*c5*s1 - d5*s234*c1 - d4*s1
    # 04 parameter
    T[4] = c1*c5 + c234*s1*s5
    # 05 parameter
    T[5] = -c6*(c1*s5 - c234*c5*s1) - s234*s1*s6

    # 06 parameter
    T[6] = s6*(c1*s5 - c234*c5*s1) - s234*c6*s1
    # 07 parameter
    T[7] = d6*(c1*c5 + c234*s1*s5) + d4*c1 - a3*c23*s1 - a2*c2*s1 - d5*s234*s1
    # 08 parameter
    T[8] = -s234*s5
    # 09 parameter
    T[9] = -c234*s6 - s234*c5*c6
    # 10 parameter
    T[10] = s234*c5*s6 - c234*c6
    # 11 parameter
    T[11] = d1 + a3*s23 + a2*s2 - d5*(c23*c4 - s23*s4) - d6*s5*(c23*s4 + s23*c4)
    # 12 parameter
    T[12] = 0.0
    # 13 parameter
    T[13] = 0.0
    # 14 parameter
    T[14] = 0.0
    # 15 parameter
    T[15] = 1.0
    T_ = List2matrix(T)

    return A_x*T_

def SIGN(x):
    return (x>0)-(x<0)
# T ------齐次矩阵 list的形式； q6_des ----wrist_3_link的角度值，默认为0
def Iverse(T,q6_des = 0.00):


    q_sols=[0.00]*48
    num_sols = 0
    T02 = - T[0]
    T00 = T[1]
    T01 = T[2]
    T03 = -T[3]
    T12 = - T[4]
    T10 = T[5]
    T11 = T[6]
    T13 = - T[7]
    T22 = T[8]
    T20 = - T[9]
    T21 = - T[10]
    T23 = T[11]

    #####shoulder rotate  joint(q1) ###########################
    q1=[0,0]
    A = d6 * T12 - T13
    B = d6 * T02 - T03
    R = A * A + B * B
    if math.fabs(A) < ZERO_THRESH:
        if math.fabs(math.fabs(d4) - math.fabs(B)) < ZERO_THRESH:
            div = -SIGN(d4) * SIGN(B)
            print("div d4",div)
        else:
            div = -d4 / B
            print("div ",div)
        
        arcsin = math.asin(div)
        if math.fabs(arcsin) < ZERO_THRESH:
            arcsin = 0.0
        if arcsin < 0.0:
            q1[0] = arcsin + 2.0 * PI
        else:
            q1[0] = arcsin
            q1[1] = PI - arcsin
    elif math.fabs(B) < ZERO_THRESH:
        if math.fabs(math.fabs(d4) - math.fabs(A)) < ZERO_THRESH:
            div = SIGN(d4) * SIGN(A)
        else:
            div = d4 / A
        arccos = math.acos(div)
        q1[0] = arccos
        q1[1] = 2.0 * PI - arccos
    elif d4 * d4 > R:
        return num_sols
    else:
        arccos = math.acos(d4 / math.sqrt(R))
        arctan = math.atan2(-B, A)
        pos = arccos + arctan
        neg = -arccos + arctan
        if math.fabs(pos) < ZERO_THRESH:
            pos = 0.0
        if math.fabs(neg) < ZERO_THRESH:
            neg = 0.0
        if pos >= 0.0:
            q1[0] = pos
        else:
            q1[0] = 2.0 * PI + pos
        if neg >= 0.0:
            q1[1] = neg
        else:
            q1[1] = 2.0 * PI + neg

    ###### wrist2  joint(q5) ##########################
    q5=numpy.zeros((2,2))#define 2*2 q5 array

    for i in range(2):
        numer = (T03 * math.sin(q1[i]) - T13 * math.cos(q1[i]) - d4)
        if math.fabs(math.fabs(numer) - math.fabs(d6)) < ZERO_THRESH:
            div = SIGN(numer) * SIGN(d6)
        else:
            div = numer / d6
        arccos = math.acos(div)
        q5[i][0] = arccos
        q5[i][1] = 2.0 * PI - arccos
    #############################################################
    for i in range(2):
        for j in range(2):
            c1 = math.cos(q1[i])
            s1 = math.sin(q1[i])
            c5 = math.cos(q5[i][j])
            s5 = math.sin(q5[i][j])
            ######################## wrist 3 joint (q6) ################################
            if math.fabs(s5) < ZERO_THRESH:
                q6 = q6_des
            else:
                q6 = math.atan2(SIGN(s5) * -(T01 * s1 - T11 * c1),SIGN(s5) * (T00 * s1 - T10 * c1))
                if math.fabs(q6) < ZERO_THRESH:
                    q6 = 0.0
                if (q6 < 0.0):
                    q6 += 2.0 * PI
            q2=[0.00,0.00]
            q3=[0.00,0.00]
            q4=[0.00,0.00]
            #####################RRR joints (q2, q3, q4) ################################
            c6 = math.cos(q6)
            s6 = math.sin(q6)
            x04x = -s5 * (T02 * c1 + T12 * s1) - c5 * (s6 * (T01 * c1 + T11 * s1) - c6 * (T00 * c1 + T10 * s1))
            x04y = c5 * (T20 * c6 - T21 * s6) - T22 * s5
            p13x = d5 * (s6 * (T00 * c1 + T10 * s1) + c6 * (T01 * c1 + T11 * s1)) - d6 * (T02 * c1 + T12 * s1) +T03 * c1 + T13 * s1
            p13y = T23 - d1 - d6 * T22 + d5 * (T21 * c6 + T20 * s6)
            c3 = (p13x * p13x + p13y * p13y - a2 * a2 - a3 * a3) / (2.0 * a2 * a3)
            if math.fabs(math.fabs(c3) - 1.0) < ZERO_THRESH:
                c3 = SIGN(c3)
            elif math.fabs(c3) > 1.0:
                # TODO NO SOLUTION
                continue
            arccos = math.acos(c3)
            q3[0] = arccos
            q3[1] = 2.0 * PI - arccos
            denom = a2 * a2 + a3 * a3 + 2 * a2 * a3 * c3
            s3 = math.sin(arccos)
            A = (a2 + a3 * c3)
            B = a3 * s3
            q2[0] = math.atan2((A * p13y - B * p13x) / denom, (A * p13x + B * p13y) / denom)
            q2[1] = math.atan2((A * p13y + B * p13x) / denom, (A * p13x - B * p13y) / denom)
            c23_0 = math.cos(q2[0] + q3[0])
            s23_0 = math.sin(q2[0] + q3[0])
            c23_1 = math.cos(q2[1] + q3[1])
            s23_1 = math.sin(q2[1] + q3[1])
            q4[0] = math.atan2(c23_0 * x04y - s23_0 * x04x, x04x * c23_0 + x04y * s23_0)
            q4[1] = math.atan2(c23_1 * x04y - s23_1 * x04x, x04x * c23_1 + x04y * s23_1)
            ###########################################
            for k in range(2):
                if math.fabs(q2[k]) < ZERO_THRESH:
                    q2[k] = 0.0
                elif q2[k] < 0.0:
                    q2[k] += 2.0 * PI
                if math.fabs(q4[k]) < ZERO_THRESH:
                    q4[k] = 0.0
                elif q4[k] < 0.0:
                    q4[k] += 2.0 * PI
                q_sols[num_sols * 6 + 0] = q1[i]
                q_sols[num_sols * 6 + 1] = q2[k]
                q_sols[num_sols * 6 + 2] = q3[k]
                q_sols[num_sols * 6 + 3] = q4[k]
                q_sols[num_sols * 6 + 4] = q5[i][j]
                q_sols[num_sols * 6 + 5] = q6
                num_sols+=1
    return num_sols,q_sols


def get_ik_data(T):
    result_list=[]
    num_sols, q_sols = Iverse(T,0)
    j = 0
    for i in range(num_sols):
        result_list.append([q_sols[j], q_sols[j + 1], q_sols[j + 2], q_sols[j + 3], q_sols[j + 4], q_sols[j + 5]])
        j += 6
    return result_list



weights=[1.0]*6
#weights=[0.1,0.1,0.1,0.5,0.5,0.1]
weights=[2,2,2,2,1,1]

# T--目标点的齐次矩阵4*4；q_guess 可以理解为当前关节角度
def IKinematic(T_,q_guess,A_x):

    T = Matrix2List(A_x*T_)

    sols = get_ik_data(T)
    valid_sols = []
    for sol in sols:
        test_sol = numpy.ones(6) * 9999.
        for i in range(6):
            for add_ang in [-2.0 * numpy.pi, 0.00, 2.0 * numpy.pi]:
                test_ang = sol[i] + add_ang
                if (abs(test_ang) <= 2. * numpy.pi and
                        abs(test_ang - q_guess[i]) < abs(test_sol[i] - q_guess[i])):
                    test_sol[i] = test_ang
        if numpy.all(test_sol != 9999.):
            valid_sols.append(test_sol)
    if len(valid_sols) == 0:
        return None
    best_sol_ind = numpy.argmin(numpy.sum((weights * (valid_sols - numpy.array(q_guess))) ** 2, 1))
    # print "#########################the best sol##################"
    # print valid_sols[best_sol_ind]
    return list(valid_sols[best_sol_ind])






#从齐次矩阵转中提取姿态矩阵
def Trans2Rot(tran_matrix):
    R = numpy.matrix([[0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00]])
    transform = [tran_matrix[0,3], tran_matrix[1,3], tran_matrix[2,3]]
    for i in range (0,3):
        for j in range (0,3):
            R[i,j] = tran_matrix[i,j]
    return transform, R
# 旋转矩阵转姿态矩阵 

# rpy角转姿态矩阵


def tf2matrix(translation,rotation):

    rpy = euler_from_quaternion([rotation.x,rotation.y,rotation.z,rotation.w]) 
    R = RPY2Rot(rpy)
    trans = [translation.x,translation.y,translation.z]
    return Rot2Trans(R,trans)

def pose2matrix(position, orientation):

    rpy = euler_from_quaternion([orientation.x,orientation.y,orientation.z,orientation.w]) 
    R = RPY2Rot(rpy)
    trans = [position.x,position.y,position.z]
    return Rot2Trans(R,trans)




def List2matrix (T_list):
    T =numpy.matrix([[0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00,0.00000000e+00],
                    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00,0.00000000e+00],
                    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]])

    T[0] = T_list[0:4]
    T[1] = T_list[4:8]
    T[2] = T_list[8:12]
    T[3] = T_list[12:]
    return T

def Matrix2List(T_matrix):
    L =[]
    for i in range (0, 4):
        for j in range (0,4):
            L.append(T_matrix[i,j])

    return L

def Rot2Matrix(rotation_matrix, transform):
    T = numpy.matrix([[0.00000000e+00, 0.00000000e+00, 0.00000000e+00, transform[0]],
                    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00,transform[1]],
                    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00,transform[2]],
                    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1]])
    for i in range (0,3):
        for j in range (0,3):
            T[i,j] = rotation_matrix[i,j]
    return T

def TF_trans2Matrix(res):
    translation = [res.transform.translation.x,res.transform.translation.y,res.transform.translation.z]
    rotation = quaternion_matrix([res.transform.rotation.x,res.transform.rotation.y,res.transform.rotation.z,res.transform.rotation.w])
    T = Rot2Matrix(rotation,translation)
    return T
def Pose2Matrix(pose):
    translation = [pose.position.x,pose.position.y,pose.position.z]
    rotation = quaternion_matrix([pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w])
    T = Rot2Matrix(rotation,translation)
    return T
    
def Matrix2Pos_rpy(tran_matrix):
    position,R =Trans2Rot(tran_matrix)
    rpy = euler_from_matrix(R,'sxyz')
    return position, rpy 



# T_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

# T_mat = List2matrix(T_list)
# print T_mat
# T_ = Matrix2List(T_mat)
# print T_


#q=[-6.27000062422984765, -1.4994383521728079, 2.350452075113452, 3.585523415380623, -1.5100265287408692, 3.868303811493945e-06]

# #q=[-3.3229193595408724,-1.4997880387975124,2.350410330471382,3.5923308934136635,-0.6284408551616303,3.567942997051432e-06]

# import copy
# T = Forward(q)
# w= copy.deepcopy(q)    

# result = IKinematic(T,w)

# print list(result)
# print q




# rotation = quaternion_matrix([0.997, 0.009, 0.073, -0.003])
# print rotation


