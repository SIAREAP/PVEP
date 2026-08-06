#!/usr/bin/python
#coding:utf-8
from rclpy.time import Time,Duration
from datetime import timedelta
import math
import numpy
from math import cos as cos
from math import sin as sin
from math import atan2 as atan2
from math import acos as acos
from math import asin as asin
from math import sqrt as sqrt

from trajectory_msgs.msg import JointTrajectory,JointTrajectoryPoint
from competition_tutorials.my_tf import *

EPS = 0.00001
ASEND = 0.01

kitting_arm_joint_names = [
            'elbow_joint', 'linear_arm_actuator_joint', 
            'shoulder_lift_joint', 'shoulder_pan_joint', 
            'wrist_1_joint','wrist_2_joint', 'wrist_3_joint'
        ]

class polynomial7_in:
    def __init__(self,ss, se, vs, ve, as_, ae, js, je, tend, dt):

        self.ss = ss
        self.se = se
        self.vs = vs
        self.ve = ve
        self.as_ = as_
        self.ae = ae
        self.js = js
        self.je = je
        self.tend = tend
        self.dt = dt
    def __del__(self):
        pass
class polynomial7_mid:
    def __init__(self):
        self.a = [0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00]
    def __del__(self):
        pass
class polynomial7_out:
    def __init__(self):
        self.t = 0.00
        self.s = 0.00
        self.v = 0.00
        self.a = 0.00
        self.j = 0.00

    def __del__(self):
        pass

class traStruct:
    def __init__(self,q_start, q_end, t_end, t_asend = ASEND):
        self.tra_in = polynomial7_in(q_start, q_end, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,t_end,t_asend)
        self.tra_mid = polynomial7_mid()
        self.tra_out = polynomial7_out()

    def polynomial7_project(self):

        ss=self.tra_in.ss
        se=self.tra_in.se
        vs=self.tra_in.vs
        ve=self.tra_in.ve
        as_=self.tra_in.as_
        ae=self.tra_in.ae
        js=self.tra_in.js
        je=self.tra_in.je
        tend=self.tra_in.tend

        tf=tend
        tf2=tend*tend
        tf3=tend*tend*tend
        tf4=tend*tend*tend*tend
        tf5=tend*tend*tend*tend*tend
        tf6=tend*tend*tend*tend*tend*tend
        tf7=tend*tend*tend*tend*tend*tend*tend

        b1=se-js/6.0*tf3 - as_/2.0*tf2 - vs*tf - ss
        b2=ve-js/2.0*tf2 - as_*tf - vs
        b3=ae-js*tf - as_
        b4=je-js
        self.tra_mid.a[0]=ss
        self.tra_mid.a[1]=vs
        self.tra_mid.a[2]=as_/2.0
        self.tra_mid.a[3]=js/6.0
        self.tra_mid.a[4]=(35*b1)/tf4 - (15*b2)/tf3 + (5*b3)/(2*tf2) - b4/(6*tf)
        self.tra_mid.a[5]=(39*b2)/tf4 - (84*b1)/tf5 - (7*b3)/tf3 + b4/(2*tf2)
        self.tra_mid.a[6]=(70*b1)/tf6 - (34*b2)/tf5 + (13*b3)/(2*tf4) - b4/(2*tf3)
        self.tra_mid.a[7]=(10*b2)/tf6 - (20*b1)/tf7 - (2*b3)/tf5 + b4/(6*tf4)
    

    def polynomial7_computer(self):

        t_=self.tra_out.t+self.tra_in.dt
        t=0
        a =self.tra_mid.a
        if t_ >= self.tra_in.tend:
            t=self.tra_in.tend
        elif(t>=0) and (t_ < self.tra_in.tend):
            t=t_
        else:
            t =0
        t2=t*t
        t3=t*t*t
        t4=t*t*t*t
        t5=t*t*t*t*t
        t6=t*t*t*t*t*t
        t7=t*t*t*t*t*t*t

        self.tra_out.s =a[7]*t7 + a[6]*t6 + a[5]*t5 + a[4]*t4 + a[3]*t3 + a[2]*t2 + a[1]*t + a[0]
        self.tra_out.v=7*a[7]*t6 + 6*a[6]*t5 + 5*a[5]*t4 + 4*a[4]*t3 + 3*a[3]*t2 + 2*a[2]*t + a[1]
        self.tra_out.a =42*a[7]*t5 + 30*a[6]*t4 + 20*a[5]*t3 + 12*a[4]*t2 + 6*a[3]*t + 2*a[2]
        self.tra_out.j =210*a[7]*t4 + 120*a[6]*t3 + 60*a[5]*t2 + 24*a[4]*t + 6*a[3]
        self.tra_out.t=t_
        if (t_>self.tra_in.tend):
            return 1
        return 0
    
    def __del__(self):
        del self.tra_in
        del self.tra_mid
        del self.tra_out
        


def single_joint_interpolation(q_begin,q_end,t_end, t_asend =ASEND):
    traj_s = []
    # traj_v = []
    # traj_a = []
    # traj_j = []
    traj_t = []
    tS = traStruct(q_begin,q_end, t_end,t_asend)
    tS.polynomial7_project()
    if abs(q_end-q_begin)<= EPS:
        traj_s.append(q_end)
        # traj_v.append(0)
        # traj_a.append(0)
        # traj_j.append(0)
        traj_t.append(0)
    while abs(q_end-q_begin)> EPS :
        tS.polynomial7_computer()
        q_begin = tS.tra_out.s

        traj_s.append(tS.tra_out.s)
        # traj_v.append(tS.tra_out.v)
        # traj_a.append(tS.tra_out.a)
        # traj_j.append(tS.tra_out.j)
        traj_t.append(tS.tra_out.t)

    
    del tS
    return  traj_s, traj_t 

def traj_generate(arm_joint_names,q_begin,q_end,t_end,t_asend =ASEND):

    trajes_s =[]  #角度？
    # trajes_v =[]
    # trajes_a =[]
    # trajes_j =[]
    trajes_t =[]  #时间

    max_piont = 0 #最大插补点数,对每一个关节进行插补
    for i in range(0,len(q_begin)):
        traj_s,traj_t  = single_joint_interpolation(q_begin[i],q_end[i],t_end, t_asend =ASEND)
        if len(traj_s)>=max_piont:
            max_piont = len(traj_s)


        trajes_s.append(traj_s)
        # trajes_v.append(traj_v)
        # trajes_a.append(traj_a)
        # trajes_j.append(traj_j)
        trajes_t.append(traj_t)



    msg = JointTrajectory()
    msg.joint_names = arm_joint_names
    # seq_id = rospy.get_rostime()
    # msg.header.seq = seq_id.secs *10e9 +seq_id.nsecs
    # msg.header.stamp = rospy.get_rostime()

    for i in range(0,max_piont):
        point = JointTrajectoryPoint()
        positions = []
        # velocities = []
        # accelerations = []
        time_from_start = []
        
        for j in range (0,len(trajes_s)):
            while len(trajes_s[j]) < max_piont:
                trajes_s[j].append(trajes_s[j][-1])

            # while len(trajes_v[j]) < max_piont:
            #     trajes_v[j].append(trajes_v[j][-1])

            # while len(trajes_a[j]) < max_piont:
            #     trajes_a[j].append(trajes_a[j][-1])

            while len(trajes_t[j]) < max_piont:
                trajes_t[j].append(trajes_t[j][-1])

            positions.append(trajes_s[j][i])
            # velocities.append(trajes_v[j][i])
            # accelerations.append(trajes_a[j][i])
            # print("trajes_t[j][i]",trajes_t[j][i])
            duration = Duration(nanoseconds= trajes_t[j][i]* 1e9)
            time_from_start.append(duration)

        point.positions = positions
        # point.velocities = velocities
        # point.accelerations = accelerations
        point.time_from_start = max(time_from_start).to_msg()
        # rospy.Duration(add_time)
        #add_time = add_time+t_asend
       
        msg.points.append(point)
    #print msg
    return msg
   




    







