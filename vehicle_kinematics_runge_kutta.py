# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 16:04:34 2018
Modified on Fri Aug 24 23:00:00 2018

@author: seong

Python Vehicle Kinematics

x' = v * sin(yaw)
y' = v * cos(yaw)
yaw' = v / L * tan(steer)
v' = a

Using Runge Kutta 4th order method

"""

import numpy as np
import matplotlib.pyplot as plt
import math
import time
# Use time.time() if you want to check the processing time of the Runge Kutta 4th order Method

# Vehicle States
class State:
    
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v

"""
Vehicle Kinematics Differential Equations

    x' = v * sin(yaw)
    y' = v * cos(yaw)
    yaw' = v / L * tan(steer)
    v' = a

"""
def xdot(v, yaw):
    return v * math.cos(yaw)
def ydot(v, yaw):
    return v * math.sin(yaw)
def yawdot(v, delta, L):
    return v / L * math.tan(delta)
def vdot(a):
    return 0   # Assume Acceleration is zero. i.e. Constant Velocity

def main():
    
    ### Simulation time
    t0 = 0; tf = 1
    h = 0.01
    t = np.arange(t0, tf, h)  # Simulation Time

    ### Vehicle Parameters
    L = 1  # Wheelbase

    ### Initial conditions
    x0 = 0.0
    y0 = 0.0
    yaw0 = 0.0
    v0 = 10.0      # 10 [m/s]
    
    """
    Double Lane Change (manual steering angle change)
        Steering input profile
        Right Lane Change: 1st straight / 2nd turn right / 3rd turn left / 4th straight
        Left Lane Change: 1st : straight / 2nd: turn left / 3rd: turn right / 4th: straight

    """
    delta1 = [0] * math.floor(0.125*len(t))
    delta2 = [-math.pi / 12] * math.ceil(0.125*len(t))
    delta3 = [math.pi / 12] * math.ceil(0.125*len(t))
    delta4 = [0] * math.floor(0.125*len(t))

    delta5 = [0] * math.floor(0.125*len(t))
    delta6 = [math.pi / 12] * math.ceil(0.125*len(t))
    delta7 = [-math.pi / 12] * math.ceil(0.125*len(t))
    delta8 = [0] * math.floor(0.125*len(t))

    delta_right_lane = np.concatenate((delta1, delta2, delta3, delta4), axis=0)
    delta_left_lane = np.concatenate((delta5, delta6, delta7, delta8), axis=0)
    delta = np.concatenate((delta_right_lane, delta_left_lane), axis =0)
    
    """
    States [Global_x, Global_y, Yaw(heading), Velocity]
        x: Global x         [m]
        y: Global y         [m]
        yaw: Heading angle  [rad]
        v: Velocity         [m/s]

    """
    state = State(x = x0, y = y0, yaw = yaw0, v = v0)
    x = [state.x]
    y = [state.y]
    yaw = [state.yaw]
    v = [state.v]
    
    i = 0          # for changing steering input
    sumTime = 0    # for checking processing time of the RK4 method
    
    ### Runge Kutta 4th order Method (Numerical Integration)
    while (t0 < tf):
        # tic
        start = time.time()
        
        kx1 = xdot(state.v, state.yaw)
        ky1 = ydot(state.v, state.yaw)
        kyaw1 = yawdot(state.v, delta[i], L)
        kv1 = vdot(state.v)
    
        kx2 = xdot(state.v + 0.5*h*kv1, state.yaw + 0.5*h*kyaw1)
        ky2 = ydot(state.v + 0.5*h*kv1, state.yaw + 0.5*h*kyaw1)
        kyaw2 = yawdot(state.v + 0.5*h*kv1, delta[i], L)
        kv2 = vdot(state.v + 0.5*h*kv1)
    
        kx3 = xdot(state.v + 0.5*h*kv2, state.yaw + 0.5*h*kyaw2)
        ky3 = ydot(state.v + 0.5*h*kv2, state.yaw + 0.5*h*kyaw2)
        kyaw3 = yawdot(state.v + 0.5*h*kv2, delta[i], L)
        kv3 = vdot(state.v + 0.5*h*kv2)
    
        kx4 = xdot(state.v + h*kv3, state.yaw + h*kyaw3)
        ky4 = ydot(state.v + h*kv3, state.yaw + h*kyaw3)
        kyaw4 = yawdot(state.v + h*kv3, delta[i], L)
        kv4 = vdot(state.v + h*kv3)
        
        dx = h*(kx1 + 2*kx2 + 2*kx3 + kx4) / 6;
        dy = h*(ky1 + 2*ky2 + 2*ky3 + ky4) / 6;
        dyaw = h*(kyaw1 + 2*kyaw2 + 2*kyaw3 + kyaw4) / 6;
        dv = h*(kv1 + 2*kv2 + 2*kv3 + kv4) / 6;
    
        state.x += dx
        state.y += dy
        state.yaw += dyaw
        state.v += dv
        
        t0 += h
        i += 1
        
        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        v.append(state.v)
        
        #toc
        end = time.time() - start
        sumTime += end
    
    print("Time per RK4 cycle :", sumTime / len(t))
    
    fig1 = plt.figure(1)
    axis = fig1.add_subplot(111)
    plt.plot(x, y)
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('y')
    axis.set_aspect(aspect=1)    # for 1:1 ratio plot
    
    ### Print all the yaw angles
    for i in range(0,len(yaw)):
        degree = yaw[i]*180/math.pi
        print(int(degree))

    plt.show()


main()
