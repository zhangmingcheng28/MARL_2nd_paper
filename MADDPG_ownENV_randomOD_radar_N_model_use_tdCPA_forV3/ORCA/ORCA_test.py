# -*- coding: utf-8 -*-
"""
@Time    : 12/4/2022 6:54 PM
@Author  : Mingcheng
@FileName: 
@Description: 
@Package dependency:
"""

import os
import math
import numpy as np
from descartes import PolygonPatch
import matplotlib.pyplot as plt
from matplotlib import animation
from shapely.geometry import LineString, Point
import csv
import time
import math
import matplotlib
import torch
import pickle
from simulator import Simulator
from vector import Vector2
import RVOmath as rvo_math

wps = Vector2(950, 275)
host_ini = Vector2(885, 400)

globalTime = 0.0
timeStep = 0.5  # second
DroneRadius = 2.5
ORCA_simulator = Simulator(globalTime, timeStep)
hostIntru = Simulator.add_agent(ORCA_simulator, host_ini, wps, DroneRadius, Vector2(0, 0), 15)
Intru1 = Simulator.add_agent(ORCA_simulator, wps, host_ini, DroneRadius, Vector2(0, 0), 15)


matplotlib.use('TkAgg')
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_aspect('equal')
step = 1
done = 0
while not done:
    ax.set_xlim([850, 980])
    ax.set_ylim([265, 430])
    for drone in ORCA_simulator.agents_:
        prefer_heading = math.atan2((drone.goalVectors.y - drone.position_.y),
                                    (drone.goalVectors.x - drone.position_.x))
        drone.pref_velocity_ = Vector2(drone.max_speed_ * math.cos(prefer_heading),
                                       drone.max_speed_ * math.sin(prefer_heading))
        if drone.id_ == 0:
            circle1 = plt.Circle((drone.position_.x, drone.position_.y), DroneRadius, color='blue', fill=False)
            ax.add_patch(circle1)
        else:
            circle2 = plt.Circle((drone.position_.x, drone.position_.y), DroneRadius, color='g', fill=False)
            ax.add_patch(circle2)
    ORCA_simulator.kd_tree_.build_agent_tree()
    a = ORCA_simulator.agents_[1].position_-ORCA_simulator.agents_[0].position_

    ORCA_simulator.agents_[0].compute_neighbors()
    ORCA_simulator.agents_[0].compute_new_velocity(0)

    print("The relative distance is {}, there are {} number of neighbors".format(np.linalg.norm((a.x, a.y)),
                                                                                 len(ORCA_simulator.agents_[0].agent_neighbors_)))

    for agentNo in range(ORCA_simulator.num_agents):
        previousPos = ORCA_simulator.agents_[agentNo].position_
        ORCA_simulator.agents_[agentNo].update(agentNo)
        currentPos = ORCA_simulator.agents_[agentNo].position_

        # check whether each agent has reached its current destination
        pass_line = LineString([(previousPos.x, previousPos.y), (currentPos.x, currentPos.y)])
        passed_volume = pass_line.buffer(DroneRadius, cap_style=1)
        agentGoalRadius = Point(ORCA_simulator.agents_[agentNo].goalVectors.x,
                                ORCA_simulator.agents_[agentNo].goalVectors.y).buffer(1)
        goal_intersect = passed_volume.intersection(agentGoalRadius)  # if host drone reaches goal?
        if not (goal_intersect.is_empty):
            done = 1
            #print('host goal reached')

    step = step + 1
    fig.canvas.draw()
    plt.show()
    time.sleep(0.5)
    fig.canvas.flush_events()
    #ax.cla()