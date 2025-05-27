# -*- coding: utf-8 -*-
"""
@Time    : 3/2/2023 7:42 PM
@Author  : Mingcheng
@FileName:
@Description:
@Package dependency:
"""
import copy
import jps
import warnings
import pickle
from jps_straight import jps_find_path
from collections import OrderedDict
from shapely.ops import nearest_points
import rtree
from shapely.strtree import STRtree
from scipy.interpolate import interp1d
from shapely.geometry import LineString, Point, Polygon
from scipy.spatial import KDTree
import random
import itertools
from copy import deepcopy
from agent_randomOD_radar_sur_drones_N_Model_BAv1 import Agent
import pandas as pd
import math
import numpy as np
import os
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D
from shapely.affinity import scale
import matplotlib.pyplot as plt
import matplotlib
import re
import time
from Utilities_own_randomOD_radar_sur_drones_N_Model_BAv1 import *
import torch as T
import torch
import torch.nn.functional as F
import torch.nn as nn


class env_simulator:
    def __init__(self, world_map, building_polygons, grid_length, bound, allGridPoly, agentConfig):  # allGridPoly[0][0] is all grid=1
        self.world_map_2D = world_map  # 2D binary matrix, in ndarray form.
        self.world_map_2D_jps = None
        self.centroid_to_position_empty = {}
        self.centroid_to_position_occupied = {}
        self.world_map_2D_polyList = allGridPoly  # [0][0] is all occupied polygon, [0][1] is all non-occupied polygon
        self.agentConfig = agentConfig
        self.gridlength = grid_length
        self.buildingPolygons = building_polygons  # contain all polygon in the world that has building
        self.world_STRtree = None  # contains all polygon in the environment
        self.allbuildingSTR = None
        self.allbuildingSTR_wBound = None
        self.list_of_occupied_grid_wBound = None
        self.allbuilding_centre = None
        self.bound = bound
        self.global_time = 0.0  # in sec
        self.time_step = 0.5  # in second as well
        self.all_agents = None
        self.cur_allAgentCoor_KD = None
        self.OU_noise = None
        self.normalizer = None
        self.dummy_agent = None  # template for create a new agent
        self.max_agent_num = None

        self.spawn_area1 = []
        self.spawn_area1_polymat = []
        self.spawn_area2 = []
        self.spawn_area2_polymat = []
        self.spawn_area3 = []
        self.spawn_area3_polymat = []
        self.spawn_area4 = []
        self.spawn_area4_polymat = []
        self.spawn_pool = None
        self.target_area1 = []
        self.target_area1_polymat = None
        self.target_area2 = []
        self.target_area2_polymat = None
        self.target_area3 = []
        self.target_area3_polymat = None
        self.target_area4 = []
        self.target_area4_polymat = None
        self.target_pool = None

    def create_world(self, total_agentNum, n_actions, gamma, tau, target_update, largest_Nsigma, smallest_Nsigma, ini_Nsigma, max_xy, max_spd, acc_range):
        # config OU_noise
        self.OU_noise = OUNoise(n_actions, largest_Nsigma, smallest_Nsigma, ini_Nsigma)
        self.normalizer = NormalizeData([self.bound[0], self.bound[1]], [self.bound[2], self.bound[3]], max_spd, acc_range)
        self.all_agents = {}
        self.allbuildingSTR = STRtree(self.world_map_2D_polyList[0][0])
        building_centroid = [poly.centroid.coords[0] for poly in self.world_map_2D_polyList[0][0]]
        self.allbuilding_centre = np.array(building_centroid)

        # self.allbuilding_centre =
        worldGrid_polyCombine = []
        worldGrid_polyCombine.append(self.world_map_2D_polyList[0][0] + self.world_map_2D_polyList[0][1])
        self.world_STRtree = STRtree(worldGrid_polyCombine[0])
        for agent_i in range(total_agentNum):
            agent = Agent(n_actions, agent_i, gamma, tau, total_agentNum, max_spd)
            agent.target_update_step = target_update
            self.all_agents[agent_i] = agent
        self.dummy_agent = self.all_agents[0]

        # adjustment to world_map_2D
        # draw world_map_scatter
        scatterX = []
        scatterY = []
        centroid_pair_empty = []
        centroid_pair_occupied = []
        for poly in self.world_map_2D_polyList[0][1]:  # [0] is occupied, [1] is non occupied centroid
            scatterX.append(poly.centroid.x)
            scatterY.append(poly.centroid.y)
            centroid_pair_empty.append((poly.centroid.x, poly.centroid.y))
        for poly in self.world_map_2D_polyList[0][0]:  # [0] is occupied, [1] is non occupied centroid
            # scatterX.append(poly.centroid.x)
            # scatterY.append(poly.centroid.y)
            centroid_pair_occupied.append((poly.centroid.x, poly.centroid.y))
        start_x = int(min(scatterX))
        start_y = int(min(scatterY))
        end_x = int(max(scatterX))
        end_y = int(max(scatterY))
        world_2D = np.zeros((len(range(int(start_x), int(end_x+1), self.gridlength)), len(range(int(start_y), int(end_y+1), self.gridlength))))
        for j_idx, j_val in enumerate(range(start_y, end_y+1, self.gridlength)):
            for i_idx, i_val in enumerate(range(start_x, end_x+1, self.gridlength)):
                if (i_val, j_val) in centroid_pair_empty:
                    world_2D[i_idx][j_idx] = 0
                    self.centroid_to_position_empty[(i_val, j_val)] = [float(i_idx), float(j_idx)]
                elif (i_val, j_val) in centroid_pair_occupied:
                    world_2D[i_idx][j_idx] = 1
                    self.centroid_to_position_occupied[(i_val, j_val)] = [float(i_idx), float(j_idx)]
                else:
                    print("no corresponding coordinate found in side world 2D grid centroids, please debug!")
        self.world_map_2D = world_2D
        self.world_map_2D_jps = world_2D.astype(int).tolist()

        # segment them using two lines
        self.spawn_pool = [self.spawn_area1, self.spawn_area2, self.spawn_area3, self.spawn_area4]
        self.target_pool = [self.target_area1, self.target_area2, self.target_area3, self.target_area4]
        # target_pool_idx = [i for i in range(len(target_pool))]
        # get centroid of all square polygon
        non_occupied_polygon = self.world_map_2D_polyList[0][1]
        x_segment = (self.bound[1] - self.bound[0]) / 2 + self.bound[0]
        y_segment = (self.bound[3] - self.bound[2]) / 2 + self.bound[2]
        x_left_bound = LineString([(self.bound[0], -9999), (self.bound[0], 9999)])
        x_right_bound = LineString([(self.bound[1], -9999), (self.bound[1], 9999)])
        y_bottom_bound = LineString([(-9999, self.bound[2]), (9999, self.bound[2])])
        y_top_bound = LineString([(-9999, self.bound[3]), (9999, self.bound[3])])
        boundary_lines = [x_left_bound, x_right_bound, y_bottom_bound, y_top_bound]
        list_occupied_grids = copy.deepcopy(self.world_map_2D_polyList[0][0])
        list_occupied_grids.extend(boundary_lines)  # add boundary line to occupied lines
        self.allbuildingSTR_wBound = STRtree(list_occupied_grids)
        self.list_of_occupied_grid_wBound = list_occupied_grids
        for poly in non_occupied_polygon:
            centre_coord = (poly.centroid.x, poly.centroid.y)
            centre_coord_pt = Point(poly.centroid.x, poly.centroid.y)
            intersects_any_boundary = any(line.intersects(centre_coord_pt)for line in boundary_lines)
            if intersects_any_boundary:
                continue
            if poly.intersects(x_left_bound):
                self.spawn_area1.append(poly)
                # left line
                self.spawn_area1_polymat.append(shapelypoly_to_matpoly(poly, inFill=True, Edgecolor='black', FcColor='y'))
                # ax.add_patch(poly_mat)
            elif poly.intersects(y_bottom_bound):
                # bottom line
                self.spawn_area2.append(poly)
                self.spawn_area2_polymat.append(
                    shapelypoly_to_matpoly(poly, inFill=True, Edgecolor='black', FcColor='m'))
                # ax.add_patch(poly_mat)
            elif poly.intersects(x_right_bound):
                # right line
                self.spawn_area3.append(poly)
                self.spawn_area3_polymat.append(
                    shapelypoly_to_matpoly(poly, inFill=True, Edgecolor='black', FcColor='b'))
                # ax.add_patch(poly_mat)
            elif poly.intersects(y_top_bound):
                # top line
                self.spawn_area4.append(poly)
                self.spawn_area4_polymat.append(
                    shapelypoly_to_matpoly(poly, inFill=True, Edgecolor='black', FcColor='g'))
                # ax.add_patch(poly_mat)

            if centre_coord[0] < x_segment and centre_coord[1] < y_segment:
                self.target_area1.append(centre_coord)
                # bottom left
                # plt.plot(centre_coord[0], centre_coord[1], marker='.', color='y', markersize=2)
            elif centre_coord[0] > x_segment and centre_coord[1] < y_segment:
                self.target_area2.append(centre_coord)
                # bottom right
                # plt.plot(centre_coord[0], centre_coord[1], marker='.', color='m', markersize=2)
            elif centre_coord[0] > x_segment and centre_coord[1] > y_segment:
                self.target_area3.append(centre_coord)
                # top right
                # plt.plot(centre_coord[0], centre_coord[1], marker='.', color='b', markersize=2)
            else:
                self.target_area4.append(centre_coord)
                # top left
                # plt.plot(centre_coord[0], centre_coord[1], marker='.', color='g', markersize=2)

    def reset_world(self, total_agentNum, full_observable_critic_flag, episode, show):  # set initialize position and observation for all agents
        self.global_time = 0.0
        self.time_step = 0.5
        # reset OU_noise as well
        self.OU_noise.reset()

        # # ----------------- using fixed OD -----------------
        # # read the Excel file into a pandas dataframe
        # # df = pd.read_excel(r"D:\Multi_agent_AAC\MA_ver1\fixedDrone_3drones.xlsx")
        # df = pd.read_excel(r"F:\githubClone\Multi_agent_AAC\MA_ver1\fixedDrone_3drones.xlsx")
        # # convert the dataframe to a NumPy array
        # custom_agent_data = np.array(df)
        # # custom_agent_data = custom_agent_data.astype(float)
        # agentsCoor_list = []  # for store all agents as circle polygon
        # agentRefer_dict = {}  # A dictionary to use agent's current pos as key, their agent name (idx) as value
        # for agentIdx in self.all_agents.keys():
        #     self.all_agents[agentIdx].pos = custom_agent_data[agentIdx][0:2]
        #     self.all_agents[agentIdx].ini_pos = custom_agent_data[agentIdx][0:2]
        #     self.all_agents[agentIdx].removed_goal = None
        #     self.all_agents[agentIdx].reach_target = False
        #     # for fixed OD, we include the initial point into our goal list as well
        #     self.all_agents[agentIdx].goal = [self.all_agents[agentIdx].ini_pos]
        #
        #     if not isinstance(custom_agent_data[agentIdx][2:4][0], str):
        #         self.all_agents[agentIdx].goal = self.all_agents[agentIdx].goal + [custom_agent_data[agentIdx][2:4]]
        #     else:
        #         x_coords = np.array([int(coord.split('; ')[0]) for coord in custom_agent_data[agentIdx][2:4]])
        #         y_coords = np.array([int(coord.split('; ')[1]) for coord in custom_agent_data[agentIdx][2:4]])
        #         self.all_agents[agentIdx].goal = self.all_agents[agentIdx].goal + [x_coords, y_coords]
        #
        #     self.all_agents[agentIdx].ref_line = LineString(self.all_agents[agentIdx].goal)
        # # -----------------end of using fixed OD -----------------

        # reset the drone index to 0,1,2, ensure all index reset at starting of a new episode
        # keys = list(self.all_agents.keys())
        # for new_Idx, current_Idx in zip(range(total_agentNum), keys):
        #     self.all_agents[new_Idx] = self.all_agents.pop(current_Idx)
        #     self.all_agents[new_Idx].agent_name = 'agent_%s' % new_Idx
        #     self.all_agents[new_Idx].pre_surroundingNeighbor = {}  # at start of each episode ensure all surrounding/pre-surrounding neighbours are cleared.
        #     self.all_agents[new_Idx].surroundingNeighbor = {}
        #     self.all_agents[new_Idx].reach_target = False

        # start_time = time.time()
        agentsCoor_list = []  # for store all agents as circle polygon
        agentRefer_dict = {}  # A dictionary to use agent's current pos as key, their agent name (idx) as value

        start_pos_memory = []

        # random_start_pos_list = [(600,380), (620,380), (650,280),(650,290),(490,270),(460,330),(580,370),(500,340)]
        # random_end_pos_list = [(490,330), (470,360), (550,350),(560,340),(500,350),(620,360),(460,270),(660,280)]

        # any_collision = 0
        # loop_count = 0
        # while not any_collision:
        #     start_pos_memory = []  # make it here just to test any initial collision condition
        random_end_pos_collection = []
        for agentIdx in self.all_agents.keys():

            # ---------------- using random initialized agent position for traffic flow ---------
            random_start_index = random.randint(0, len(self.target_pool) - 1)
            numbers_left = list(range(0, random_start_index)) + list(range(random_start_index + 1, len(self.target_pool)))
            random_target_index = random.choice(numbers_left)
            random_start_pos = random.choice(self.target_pool[random_start_index])
            if len(start_pos_memory) > 0:
                while len(start_pos_memory) < len(self.all_agents):  # make sure the starting drone generated do not collide with any existing drone
                    # Generate a new point
                    random_start_index = random.randint(0, len(self.target_pool) - 1)
                    numbers_left = list(range(0, random_start_index)) + list(
                        range(random_start_index + 1, len(self.target_pool)))
                    random_target_index = random.choice(numbers_left)
                    random_start_pos = random.choice(self.target_pool[random_start_index])
                    # Check that the distance to all existing points is more than 5
                    if all(np.linalg.norm(np.array(random_start_pos)-point) > self.all_agents[agentIdx].protectiveBound*2 for point in start_pos_memory):
                        break

            random_end_pos = random.choice(self.target_pool[random_target_index])
            dist_between_se = np.linalg.norm(np.array(random_end_pos) - np.array(random_start_pos))

            # while dist_between_se >= 100:  # the distance between start & end point is more than a threshold, we reset SE pairs.
            #     random_end_pos = random.choice(self.target_pool[random_target_index])
            #     dist_between_se = np.linalg.norm(np.array(random_end_pos) - np.array(random_start_pos))

            # own self set
            # if agentIdx == 0:
            #     random_start_pos = (480, 360)
            #     random_end_pos = (600, 360)
            #     pass

            # random_start_pos = one_set_SE_collection[episode-1][agentIdx][0]
            # random_end_pos = one_set_SE_collection[episode-1][agentIdx][1]
            # random_start_pos = one_set_SE_collection[agentIdx][0]
            # random_end_pos = one_set_SE_collection[agentIdx][1]

            random_end_pos_collection.append([random_start_pos, random_end_pos])
            host_current_circle = Point(np.array(random_start_pos)[0], np.array(random_start_pos)[1]).buffer(self.all_agents[agentIdx].protectiveBound)

            possiblePoly = self.allbuildingSTR.query(host_current_circle)
            for element in possiblePoly:
                if self.allbuildingSTR.geometries.take(element).intersection(host_current_circle):
                    any_collision = 1
                    print("Initial start point {} collision with buildings".format(np.array(random_start_pos)))
                    break

            # random_start_pos = random_start_pos_list[agentIdx]
            # random_end_pos = random_end_pos_list[agentIdx]

            self.all_agents[agentIdx].pos = np.array(random_start_pos)
            self.all_agents[agentIdx].pre_pos = np.array(random_start_pos)
            self.all_agents[agentIdx].ini_pos = np.array(random_start_pos)
            start_pos_memory.append(np.array(random_start_pos))
            self.all_agents[agentIdx].removed_goal = None
            self.all_agents[agentIdx].bound_collision = False
            self.all_agents[agentIdx].building_collision = False
            self.all_agents[agentIdx].drone_collision = False
            # make sure we reset reach target
            self.all_agents[agentIdx].reach_target = False
            self.all_agents[agentIdx].collide_wall_count = 0

            # large_start = [random_start_pos[0] / self.gridlength, random_start_pos[1] / self.gridlength]
            # large_end = [random_end_pos[0] / self.gridlength, random_end_pos[1] / self.gridlength]
            # small_area_map_start = [large_start[0] - math.ceil(self.bound[0] / self.gridlength),
            #                         large_start[1] - math.ceil(self.bound[2] / self.gridlength)]
            # small_area_map_end = [large_end[0] - math.ceil(self.bound[0] / self.gridlength),
            #                       large_end[1] - math.ceil(self.bound[2] / self.gridlength)]

            small_area_map_s = self.centroid_to_position_empty[random_start_pos]
            small_area_map_e = self.centroid_to_position_empty[random_end_pos]

            width = self.world_map_2D.shape[0]
            height = self.world_map_2D.shape[1]

            jps_map = self.world_map_2D_jps

            outPath = jps_find_path((int(small_area_map_s[0]),int(small_area_map_s[1])), (int(small_area_map_e[0]),int(small_area_map_e[1])), jps_map)

            # outPath = jps.find_path(small_area_map_s, small_area_map_e, width, height, jps_map)[0]

            refinedPath = []
            curHeading = math.atan2((outPath[1][1] - outPath[0][1]),
                                    (outPath[1][0] - outPath[0][0]))
            refinedPath.append(outPath[0])
            for id_ in range(2, len(outPath)):
                nextHeading = math.atan2((outPath[id_][1] - outPath[id_ - 1][1]),
                                         (outPath[id_][0] - outPath[id_ - 1][0]))
                if curHeading != nextHeading:  # add the "id_-1" th element
                    refinedPath.append(outPath[id_ - 1])
                    curHeading = nextHeading  # update the current heading
            refinedPath.append(outPath[-1])

            # load the to goal, but remove/exclude the 1st point, which is the initial position
            self.all_agents[agentIdx].goal = [[(points[0] + math.ceil(self.bound[0] / self.gridlength)) * self.gridlength,
                                               (points[1] + math.ceil(self.bound[2] / self.gridlength)) * self.gridlength]
                                              for points in refinedPath if not np.array_equal(np.array([(points[0] + math.ceil(self.bound[0] / self.gridlength)) * self.gridlength,
                                                                                                        (points[1] + math.ceil(self.bound[2] / self.gridlength)) * self.gridlength]),
                                                                                              self.all_agents[agentIdx].ini_pos)]  # if not np.array_equal(np.array(points), self.all_agents[agentIdx].ini_pos)

            self.all_agents[agentIdx].waypoints = deepcopy(self.all_agents[agentIdx].goal)

            # load the to goal but we include the initial position
            goalPt_withini = [[(points[0] + math.ceil(self.bound[0] / self.gridlength)) * self.gridlength,
                                               (points[1] + math.ceil(self.bound[2] / self.gridlength)) * self.gridlength]
                                              for points in refinedPath]

            self.all_agents[agentIdx].ref_line = LineString(goalPt_withini)
            # ---------------- end of using random initialized agent position for traffic flow ---------

            self.all_agents[agentIdx].ref_line_segments = {}
            # Iterate over line coordinates and create line segments
            for i in range(len(self.all_agents[agentIdx].ref_line.coords) - 1):
                start_point = self.all_agents[agentIdx].ref_line.coords[i]
                end_point = self.all_agents[agentIdx].ref_line.coords[i + 1]
                segment = LineString([start_point, end_point])
                self.all_agents[agentIdx].ref_line_segments[(start_point, end_point)] = segment

            # heading in rad, must be goal_pos-intruder_pos, and y2-y1, x2-x1
            # this is the initialized heading.
            self.all_agents[agentIdx].heading = math.atan2(self.all_agents[agentIdx].goal[0][1] -
                                                           self.all_agents[agentIdx].pos[1],
                                                           self.all_agents[agentIdx].goal[0][0] -
                                                           self.all_agents[agentIdx].pos[0])

            # random_spd = random.randint(1, self.all_agents[agentIdx].maxSpeed)  # initial speed is randomly picked from 1 to max speed
            # random_spd = random.randint(1, 3)  # initial speed is randomly picked from 1 to max speed
            # random_spd = 1  # we fixed a initialized spd
            random_spd = 0  # we fixed a initialized spd
            self.all_agents[agentIdx].vel = np.array([random_spd*math.cos(self.all_agents[agentIdx].heading),
                                             random_spd*math.sin(self.all_agents[agentIdx].heading)])
            self.all_agents[agentIdx].pre_vel = np.array([random_spd*math.cos(self.all_agents[agentIdx].heading),
                                             random_spd*math.sin(self.all_agents[agentIdx].heading)])

            # NOTE: UAV's max speed don't change with time, so when we find it normalized bound, we use max speed
            # the below is the maximum normalized velocity range for map range -1 to 1, and maxSPD = 15m/s
            norm_vel_x_range = [
                -self.normalizer.norm_scale([self.all_agents[agentIdx].maxSpeed, self.all_agents[agentIdx].maxSpeed])[0],
                self.normalizer.norm_scale([self.all_agents[agentIdx].maxSpeed, self.all_agents[agentIdx].maxSpeed])[0]]
            norm_vel_y_range = [
                -self.normalizer.norm_scale([self.all_agents[agentIdx].maxSpeed, self.all_agents[agentIdx].maxSpeed])[1],
                self.normalizer.norm_scale([self.all_agents[agentIdx].maxSpeed, self.all_agents[agentIdx].maxSpeed])[1]]

            # ----------------end of initialize normalized velocity, but based on normalized map. map pos_x & pos_y are normalized to [-1, 1]---------------

            self.all_agents[agentIdx].observableSpace = self.current_observable_space(self.all_agents[agentIdx])

            cur_circle = Point(self.all_agents[agentIdx].pos[0],
                               self.all_agents[agentIdx].pos[1]).buffer(self.all_agents[agentIdx].protectiveBound,
                                                                        cap_style='round')
            # # ----------------------- end of random initialized ------------------------------

            agentRefer_dict[(self.all_agents[agentIdx].pos[0],
                             self.all_agents[agentIdx].pos[1])] = self.all_agents[agentIdx].agent_name

            agentsCoor_list.append(self.all_agents[agentIdx].pos)

        #     loop_count = loop_count + 1
        #     if loop_count % 500 == 0:
        #         print("set {} generated".format(loop_count))
        # print("The {}th set starting point generated, one initial collision case happened".format(loop_count))
        # all_eps_OD.append(random_end_pos_collection)
        # overall_state, norm_overall_state = self.cur_state_norm_state_fully_observable(agentRefer_dict)
        # print('time used is {}'.format(time.time() - start_time))
        overall_state, norm_overall_state, polygons_list, all_agent_st_pos, all_agent_ed_pos, all_agent_intersection_point_list, \
        all_agent_line_collection, all_agent_mini_intersection_list = self.cur_state_norm_state_v3(agentRefer_dict, full_observable_critic_flag)

        if show:
            os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
            matplotlib.use('TkAgg')
            fig, ax = plt.subplots(1, 1)
            for agentIdx, agent in self.all_agents.items():
                plt.plot(agent.pos[0], agent.pos[1], marker=MarkerStyle(">", fillstyle="right",
                                                                        transform=Affine2D().rotate_deg(
                                                                            math.degrees(agent.heading))), color='y')
                plt.text(agent.pos[0], agent.pos[1], agent.agent_name)
                # plot self_circle of the drone
                self_circle = Point(agent.pos[0], agent.pos[1]).buffer(agent.protectiveBound, cap_style='round')
                grid_mat_Scir = shapelypoly_to_matpoly(self_circle, False, 'k')
                # ax.add_patch(grid_mat_Scir)

                # plot drone's detection range
                detec_circle = Point(agent.pos[0], agent.pos[1]).buffer(agent.detectionRange / 2, cap_style='round')
                detec_circle_mat = shapelypoly_to_matpoly(detec_circle, False, 'r')
                # ax.add_patch(detec_circle_mat)

                # link individual drone's starting position with its goal
                ini = agent.ini_pos
                for wp in agent.goal:
                    plt.plot(wp[0], wp[1], marker='*', color='y', markersize=10)
                    plt.plot([wp[0], ini[0]], [wp[1], ini[1]], '--', color='c')
                    ini = wp
                plt.plot(agent.goal[-1][0], agent.goal[-1][1], marker='*', color='y', markersize=10)
                plt.text(agent.goal[-1][0], agent.goal[-1][1], agent.agent_name)

            # draw occupied_poly
            for one_poly in self.world_map_2D_polyList[0][0]:
                one_poly_mat = shapelypoly_to_matpoly(one_poly, True, 'y', 'b')
                # ax.add_patch(one_poly_mat)
            # draw non-occupied_poly
            for zero_poly in self.world_map_2D_polyList[0][1]:
                zero_poly_mat = shapelypoly_to_matpoly(zero_poly, False, 'y')
                ax.add_patch(zero_poly_mat)

            # show building obstacles
            for poly in self.buildingPolygons:
                matp_poly = shapelypoly_to_matpoly(poly, False, 'red')  # the 3rd parameter is the edge color
                # ax.add_patch(matp_poly)

            # show the nearest building obstacles
            # nearest_buildingPoly_mat = shapelypoly_to_matpoly(nearest_buildingPoly, True, 'g', 'k')
            # ax.add_patch(nearest_buildingPoly_mat)

            # for demo purposes
            # for poly in polygons_list:
            #     if poly.geom_type == "Polygon":
            #         matp_poly = shapelypoly_to_matpoly(poly, False, 'red')  # the 3rd parameter is the edge color
            #         ax.add_patch(matp_poly)
            #     else:
            #         x, y = poly.xy
                    # ax.plot(x, y, color='green', linewidth=2, solid_capstyle='round', zorder=3)
            # # Plot each start point
            # for point_deg, point_pos in st_points.items():
            #     ax.plot(point_pos.x, point_pos.y, 'o', color='blue')
            #
            # # Plot each end point
            # for point_deg, point_pos in ed_points.items():
            #     ax.plot(point_pos.x, point_pos.y, 'o', color='green')
            #
            # # Plot the lines of the LineString
            # for lines in line_collection:
            #     x, y = lines.xy
            #     ax.plot(x, y, color='blue', linewidth=2, solid_capstyle='round', zorder=2)
            #
            # # point_counter = 0
            # # # Plot each intersection point
            # # for point in intersection_point_list:
            # #     for ea_pt in point.geoms:
            # #         point_counter = point_counter + 1
            # #         ax.plot(ea_pt.x, ea_pt.y, 'o', color='red')
            #
            # # plot minimum intersection point
            # # for pt_dist, pt_pos in mini_intersection_list.items():
            # for pt_pos in mini_intersection_list:
            #     if pt_pos.type == 'MultiPoint':
            #         for ea_pt in pt_pos.geoms:
            #             ax.plot(ea_pt.x, ea_pt.y, 'o', color='yellow')
            #     else:
            #         ax.plot(pt_pos.x, pt_pos.y, 'o', color='red')



            # for ele in self.spawn_area1_polymat:
            #     ax.add_patch(ele)
            # for ele2 in self.spawn_area2_polymat:
            #     ax.add_patch(ele2)
            # for ele3 in self.spawn_area3_polymat:
            #     ax.add_patch(ele3)
            # for ele4 in self.spawn_area4_polymat:
            #     ax.add_patch(ele4)

            # plt.axvline(x=self.bound[0], c="green")
            # plt.axvline(x=self.bound[1], c="green")
            # plt.axhline(y=self.bound[2], c="green")
            # plt.axhline(y=self.bound[3], c="green")

            plt.xlabel("X axis")
            plt.ylabel("Y axis")
            plt.axis('equal')
            plt.show()

        return overall_state, norm_overall_state

    def reset_world_fixedOD(self, show):
        self.global_time = 0.0
        self.time_step = 0.5
        # reset OU_noise as well
        self.OU_noise.reset()

        #  custom agent position
        # x-bound: [0, 1800), y-bound: [0, 1300)
        # read the Excel file into a pandas dataframe
        df = pd.read_excel(self.agentConfig)
        # convert the dataframe to a NumPy array
        custom_agent_data = np.array(df)
        # custom_agent_data = custom_agent_data.astype(float)
        agentsCoor_list = []  # for store all agents as circle polygon
        agentRefer_dict = {}  # A dictionary to use agent's current pos as key, their agent name (idx) as value
        for agentIdx in self.all_agents.keys():
            self.all_agents[agentIdx].pos = custom_agent_data[agentIdx][0:2]
            self.all_agents[agentIdx].ini_pos = custom_agent_data[agentIdx][0:2]
            self.all_agents[agentIdx].removed_goal = None

            if not isinstance(custom_agent_data[agentIdx][2:4][0], str):
                self.all_agents[agentIdx].goal = [custom_agent_data[agentIdx][2:4]]
            else:
                x_coords = np.array([int(coord.split('; ')[0]) for coord in custom_agent_data[agentIdx][2:4]])
                y_coords = np.array([int(coord.split('; ')[1]) for coord in custom_agent_data[agentIdx][2:4]])
                self.all_agents[agentIdx].goal = [x_coords, y_coords]

            # self.all_agents[agentIdx].vel = custom_agent_data[agentIdx][4:6]

            # heading in rad, must be goal_pos-intruder_pos, and y2-y1, x2-x1
            self.all_agents[agentIdx].heading = math.atan2(self.all_agents[agentIdx].goal[0][1] -
                                                           self.all_agents[agentIdx].pos[1],
                                                           self.all_agents[agentIdx].goal[0][0] -
                                                           self.all_agents[agentIdx].pos[0])

            random_spd = random.randint(0, self.all_agents[agentIdx].maxSpeed)  # initial speed is randomly picked from 0 to max speed
            self.all_agents[agentIdx].vel = np.array([random_spd*math.cos(self.all_agents[agentIdx].heading),
                                             random_spd*math.sin(self.all_agents[agentIdx].heading)])

            self.all_agents[agentIdx].observableSpace = self.current_observable_space(self.all_agents[agentIdx])
            cur_circle = Point(self.all_agents[agentIdx].pos[0],
                               self.all_agents[agentIdx].pos[1]).buffer(self.all_agents[agentIdx].protectiveBound,
                                                                        cap_style='round')
            agentRefer_dict[(self.all_agents[agentIdx].pos[0],
                             self.all_agents[agentIdx].pos[1])] = self.all_agents[agentIdx].agent_name

            # agentSTR_list.append(cur_circle)
            agentsCoor_list.append(self.all_agents[agentIdx].pos)

        self.cur_allAgentCoor_KD = KDTree(agentsCoor_list)
        overall_state, norm_overall_state = self.cur_state_norm_state_v3(agentRefer_dict)

        if show:
            os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
            matplotlib.use('TkAgg')
            fig, ax = plt.subplots(1, 1)
            for agentIdx, agent in self.all_agents.items():
                plt.plot(agent.pos[0], agent.pos[1], marker=MarkerStyle(">", fillstyle="right",
                                                                        transform=Affine2D().rotate_deg(
                                                                            math.degrees(agent.heading))), color='y')
                plt.text(agent.pos[0], agent.pos[1], agent.agent_name)
                # plot self_circle of the drone
                self_circle = Point(agent.pos[0], agent.pos[1]).buffer(agent.protectiveBound, cap_style='round')
                grid_mat_Scir = shapelypoly_to_matpoly(self_circle, False, 'k')
                ax.add_patch(grid_mat_Scir)

                # plot drone's detection range
                detec_circle = Point(agent.pos[0], agent.pos[1]).buffer(agent.detectionRange / 2, cap_style='round')
                detec_circle_mat = shapelypoly_to_matpoly(detec_circle, False, 'r')
                ax.add_patch(detec_circle_mat)

                ini = agent.pos
                for wp in agent.goal:
                    plt.plot(wp[0], wp[1], marker='*', color='y', markersize=10)
                    plt.plot([wp[0], ini[0]], [wp[1], ini[1]], '--', color='c')
                    ini = wp

            # draw occupied_poly
            for one_poly in self.world_map_2D_polyList[0][0]:
                one_poly_mat = shapelypoly_to_matpoly(one_poly, True, 'y', 'b')
                ax.add_patch(one_poly_mat)
            # draw non-occupied_poly
            for zero_poly in self.world_map_2D_polyList[0][1]:
                zero_poly_mat = shapelypoly_to_matpoly(zero_poly, False, 'y')
                # ax.add_patch(zero_poly_mat)

            # show building obstacles
            for poly in self.buildingPolygons:
                matp_poly = shapelypoly_to_matpoly(poly, False, 'red')  # the 3rd parameter is the edge color
                ax.add_patch(matp_poly)

            # plt.axvline(x=self.bound[0], c="green")
            # plt.axvline(x=self.bound[1], c="green")
            # plt.axhline(y=self.bound[2], c="green")
            # plt.axhline(y=self.bound[3], c="green")

            plt.xlabel("X axis")
            plt.ylabel("Y axis")
            plt.axis('equal')
            plt.show()

        return overall_state, norm_overall_state

    def fill_agent_reset(self, cur_state, norm_cur_state, added_agent_keys):
        if len(added_agent_keys) == 0:
            return cur_state, norm_cur_state
        # this function is serving as a small reset function, hence when filling "agentsCoor_list" & "agentRefer_dict" we need to include all existing agents
        agentsCoor_list = []  # for store all agents as circle polygon
        agentRefer_dict = {}  # A dictionary to use agent's current pos as key, their agent name (idx) as value
        # segment them using two lines
        spawn_area1 = []  # (yellow, bottom left)
        spawn_area2 = []  # (green, top left)
        spawn_area3 = []  # (megent, bottom right)
        spawn_area4 = []  # (black, middle right)
        spawn_pool = [spawn_area1, spawn_area2, spawn_area3, spawn_area4]
        target_area1 = []
        target_area2 = []
        target_area3 = []
        target_area4 = []
        target_pool = [target_area1, target_area2, target_area3, target_area4]
        # target_pool_idx = [i for i in range(len(target_pool))]
        # get centroid of all square polygon
        non_occupied_polygon = self.world_map_2D_polyList[0][1]
        x_segment = (self.bound[1] - self.bound[0]) / 2 + self.bound[0]
        y_segment = (self.bound[3] - self.bound[2]) / 2 + self.bound[2]
        x_left_bound = LineString([(self.bound[0], -9999), (self.bound[0], 9999)])
        x_right_bound = LineString([(self.bound[1], -9999), (self.bound[1], 9999)])
        y_bottom_bound = LineString([(-9999, self.bound[2]), (9999, self.bound[2])])
        y_top_bound = LineString([(-9999, self.bound[3]), (9999, self.bound[3])])
        for poly in non_occupied_polygon:
            centre_coord = (poly.centroid.x, poly.centroid.y)
            if poly.intersects(x_left_bound):
                spawn_area1.append(poly)
                # left line
                poly_mat = shapelypoly_to_matpoly(poly, inFill=True, Edgecolor='black', FcColor='y')
                # ax.add_patch(poly_mat)
            elif poly.intersects(y_bottom_bound):
                # bottom line
                spawn_area2.append(poly)
                poly_mat = shapelypoly_to_matpoly(poly, inFill=True, Edgecolor='black', FcColor='m')
                # ax.add_patch(poly_mat)
            elif poly.intersects(x_right_bound):
                # right line
                spawn_area3.append(poly)
                poly_mat = shapelypoly_to_matpoly(poly, inFill=True, Edgecolor='black', FcColor='b')
                # ax.add_patch(poly_mat)
            elif poly.intersects(y_top_bound):
                # top line
                spawn_area4.append(poly)
                poly_mat = shapelypoly_to_matpoly(poly, inFill=True, Edgecolor='black', FcColor='g')
                # ax.add_patch(poly_mat)

            if centre_coord[0] < x_segment and centre_coord[1] < y_segment:
                target_area1.append(centre_coord)
                # bottom left
                # plt.plot(centre_coord[0], centre_coord[1], marker='.', color='y', markersize=2)
            elif centre_coord[0] > x_segment and centre_coord[1] < y_segment:
                target_area2.append(centre_coord)
                # bottom right
                # plt.plot(centre_coord[0], centre_coord[1], marker='.', color='m', markersize=2)
            elif centre_coord[0] > x_segment and centre_coord[1] > y_segment:
                target_area3.append(centre_coord)
                # top right
                # plt.plot(centre_coord[0], centre_coord[1], marker='.', color='b', markersize=2)
            else:
                target_area4.append(centre_coord)
                # top left
                # plt.plot(centre_coord[0], centre_coord[1], marker='.', color='g', markersize=2)

        for agentIdx in added_agent_keys:  # initialize for all newly added agents
            # ---------------- using random initialized agent position for traffic flow ---------
            random_start_index = random.randint(0, len(target_pool) - 1)
            numbers_left = list(range(0, random_start_index)) + list(range(random_start_index + 1, len(target_pool)))
            random_target_index = random.choice(numbers_left)
            random_start_pos = random.choice(target_pool[random_start_index])
            random_end_pos = random.choice(target_pool[random_target_index])
            dist_between = np.linalg.norm(np.array(random_end_pos) - np.array(random_start_pos))
            while dist_between <= 30:  # the distance between start & end point must be more than 30 meters
                random_end_pos = random.choice(target_pool[random_target_index])
                dist_between = np.linalg.norm(np.array(random_end_pos) - np.array(random_start_pos))

            self.all_agents[agentIdx].pre_surroundingNeighbor = {}
            self.all_agents[agentIdx].surroundingNeighbor = {}
            self.all_agents[agentIdx].observableSpace = []
            self.all_agents[agentIdx].reach_target = False
            self.all_agents[agentIdx].pre_vel = None
            self.all_agents[agentIdx].pre_pos = None
            self.all_agents[agentIdx].pos = np.array(random_start_pos)
            self.all_agents[agentIdx].ini_pos = np.array(random_start_pos)

            large_start = [random_start_pos[0] / self.gridlength, random_start_pos[1] / self.gridlength]
            large_end = [random_end_pos[0] / self.gridlength, random_end_pos[1] / self.gridlength]
            small_area_map_start = [large_start[0] - math.ceil(self.bound[0] / self.gridlength),
                                    large_start[1] - math.ceil(self.bound[2] / self.gridlength)]
            small_area_map_end = [large_end[0] - math.ceil(self.bound[0] / self.gridlength),
                                  large_end[1] - math.ceil(self.bound[2] / self.gridlength)]
            width = self.world_map_2D.shape[0]
            height = self.world_map_2D.shape[1]
            outPath = jps.find_path(small_area_map_start, small_area_map_end, width, height)[0]

            refinedPath = []
            curHeading = math.atan2((outPath[1][1] - outPath[0][1]),
                                    (outPath[1][0] - outPath[0][0]))
            refinedPath.append(outPath[0])
            for id_ in range(2, len(outPath)):
                nextHeading = math.atan2((outPath[id_][1] - outPath[id_ - 1][1]),
                                         (outPath[id_][0] - outPath[id_ - 1][0]))
                if curHeading != nextHeading:  # add the "id_-1" th element
                    refinedPath.append(outPath[id_ - 1])
                    curHeading = nextHeading  # update the current heading
            refinedPath.append(outPath[-1])

            self.all_agents[agentIdx].goal = [
                [(points[0] + math.ceil(self.bound[0] / self.gridlength)) * self.gridlength,
                 (points[1] + math.ceil(self.bound[2] / self.gridlength)) * self.gridlength] for points in refinedPath if
                not np.array_equal(np.array([(points[0] + math.ceil(self.bound[0] / self.gridlength)) * self.gridlength,
                                             (points[1] + math.ceil(
                                                 self.bound[2] / self.gridlength)) * self.gridlength]), self.all_agents[
                                       agentIdx].ini_pos)]  # if not np.array_equal(np.array(points), self.all_agents[agentIdx].ini_pos)

            # heading in rad, must be goal_pos-intruder_pos, and y2-y1, x2-x1
            self.all_agents[agentIdx].heading = math.atan2(self.all_agents[agentIdx].goal[0][1] -
                                                           self.all_agents[agentIdx].pos[1],
                                                           self.all_agents[agentIdx].goal[0][0] -
                                                           self.all_agents[agentIdx].pos[0])

            random_spd = random.randint(0, self.all_agents[agentIdx].maxSpeed)
            self.all_agents[agentIdx].vel = np.array([random_spd * math.cos(self.all_agents[agentIdx].heading),
                                                      random_spd * math.sin(self.all_agents[agentIdx].heading)])

            self.all_agents[agentIdx].observableSpace = self.current_observable_space(self.all_agents[agentIdx])
            cur_circle = Point(self.all_agents[agentIdx].pos[0],
                               self.all_agents[agentIdx].pos[1]).buffer(self.all_agents[agentIdx].protectiveBound,
                                                                        cap_style='round')
        # after add an agent, all agent's neighbour should be updated
        for agentIdx in self.all_agents.keys():
            agentRefer_dict[(self.all_agents[agentIdx].pos[0],
                             self.all_agents[agentIdx].pos[1])] = self.all_agents[agentIdx].agent_name

            # agentSTR_list.append(cur_circle)
            agentsCoor_list.append(self.all_agents[agentIdx].pos)
        # self.cur_allAgentCoor_KD = KDTree(agentsCoor_list)
        overall_state, norm_overall_state = self.cur_state_norm_state_fully_observable(agentRefer_dict)  # update agent's surrounding is inside here
        return overall_state, norm_overall_state

    def get_current_agent_nei(self, cur_agent, agentRefer_dict, queue):
        # identify neighbors (use distance)
        point_to_search = cur_agent.pos
        # subtract a small value to exclude point at exactly "search_distance"
        # search_distance = (cur_agent.detectionRange / 2) + cur_agent.protectiveBound - 1e-6
        search_distance = 10000
        distance_neigh_agent_list = []
        for agent_idx, agent in self.all_agents.items():
            if agent.agent_name == cur_agent.agent_name:
                continue
            # get neigh distance
            cur_ts_dist = np.linalg.norm(agent.pos - cur_agent.pos)
            if cur_ts_dist < search_distance:
                if queue:
                    distance_neigh_agent_list.append(
                        (cur_ts_dist, agent_idx, np.array([
                            agent.pos[0], agent.pos[1],
                            agent.vel[0], agent.vel[1],
                            agent.protectiveBound
                        ]))
                    )
                    # Sort the list by distance
                    distance_neigh_agent_list.sort(key=lambda x: x[0])

                    # Create a new ordered dictionary with sorted items
                    cur_agent.surroundingNeighbor = OrderedDict(
                        (neigh_agent_data[1], neigh_agent_data[2]) for neigh_agent_data in distance_neigh_agent_list
                    )
                else:
                    cur_agent.surroundingNeighbor[agent_idx] = np.array([agent.pos[0], agent.pos[1],
                                                                         agent.vel[0], agent.vel[1],
                                                                         agent.protectiveBound])
        return cur_agent.surroundingNeighbor

    def cur_state_norm_state_v2(self, agentRefer_dict):
        overall = []
        norm_overall = []
        # prepare for output states
        overall_state_p1 = []
        overall_state_p2 = []
        # prepare normalized output states
        norm_overall_state_p1 = []
        norm_overall_state_p2 = []
        # loop over all agent again to obtain each agent's detectable neighbor
        # second loop is required, because 1st loop is used to create the STR-tree of all agents
        # circle centre at their position
        for agentIdx, agent in self.all_agents.items():

            # get current agent's name in term of integer
            match = re.search(r'\d+(\.\d+)?', agent.agent_name)
            if match:
                agent_idx = int(match.group())
            else:
                agent_idx = None
                raise ValueError('No number found in string')
            # get agent's observable space around it
            self.all_agents[agentIdx].observableSpace = self.current_observable_space_fixedLength(self.all_agents[agentIdx])
            # identify neighbors (use distance)
            # update the "surroundingNeighbor" attribute
            agent.surroundingNeighbor = self.get_current_agent_nei(agent, agentRefer_dict)
            # reset_world function we initialized both "surroundingNeighbor" and "pre_surroundingNeighbor" identically
            agent.pre_surroundingNeighbor = agent.surroundingNeighbor

            # populate the output stateV2 overall_state is a list of length equals to total number of agents
            agent_own = np.array(
                [agent.pos[0], agent.pos[1], agent.goal[0][0] - agent.pos[0], agent.goal[0][1] - agent.pos[1],
                 agent.vel[0], agent.vel[1]])
            # populate normalized agent_own
            # norm_agent_own = []
            norm_pos = self.normalizer.nmlz_pos([agent.pos[0], agent.pos[1]])

            norm_G_diff = self.normalizer.nmlz_pos_diff(
                [agent.goal[0][0] - agent.pos[0], agent.goal[0][1] - agent.pos[1]])

            norm_vel = self.normalizer.nmlz_vel([agent.vel[0], agent.vel[1]])
            norm_agent_own = list(norm_pos + norm_G_diff + norm_vel)

            other_pos = []
            norm_other_pos = []
            for other_agentIdx, other_agent in self.all_agents.items():
                if other_agentIdx != agent_idx:
                    other_pos.append(other_agent.pos - agent.pos)
                    norm_pos = self.normalizer.nmlz_pos(other_agent.pos - agent.pos)
                    norm_other_pos.append(np.array(norm_pos))

            # overall_state.append(np.array([agent_own, agent.observableSpace, agent.surroundingNeighbor], dtype=object))
            overall_state_p1.append(np.concatenate((agent_own, np.array(other_pos).flatten())))
            overall_state_p2.append(agent.observableSpace)
            norm_overall_state_p1.append(np.concatenate((norm_agent_own, np.array(norm_other_pos).flatten())))
            norm_overall_state_p2.append(agent.observableSpace)
        overall.append(overall_state_p1)
        overall.append(overall_state_p2)
        norm_overall.append(norm_overall_state_p1)
        norm_overall.append(norm_overall_state_p2)
        return overall, norm_overall

    def cur_state_norm_state_v3(self, agentRefer_dict, full_observable_critic_flag):
        overall = []
        norm_overall = []
        # prepare for output states
        overall_state_p1 = []
        combine_overall_state_p1 = []
        overall_state_p2 = []
        combine_overall_state_p2 = []
        overall_state_p2_radar = []
        combine_overall_state_p2_radar = []
        overall_state_p3 = []

        # prepare normalized output states
        norm_overall_state_p1 = []
        combine_norm_overall_state_p1 = []
        norm_overall_state_p2 = []
        combine_norm_overall_state_p2 = []
        norm_overall_state_p2_radar = []
        combine_norm_overall_state_p2_radar = []
        norm_overall_state_p3 = []

        # record surrounding grids for all drones
        all_agent_st_pos = []
        all_agent_ed_pos = []
        all_agent_intersection_point_list = []
        all_agent_line_collection = []
        all_agent_mini_intersection_list = []
        # loop over all agent again to obtain each agent's detectable neighbor
        # second loop is required, because 1st loop is used to create the STR-tree of all agents
        # circle centre at their position
        for agentIdx, agent in self.all_agents.items():

            # get current agent's name in term of integer
            match = re.search(r'\d+(\.\d+)?', agent.agent_name)
            if match:
                agent_idx = int(match.group())
            else:
                agent_idx = None
                raise ValueError('No number found in string')

            # get agent's observable space around it
            # obs_grid_time = time.time()
            # self.all_agents[agentIdx].observableSpace = self.current_observable_space_fixedLength_fromv2_flow(self.all_agents[agentIdx])
            # self.all_agents[agentIdx].observableSpace = np.zeros((9))
            # print("generate grid time is {} milliseconds".format((time.time()-obs_grid_time)*1000))
            #
            # identify neighbors (use distance)
            # obs_nei_time = time.time()
            agent.surroundingNeighbor = self.get_current_agent_nei(agent, agentRefer_dict, queue=True)
            # # print("generate nei time is {} milliseconds".format((time.time() - obs_nei_time) * 1000))

            #region start of create radar (with UAV detection) ------------- #
            # drone_ctr = Point(agent.pos)
            # nearest_buildingPoly_idx = self.allbuildingSTR.nearest(drone_ctr)
            # nearest_buildingPoly = self.world_map_2D_polyList[0][0][nearest_buildingPoly_idx]
            # dist_nearest = drone_ctr.distance(nearest_buildingPoly)
            #
            # # Re-calculate the 20 equally spaced points around the circle
            # st_points = {degree: Point(drone_ctr.x + math.cos(math.radians(degree)) * agent.protectiveBound,
            #                              drone_ctr.y + math.sin(math.radians(degree)) * agent.protectiveBound)
            #                for degree in range(0, 360, 20)}
            # # use centre point as start point
            # st_points = {degree: drone_ctr for degree in range(0, 360, 20)}
            # all_agent_st_pos.append(st_points)
            #
            # # radar_dist = (agent.detectionRange / 2) - agent.protectiveBound
            # radar_dist = (agent.detectionRange / 2)
            # # Re-define the polygons and build the STRtree again
            # # polygons_list = [
            # #     Polygon([(1, 1), (1, 3), (3, 3), (3, 1)]),
            # #     Polygon([(2, -1), (2, -3), (4, -3), (4, -1)]),
            # #     Polygon([(-3, -1), (-3, -3), (-1, -3), (-1, -1)]),
            # #     Polygon([(-4, 2), (-4, 4), (-2, 4), (-2, 2)])
            # # ]
            # polygons_list_wBound = self.list_of_occupied_grid_wBound
            # polygons_tree_wBound = self.allbuildingSTR_wBound
            #
            # distances = []
            # intersection_point_list = []
            # mini_intersection_list = []
            # ed_points = {}
            # line_collection = []
            # for point_deg, point_pos in st_points.items():
            #     drone_nearest_flag = -1
            #     building_nearest_flag = -1
            #     # Create a line segment from the circle's center to the point on the perimeter
            #     # end_x = point_pos.x + radar_dist * math.cos(math.radians(point_deg))
            #     # end_y = point_pos.y + radar_dist * math.sin(math.radians(point_deg))
            #
            #     # Create a line segment from the circle's center
            #     end_x = drone_ctr.x + radar_dist * math.cos(math.radians(point_deg))
            #     end_y = drone_ctr.y + radar_dist * math.sin(math.radians(point_deg))
            #
            #     end_point = Point(end_x, end_y)
            #     ed_points[point_deg] = end_point
            #     min_intersection_pt = end_point  # initialize the min_intersection_pt
            #
            #     # Create the LineString from the start point to the end point
            #     line = LineString([point_pos, end_point])
            #     line_collection.append(line)
            #     # Query the STRtree for polygons that intersect with the line segment
            #     intersecting_polygons = polygons_tree_wBound.query(line)
            #
            #     drone_min_dist = line.length
            #     min_distance = line.length
            #
            #     # Build other drone's position circle, and decide the minimum intersection distance from cur host drone to other drone
            #     for other_agents_idx, others in self.all_agents.items():
            #         if other_agents_idx == agentIdx:
            #             continue
            #         other_circle = Point(others.pos).buffer(agent.protectiveBound)
            #         # Check if the LineString intersects with the circle
            #         if line.intersects(other_circle):
            #             drone_nearest_flag = 0
            #             # Find the intersection point(s)
            #             intersection = line.intersection(other_circle)
            #             # The intersection could be a Point or a MultiPoint
            #             # If it's a MultiPoint, we'll calculate the distance to the first intersection
            #             if intersection.geom_type == 'MultiPoint':
            #                 # Calculate distance from the starting point of the LineString to each intersection point
            #                 drone_perimeter_point = min(intersection.geoms, key=lambda point: drone_ctr.distance(point))
            #
            #             elif intersection.geom_type == 'Point':
            #                 # Calculate the distance from the start of the LineString to the intersection point
            #                 drone_perimeter_point = intersection
            #             elif intersection.geom_type in ['LineString', 'MultiLineString']:
            #                 # The intersection is a line (or part of the line lies on the circle's edge)
            #                 # Find the nearest point on this "intersection line" to the start of the original line
            #                 drone_perimeter_point = nearest_points(drone_ctr, intersection)[1]
            #             elif intersection.geom_type == 'GeometryCollection':
            #                 complex_min_dist = math.inf
            #                 for geom in intersection:
            #                     if geom.geom_type == 'Point':
            #                         dist = drone_ctr.distance(geom)
            #                         if dist < complex_min_dist:
            #                             complex_min_dist = dist
            #                             drone_perimeter_point = geom
            #                     elif geom.geom_type == 'LineString':
            #                         nearest_geom_point = nearest_points(drone_ctr, geom)[1]
            #                         dist = drone_ctr.distance(nearest_geom_point)
            #                         if dist < complex_min_dist:
            #                             complex_min_dist = dist
            #                             drone_perimeter_point = nearest_geom_point
            #             else:
            #                 raise ValueError(
            #                     "Intersection is not a point or multipoint, which is unexpected for LineString and Polygon intersection.")
            #             intersection_point_list.append(drone_perimeter_point)
            #             drone_distance = drone_ctr.distance(drone_perimeter_point)
            #             if drone_distance < drone_min_dist:
            #                 drone_min_dist = drone_distance
            #                 drone_nearest_pt = drone_perimeter_point
            #     # ------------ end of radar check surrounding drone's position -------------------------
            #
            #     # # If there are intersecting polygons, find the nearest intersection point
            #     if len(intersecting_polygons) != 0:  # check if a list is empty
            #         building_nearest_flag = 1
            #         # Initialize the minimum distance to be the length of the line segment
            #         for polygon_idx in intersecting_polygons:
            #             # Check if the line intersects with the building polygon's boundary
            #             if polygons_list_wBound[polygon_idx].geom_type == "Polygon":  # intersection with buildings
            #                 # pass
            #                 if line.intersects(polygons_list_wBound[polygon_idx]):
            #                     intersection_point = line.intersection(polygons_list_wBound[polygon_idx].boundary)
            #                     if intersection_point.type == 'MultiPoint':
            #                         nearest_point = min(intersection_point.geoms,
            #                                             key=lambda point: drone_ctr.distance(point))
            #                     else:
            #                         nearest_point = intersection_point
            #                     intersection_point_list.append(nearest_point)
            #                     distance = drone_ctr.distance(nearest_point)
            #                     # min_distance = min(min_distance, distance)
            #                     if distance < min_distance:
            #                         min_distance = distance
            #                         min_intersection_pt = nearest_point
            #             else:  # possible intersection is not a polygon but a LineString, intersection with boundaries
            #                 if line.intersects(polygons_list_wBound[polygon_idx]):
            #                     intersection = line.intersection(polygons_list_wBound[polygon_idx])
            #                     if intersection.geom_type == 'Point':
            #                         intersection_distance = intersection.distance(drone_ctr)
            #                         if intersection_distance < min_distance:
            #                             min_distance = intersection_distance
            #                             min_intersection_pt = intersection
            #                     # If it's a line of intersection, add each end points of the intersection line
            #                     elif intersection.geom_type == 'LineString':
            #                         for point in intersection.coords:  # loop through both end of the intersection line
            #                             one_end_of_intersection_line = Point(point)
            #                             intersection_distance = one_end_of_intersection_line.distance(drone_ctr)
            #                             if intersection_distance < min_distance:
            #                                 min_distance = intersection_distance
            #                                 min_intersection_pt = one_end_of_intersection_line
            #                     intersection_point_list.append(min_intersection_pt)
            #
            #         # make sure each look there are only one minimum intersection point
            #         distances.append([min_distance, building_nearest_flag])
            #         mini_intersection_list.append(min_intersection_pt)
            #     else:
            #         # If no intersections, the distance is the length of the line segment
            #         distances.append([line.length, building_nearest_flag])
            #     # ------ end of check intersection on polygon or boundaries ------
            #
            #     # Now we compare the minimum distance of intersection for both polygons and drones
            #     # whichever is short, we will load into the last list.
            #     # distances.append([line.length, building_nearest_flag])  # use this for we don't consider obstacles
            #
            #     if drone_min_dist < min_distance:   # one of the other drone is nearer to cur drone
            #         # replace the minimum distance and minimum intersection point
            #         if len(distances) == 0:
            #             distances.append([drone_min_dist, drone_nearest_flag])
            #         else:
            #             distances[-1] = [drone_min_dist, drone_nearest_flag]
            #         if len(mini_intersection_list) == 0:  # if no building polygon surrounding the host drone, mini_intersection_list will not be populated
            #             mini_intersection_list.append(drone_nearest_pt)
            #         else:
            #             mini_intersection_list[-1] = drone_nearest_pt
            #
            # all_agent_ed_pos.append(ed_points)
            # all_agent_intersection_point_list.append(intersection_point_list)  # this is to save all intersection point for each agent
            # all_agent_line_collection.append(line_collection)
            # all_agent_mini_intersection_list.append(mini_intersection_list)
            # self.all_agents[agentIdx].observableSpace = distances
            #endregion  end of create radar --------------- #

            #region start of create radar only used for buildings no boundary, and return building block's x,y, coord
            # drone_ctr = Point(agent.pos)
            # # current pos normalized
            # norm_pos = self.normalizer.nmlz_pos([agent.pos[0], agent.pos[1]])
            # # Re-calculate the 20 equally spaced points around the circle
            # # use centre point as start point
            # st_points = {degree: drone_ctr for degree in range(0, 360, 20)}
            # all_agent_st_pos.append(st_points)
            #
            # radar_dist = (agent.detectionRange / 2)
            #
            # polygons_list_wBound = self.list_of_occupied_grid_wBound
            # polygons_tree_wBound = self.allbuildingSTR_wBound
            #
            # distances = []
            # radar_info = []
            # intersection_point_list = []  # the current radar prob may have multiple intersections points with other geometries
            # mini_intersection_list = []  # only record the intersection point that is nearest to the drone's centre
            # ed_points = {}
            # line_collection = []  # a collection of all 20 radar's prob
            # for point_deg, point_pos in st_points.items():
            #     # Create a line segment from the circle's center
            #     end_x = drone_ctr.x + radar_dist * math.cos(math.radians(point_deg))
            #     end_y = drone_ctr.y + radar_dist * math.sin(math.radians(point_deg))
            #
            #     end_point = Point(end_x, end_y)
            #
            #     # current radar prob heading
            #     cur_prob_heading = math.atan2(end_y-agent.pos[1], end_x-agent.pos[0])
            #
            #     ed_points[point_deg] = end_point
            #     min_intersection_pt = end_point
            #
            #     # Create the LineString from the start point to the end point
            #     line = LineString([point_pos, end_point])
            #     line_collection.append(line)
            #     possible_interaction = polygons_tree_wBound.query(line)
            #     # Check if the LineString intersects with the circle
            #     shortest_dist = math.inf  # initialize shortest distance
            #     sensed_shortest_dist = line.length  # initialize actual prob distance
            #     distances.append(line.length)
            #     if len(possible_interaction) != 0:  # check if a list is empty
            #         building_nearest_flag = 1
            #         # Initialize the minimum distance to be the length of the line segment
            #         for polygon_idx in possible_interaction:
            #             # Check if the line intersects with the building polygon's boundary
            #             if polygons_list_wBound[polygon_idx].geom_type == "Polygon":
            #                 if line.intersects(polygons_list_wBound[polygon_idx]):
            #                     with warnings.catch_warnings():
            #                         warnings.simplefilter('ignore', category=RuntimeWarning)
            #                         intersection_point = line.intersection(polygons_list_wBound[polygon_idx].boundary)
            #                     if intersection_point.geom_type == 'MultiPoint':
            #                         nearest_point = min(intersection_point.geoms,
            #                                             key=lambda point: drone_ctr.distance(point))
            #                     else:
            #                         nearest_point = intersection_point
            #                     intersection_point_list.append(nearest_point)
            #                     sensed_shortest_dist = drone_ctr.distance(nearest_point)
            #                     if sensed_shortest_dist < shortest_dist:
            #                         shortest_dist = sensed_shortest_dist
            #                         min_intersection_pt = nearest_point
            #                         end_point = min_intersection_pt
            #                         # intersection_obstacle_centroid = polygons_list_wBound[polygon_idx].centroid
            #                         # norm_intersection_obstacle_centroid = self.normalizer.nmlz_pos([intersection_obstacle_centroid.x, intersection_obstacle_centroid.y])
            #                         # norm_intersection_delta_pos = norm_pos - norm_intersection_obstacle_centroid
            #                         norm_intersection_obstacle = self.normalizer.nmlz_pos([min_intersection_pt.x, min_intersection_pt.y])
            #                         norm_intersection_delta_pos = norm_pos - norm_intersection_obstacle
            #             else:  # possible intersection is not a polygon but a LineString, meaning it is a boundary line
            #                 if line.intersects(polygons_list_wBound[polygon_idx]):
            #                     with warnings.catch_warnings():
            #                         warnings.simplefilter('ignore', category=RuntimeWarning)
            #                         intersection = line.intersection(polygons_list_wBound[polygon_idx])
            #                     if intersection.geom_type == 'Point':
            #                         sensed_shortest_dist = intersection.distance(drone_ctr)
            #                         if sensed_shortest_dist < shortest_dist:
            #                             shortest_dist = sensed_shortest_dist
            #                             min_intersection_pt = intersection
            #                             end_point = min_intersection_pt
            #                             # if the radar prob intersects with the boundary line, this is a special type of obstacle, we just store the coordinates of the intersection point.
            #                             intersection_obstacle_centroid = min_intersection_pt
            #                             norm_intersection_obstacle_centroid = self.normalizer.nmlz_pos(
            #                                 [intersection_obstacle_centroid.x, intersection_obstacle_centroid.y])
            #                             norm_intersection_delta_pos = norm_pos - norm_intersection_obstacle_centroid
            #                     # If it's a line of intersection, add each end points of the intersection line
            #                     elif intersection.geom_type == 'LineString':
            #                         for point in intersection.coords:  # loop through both end of the intersection line
            #                             one_end_of_intersection_line = Point(point)
            #                             sensed_shortest_dist = one_end_of_intersection_line.distance(drone_ctr)
            #                             if sensed_shortest_dist < shortest_dist:
            #                                 shortest_dist = sensed_shortest_dist
            #                                 min_intersection_pt = one_end_of_intersection_line
            #                                 end_point = min_intersection_pt
            #                                 # if the radar prob intersects with the boundary line, this is a special type of obstacle, we just store the coordinates of the intersection point.
            #                                 intersection_obstacle_centroid = min_intersection_pt
            #                                 norm_intersection_obstacle_centroid = self.normalizer.nmlz_pos(
            #                                     [intersection_obstacle_centroid.x, intersection_obstacle_centroid.y])
            #                                 norm_intersection_delta_pos = norm_pos - norm_intersection_obstacle_centroid
            #                     intersection_point_list.append(min_intersection_pt)
            #
            #         # make sure each look there are only one minimum intersection point
            #         distances[-1] = sensed_shortest_dist
            #         mini_intersection_list.append(min_intersection_pt)
            #     else:
            #         # If no intersections, the distance is the length of the line segment
            #         distances[-1] = line.length
            #         mini_intersection_list.append(min_intersection_pt)
            #         norm_intersection_obstacle_centroid = np.array([-2, -2])
            #         norm_intersection_delta_pos = np.array([-2, -2])
            #     # radar_info.append(norm_intersection_obstacle_centroid[0])
            #     radar_info.append(norm_intersection_delta_pos[0])
            #     # radar_info.append(norm_intersection_obstacle_centroid[1])
            #     radar_info.append(norm_intersection_delta_pos[1])
            #     self.all_agents[agentIdx].probe_line[point_deg] = LineString([point_pos, end_point])
            # all_agent_ed_pos.append(ed_points)
            # all_agent_intersection_point_list.append(intersection_point_list)
            # all_agent_line_collection.append(line_collection)
            # all_agent_mini_intersection_list.append(mini_intersection_list)
            # # self.all_agents[agentIdx].observableSpace = np.array(distances)
            # self.all_agents[agentIdx].observableSpace = np.array(radar_info)

            #endregion end of create radar only used for buildings no boundary, and return building block's x,y, coord

            #region  ---- start of radar creation (only detect surrounding obstacles) ----
            drone_ctr = Point(agent.pos)

            # Re-calculate the 20 equally spaced points around the circle

            # use centre point as start point
            st_points = {degree: drone_ctr for degree in range(0, 360, 20)}
            all_agent_st_pos.append(st_points)

            radar_dist = (agent.detectionRange / 2)

            polygons_list_wBound = self.list_of_occupied_grid_wBound
            polygons_tree_wBound = self.allbuildingSTR_wBound

            distances = []
            radar_info = []
            intersection_point_list = []  # the current radar prob may have multiple intersections points with other geometries
            mini_intersection_list = []  # only record the intersection point that is nearest to the drone's centre
            ed_points = {}
            line_collection = []  # a collection of all 20 radar's prob
            for point_deg, point_pos in st_points.items():
                # Create a line segment from the circle's center
                end_x = drone_ctr.x + radar_dist * math.cos(math.radians(point_deg))
                end_y = drone_ctr.y + radar_dist * math.sin(math.radians(point_deg))

                end_point = Point(end_x, end_y)

                # current radar prob heading
                cur_prob_heading = math.atan2(end_y-agent.pos[1], end_x-agent.pos[0])

                ed_points[point_deg] = end_point
                min_intersection_pt = end_point
                drone_perimeter_point = end_point

                # Create the LineString from the start point to the end point
                line = LineString([point_pos, end_point])
                line_collection.append(line)
                possible_interaction = polygons_tree_wBound.query(line)
                # Check if the LineString intersects with the circle
                shortest_dist = math.inf  # initialize shortest distance
                sensed_shortest_dist = line.length  # initialize actual prob distance
                distances.append(line.length)
                if len(possible_interaction) != 0:  # check if a list is empty
                    building_nearest_flag = 1
                    # Initialize the minimum distance to be the length of the line segment
                    for polygon_idx in possible_interaction:
                        # Check if the line intersects with the building polygon's boundary
                        if polygons_list_wBound[polygon_idx].geom_type == "Polygon":
                            if line.intersects(polygons_list_wBound[polygon_idx]):
                                intersection_point = line.intersection(polygons_list_wBound[polygon_idx].boundary)
                                if intersection_point.geom_type == 'MultiPoint':
                                    nearest_point = min(intersection_point.geoms,
                                                        key=lambda point: drone_ctr.distance(point))
                                else:
                                    nearest_point = intersection_point
                                intersection_point_list.append(nearest_point)
                                sensed_shortest_dist = drone_ctr.distance(nearest_point)
                                if sensed_shortest_dist < shortest_dist:
                                    shortest_dist = sensed_shortest_dist
                                    min_intersection_pt = nearest_point
                        else:  # possible intersection is not a polygon but a LineString, meaning it is a boundary line
                            if line.intersects(polygons_list_wBound[polygon_idx]):
                                intersection = line.intersection(polygons_list_wBound[polygon_idx])
                                if intersection.geom_type == 'Point':
                                    sensed_shortest_dist = intersection.distance(drone_ctr)
                                    if sensed_shortest_dist < shortest_dist:
                                        shortest_dist = sensed_shortest_dist
                                        min_intersection_pt = intersection
                                # If it's a line of intersection, add each end points of the intersection line
                                elif intersection.geom_type == 'LineString':
                                    for point in intersection.coords:  # loop through both end of the intersection line
                                        one_end_of_intersection_line = Point(point)
                                        sensed_shortest_dist = one_end_of_intersection_line.distance(drone_ctr)
                                        if sensed_shortest_dist < shortest_dist:
                                            shortest_dist = sensed_shortest_dist
                                            min_intersection_pt = one_end_of_intersection_line
                                intersection_point_list.append(min_intersection_pt)

                    # make sure each look there are only one minimum intersection point
                    distances[-1] = sensed_shortest_dist
                    mini_intersection_list.append(min_intersection_pt)
                else:
                    # If no intersections, the distance is the length of the line segment
                    distances[-1] = line.length
                    mini_intersection_list.append(min_intersection_pt)
                radar_info.append(sensed_shortest_dist)
                radar_info.append(cur_prob_heading)
            all_agent_ed_pos.append(ed_points)
            all_agent_intersection_point_list.append(intersection_point_list)
            all_agent_line_collection.append(line_collection)
            all_agent_mini_intersection_list.append(mini_intersection_list)
            self.all_agents[agentIdx].observableSpace = np.array(distances)
            # self.all_agents[agentIdx].observableSpace = np.array(radar_info)
            #endregion ---- end of radar creation (only detect surrounding obstacles) ----

            # -------- normalize radar reading by its maximum range -----
            # for ea_dist_idx, ea_dist in enumerate(self.all_agents[agentIdx].observableSpace):
            #     ea_dist = ea_dist / (self.all_agents[agentIdx].detectionRange / 2)
            #     self.all_agents[agentIdx].observableSpace[ea_dist_idx] = ea_dist
            # -------- end of normalize radar reading by its maximum range -----

            rest_compu_time = time.time()

            host_current_point = Point(agent.pos[0], agent.pos[1])
            cross_err_distance, x_error, y_error, nearest_pt = self.cross_track_error(host_current_point,
                                                                          agent.ref_line)  # deviation from the reference line, cross track error
            norm_cross_track_deviation_x = x_error * self.normalizer.x_scale
            norm_cross_track_deviation_y = y_error * self.normalizer.y_scale

            # no_norm_cross = np.array([x_error, y_error])
            norm_cross = np.array([norm_cross_track_deviation_x, norm_cross_track_deviation_y])

            # ----- discrete the ref line --------------
            if agent.pre_pos is None:
                cur_heading_rad = agent.heading
            else:
                cur_heading_rad = math.atan2(agent.pos[1]-agent.pre_pos[1], agent.pos[0]-agent.pre_pos[0])

            host_detection_circle = host_current_point.buffer(agent.detectionRange / 2)

            point_b = nearest_points(agent.ref_line, host_current_point)[0]  # [0] meaning return must be nearer to the 1st input variable
            dist_to_b = agent.ref_line.project(point_b)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                line_within_circle = agent.ref_line.intersection(host_detection_circle)
            if line_within_circle.length == 0:
                # If there is no intersection, we determine whether this drone is on the left or right of the nearest line segment
                # Identify the closest segment to the nearest point on the line
                segments = list(zip(agent.ref_line.coords[:-1], agent.ref_line.coords[1:]))
                closest_segment = min(segments, key=lambda seg: LineString(seg).distance(point_b))
                # Calculate the side using cross product logic
                A = closest_segment[0]
                B = closest_segment[1]
                C = (agent.pos[0], agent.pos[1])
                # Compute the cross product
                cross_product = (B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0])
                if cross_product > 0:  # on left of the closest line segment
                    points_spread = [-2 for _ in range(20)]
                elif cross_product < 0:  # on the right of the closest line segment
                    points_spread = [2 for _ in range(20)]
                else:
                    points_spread = [0 for _ in range(20)]
                    print("point is on the line, which has very low chance, in that case we just assign 0.")
                ref_line_obs = points_spread
                norm_ref_line_obs = np.array(points_spread)

            else:
                # Calculate the total distance we can spread out points from Point B
                total_spread_distance = min(agent.detectionRange / 2, line_within_circle.length)
                # Calculate the interval for the points
                interval = total_spread_distance / 10
                # Get 10 points along the LineString from Point B
                points_spread = [line_within_circle.interpolate(dist_to_b + interval * i) for i in range(1, 11)]
                # For demonstration, return the coordinates of the points
                ref_line_obs = [coord for point in points_spread for coord in point.coords[0]]
                # we normalize these ref_line_coordinates
                norm_ref_line_obs = np.array(
                    [norm_coo for point in points_spread for norm_coo in self.normalizer.scale_pos(point.coords[0])])

            # ----- end of discrete the ref line --------------

            # ------ find nearest neighbour ------
            # loop through neighbors from current time step, and search for the nearest neighbour and its neigh_keys
            nearest_neigh_key = None
            shortest_neigh_dist = math.inf
            for neigh_keys in self.all_agents[agentIdx].surroundingNeighbor:
                # ----- start of make nei invis when neigh reached their goal -----
                # check if this drone reached their goal yet
                nei_cur_circle = Point(self.all_agents[neigh_keys].pos[0],
                                            self.all_agents[neigh_keys].pos[1]).buffer(self.all_agents[neigh_keys].protectiveBound)

                nei_tar_circle = Point(self.all_agents[neigh_keys].goal[-1]).buffer(1,
                                                                               cap_style='round')  # set to [-1] so there are no more reference path
                # when there is no intersection between two geometries, "RuntimeWarning" will appear
                # RuntimeWarning is, "invalid value encountered in intersection"
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', category=RuntimeWarning)
                    nei_goal_intersect = nei_cur_circle.intersection(nei_tar_circle)
                # if not nei_goal_intersect.is_empty:  # current neigh has reached their goal  # this will affect the drone's state space observation do note of this.
                #     continue  # straight away pass this neigh which has already reached.

                # ----- end of make nei invis when neigh reached their goal -----
                # get distance from host to all the surrounding vehicles
                diff_dist_vec = agent.pos - self.all_agents[neigh_keys].pos  # host pos vector - intruder pos vector
                euclidean_dist_diff = np.linalg.norm(diff_dist_vec)
                if euclidean_dist_diff < shortest_neigh_dist:
                    shortest_neigh_dist = euclidean_dist_diff
                    nearest_neigh_key = neigh_keys

            if nearest_neigh_key == None:
                nearest_neigh_pos = [-2, -2]
                norm_nearest_neigh_pos = nearest_neigh_pos
                delta_nei = nearest_neigh_pos
                norm_delta_nei = np.array(nearest_neigh_pos)
                nearest_neigh_vel = nearest_neigh_pos
                norm_nearest_neigh_vel = nearest_neigh_pos
            else:
                nearest_neigh_pos = self.all_agents[nearest_neigh_key].pos
                norm_nearest_neigh_pos = self.normalizer.nmlz_pos(nearest_neigh_pos)
                delta_nei = nearest_neigh_pos - agent.pos
                norm_delta_nei = norm_nearest_neigh_pos - self.normalizer.nmlz_pos([agent.pos[0], agent.pos[1]])
                nearest_neigh_vel = self.all_agents[nearest_neigh_key].vel
                norm_nearest_neigh_vel = self.normalizer.norm_scale(
                    [nearest_neigh_vel[0], nearest_neigh_vel[1]])  # normalization using scale

            # ------- end if find nearest neighbour ------

            # norm_pos = self.normalizer.scale_pos([agent.pos[0], agent.pos[1]])
            norm_pos = self.normalizer.nmlz_pos([agent.pos[0], agent.pos[1]])

            # norm_vel = self.normalizer.norm_scale([agent.vel[0], agent.vel[1]])  # normalization using scale
            norm_vel = self.normalizer.nmlz_vel([agent.vel[0], agent.vel[1]])  # normalization using min_max

            # norm_acc = self.normalizer.norm_scale([agent.acc[0], agent.acc[1]])
            norm_acc = self.normalizer.nmlz_acc([agent.acc[0], agent.acc[1]])  # norm using min_max

            norm_G = self.normalizer.nmlz_pos([agent.goal[-1][0], agent.goal[-1][1]])
            norm_deltaG = norm_G - norm_pos  # drone's position relative to goal, so is like treat goal as the origin.

            norm_seg = self.normalizer.nmlz_pos([agent.goal[0][0], agent.goal[0][1]])
            norm_delta_segG = norm_seg - norm_pos

            # agent_own = np.array([agent.vel[0], agent.vel[1], agent.acc[0], agent.acc[1],
            #                       agent.goal[-1][0]-agent.pos[0], agent.goal[-1][1]-agent.pos[1]])
            # agent_own = np.array([agent.pos[0], agent.pos[1], agent.vel[0], agent.vel[1], agent.acc[0], agent.acc[1],
            #                       agent.goal[-1][0]-agent.pos[0], agent.goal[-1][1]-agent.pos[1]])

            # agent_own = np.array([agent.pos[0], agent.pos[1], agent.vel[0], agent.vel[1],
            #                       agent.goal[-1][0]-agent.pos[0], agent.goal[-1][1]-agent.pos[1]])

            # agent_own = np.array([agent.pos[0], agent.pos[1], agent.vel[0], agent.vel[1], x_error, y_error,
            #                       agent.goal[-1][0]-agent.pos[0], agent.goal[-1][1]-agent.pos[1]])

            # agent_own = np.array([agent.pos[0], agent.pos[1], agent.vel[0], agent.vel[1], x_error, y_error,
            #                       agent.goal[-1][0]-agent.pos[0], agent.goal[-1][1]-agent.pos[1], nearest_neigh_pos[0],
            #                       nearest_neigh_pos[1]])

            # agent_own = np.array([agent.pos[0], agent.pos[1], agent.vel[0], agent.vel[1], x_error, y_error,
            #                       agent.goal[-1][0]-agent.pos[0], agent.goal[-1][1]-agent.pos[1], delta_nei[0], delta_nei[1]])

            # agent_own = np.array([agent.pos[0], agent.pos[1], agent.vel[0], agent.vel[1], x_error, y_error,
            #                       agent.goal[-1][0]-agent.pos[0], agent.goal[-1][1]-agent.pos[1], delta_nei[0], delta_nei[1],
            #                       nearest_neigh_vel[0], nearest_neigh_vel[1]])

            # agent_own = np.array([agent.pos[0], agent.pos[1], agent.vel[0], agent.vel[1]]+ref_line_obs+
            #                       [agent.goal[-1][0]-agent.pos[0], agent.goal[-1][1]-agent.pos[1]])

            # agent_own = np.array([agent.pos[0], agent.pos[1], agent.vel[0], agent.vel[1],
            #                       agent.goal[-1][0]-agent.pos[0], agent.goal[-1][1]-agent.pos[1],
            #                       agent.goal[0][0]-agent.pos[0], agent.goal[0][1]-agent.pos[1]])

            # agent_own = np.array([agent.vel[0], agent.vel[1],
            #                       agent.goal[-1][0]-agent.pos[0], agent.goal[-1][1]-agent.pos[1]])

            # norm_agent_own = np.concatenate([norm_pos, norm_vel, norm_deltaG], axis=0)
            # norm_agent_own = np.concatenate([norm_pos, norm_vel, norm_cross, norm_deltaG], axis=0)
            # norm_agent_own = np.concatenate([norm_pos, norm_vel, norm_cross, norm_deltaG, norm_nearest_neigh_pos], axis=0)
            # norm_agent_own = np.concatenate([norm_pos, norm_vel, norm_cross, norm_deltaG, norm_delta_nei], axis=0)
            # norm_agent_own = np.concatenate([norm_pos, norm_vel, norm_cross, norm_deltaG, norm_delta_nei, norm_nearest_neigh_vel], axis=0)
            # norm_agent_own = np.concatenate([norm_pos, norm_vel, norm_ref_line_obs, norm_deltaG], axis=0)
            # norm_agent_own = np.concatenate([norm_pos, norm_vel, norm_acc, norm_deltaG], axis=0)
            # norm_agent_own = np.concatenate([norm_vel, norm_acc, norm_deltaG], axis=0)

            # norm_agent_own = np.concatenate([norm_pos, norm_vel, norm_deltaG, norm_delta_segG], axis=0)
            # norm_agent_own = np.concatenate([norm_vel, norm_deltaG], axis=0)

            # ---------- based on 1 Dec 2023, add obs for ref line -----------
            # host_current_point = Point(agent.pos[0], agent.pos[1])
            # cross_err_distance, x_error, y_error = self.cross_track_error(host_current_point, agent.ref_line)  # deviation from the reference line, cross track error
            # norm_cross_track_deviation_x = x_error * self.normalizer.x_scale
            # norm_cross_track_deviation_y = y_error * self.normalizer.y_scale
            #
            # agent_own = np.array([agent.pos[0], agent.pos[1], agent.vel[0], agent.vel[1],
            #                       agent.goal[-1][0]-agent.pos[0], agent.goal[-1][1]-agent.pos[1], x_error, y_error, cross_err_distance])
            #
            # combine_normXY = math.sqrt(norm_cross_track_deviation_x**2 + norm_cross_track_deviation_y**2)
            # norm_cross = np.array([norm_cross_track_deviation_x, norm_cross_track_deviation_y, combine_normXY])
            #
            # norm_agent_own = np.concatenate([norm_pos, norm_vel, norm_deltaG, norm_cross], axis=0)
            # ---------- end of based on 1 Dec 2023, add obs for ref line -----------

            other_agents = []
            norm_other_agents = []
            p1_other_agents = []
            p1_norm_other_agents = []
            # p2_just_euclidean_delta = []
            p2_just_neighbour = []
            p2_norm_just_neighbour = []
            nearest_neight = []
            norm_nearest_neigh = []
            # filling term for no surrounding agent detected
            pre_total_possible_conflict = 0  # total possible conflict between the host drone and the current neighbour
            cur_total_possible_conflict = 0  # total possible conflict between the host drone and the current neighbour
            tcpa = -10
            pre_tcpa = -10
            d_tcpa = -10
            pre_d_tcpa = -10
            include_neigh_count = 0
            if len(agent.surroundingNeighbor) > 0:  # meaning there is surrounding neighbors around the current agent
                for other_agentIdx, other_agent in agent.surroundingNeighbor.items():
                    if other_agentIdx != agent_idx:
                        nei_px = self.all_agents[other_agentIdx].pos[0]
                        nei_py = self.all_agents[other_agentIdx].pos[1]
                        delta_host_x = self.all_agents[other_agentIdx].pos[0] - agent.pos[0]
                        delta_host_y = self.all_agents[other_agentIdx].pos[1] - agent.pos[1]
                        euclidean_dist = np.linalg.norm(self.all_agents[other_agentIdx].pos - agent.pos)

                        # norm_delta_pos = self.normalizer.scale_pos([delta_host_x, delta_host_y])
                        norm_nei_pos = self.normalizer.nmlz_pos([self.all_agents[other_agentIdx].pos[0],
                                                                 self.all_agents[other_agentIdx].pos[1]])
                        norm_delta_pos = norm_pos - norm_nei_pos # neigh's position relative to host drone. Host drone as origin.

                        norm_euclidean_dist = np.linalg.norm(norm_delta_pos)

                        nei_goal_diff_x = self.all_agents[other_agentIdx].goal[-1][0]-agent.pos[0]
                        nei_goal_diff_y = self.all_agents[other_agentIdx].goal[-1][1]-agent.pos[1]

                        nei_heading = self.all_agents[other_agentIdx].heading
                        nei_acc = self.all_agents[other_agentIdx].acc
                        nei_norm_acc = self.normalizer.nmlz_acc([nei_acc[0], nei_acc[1]])

                        cur_neigh_vx = self.all_agents[other_agentIdx].vel[0]
                        cur_neigh_vy = self.all_agents[other_agentIdx].vel[1]
                        norm_neigh_vel = self.normalizer.nmlz_vel([cur_neigh_vx, cur_neigh_vy])  # normalization using min_max
                        cur_neigh_ax = self.all_agents[other_agentIdx].acc[0]
                        cur_neigh_ay = self.all_agents[other_agentIdx].acc[1]
                        # norm_neigh_acc = self.normalizer.norm_scale([cur_neigh_ax, cur_neigh_ay])
                        norm_neigh_acc = self.normalizer.nmlz_acc([cur_neigh_ax, cur_neigh_ay])

                        # calculate current t_cpa/d_cpa
                        tcpa, d_tcpa, cur_total_possible_conflict = compute_t_cpa_d_cpa_potential_col(self.all_agents[other_agentIdx].pos, agent.pos, self.all_agents[other_agentIdx].vel, agent.vel, self.all_agents[other_agentIdx].protectiveBound, agent.protectiveBound, cur_total_possible_conflict)
                        # -------------------------------------------------

                        # calculate previous t_cpa/d_cpa
                        pre_tcpa, pre_d_tcpa, pre_total_possible_conflict = compute_t_cpa_d_cpa_potential_col(
                            self.all_agents[other_agentIdx].pre_pos, agent.pre_pos, self.all_agents[other_agentIdx].pre_vel,
                            agent.pre_vel, self.all_agents[other_agentIdx].protectiveBound, agent.protectiveBound,
                            pre_total_possible_conflict)
                        # ---------------------------
                        if len(nearest_neight) == 0:
                            # nearest_neight = np.array([delta_host_x, delta_host_y, cur_neigh_vx, cur_neigh_vy, nei_heading])
                            nearest_neight = np.array([delta_host_x, delta_host_y])
                        if len(norm_nearest_neigh) == 0:
                            # norm_nearest_neigh = np.array([norm_delta_pos[0], norm_delta_pos[1], norm_neigh_vel[0], [1]])
                            # norm_nearest_neigh = np.append(norm_nearest_neigh, agent.heading)
                            norm_nearest_neigh = np.array([norm_delta_pos[0], norm_delta_pos[1]])

                        # p1_surround_agent = np.array([delta_host_x, delta_host_y, cur_neigh_vx, cur_neigh_vy])
                        # p1_surround_agent = np.array([delta_host_x, delta_host_y, euclidean_dist, cur_neigh_vx, cur_neigh_vy])
                        # p1_surround_agent = np.array([delta_host_x, delta_host_y, euclidean_dist, cur_neigh_vx, cur_neigh_vy, nei_heading])
                        # p1_surround_agent = np.array([delta_host_x, delta_host_y, euclidean_dist, cur_neigh_vx,
                        #                               cur_neigh_vy, nei_acc[0], nei_acc[1], nei_heading])
                        p1_surround_agent = np.array([delta_host_x, delta_host_y, cur_neigh_vx, cur_neigh_vy, nei_heading])
                        # p1_surround_agent = np.array([nei_px, nei_py, cur_neigh_vx, cur_neigh_vy, nei_goal_diff_x,
                        #                               nei_goal_diff_y, nei_heading])
                        # p1_norm_surround_agent = np.concatenate([norm_delta_pos, norm_neigh_vel], axis=0)
                        # p1_norm_surround_agent = np.concatenate([norm_delta_pos, np.array([euclidean_dist]), norm_neigh_vel], axis=0)
                        # p1_norm_surround_agent = np.concatenate([norm_delta_pos, np.array([euclidean_dist]), norm_neigh_vel], axis=0)
                        # p1_norm_surround_agent = np.append(p1_norm_surround_agent, agent.heading)
                        # p1_norm_surround_agent = np.concatenate([norm_delta_pos, np.array([euclidean_dist]), norm_neigh_vel, nei_norm_acc], axis=0)
                        # p1_norm_surround_agent = np.concatenate([norm_delta_pos, np.array([norm_euclidean_dist]), norm_neigh_vel, nei_norm_acc], axis=0)
                        p1_norm_surround_agent = np.concatenate([norm_delta_pos, norm_neigh_vel], axis=0)
                        p1_norm_surround_agent = np.append(p1_norm_surround_agent, agent.heading)
                        # p1_norm_surround_agent = np.concatenate([norm_nei_pos, norm_neigh_vel, ], axis=0)

                        surround_agent = np.array([[other_agent[0] - agent.pos[0],
                                                   other_agent[1] - agent.pos[1],
                                                   other_agent[-2] - other_agent[0],
                                                   other_agent[-1] - other_agent[1],
                                                   other_agent[2], other_agent[3]]])

                        norm_pos_diff = self.normalizer.nmlz_pos_diff(
                            [other_agent[0] - agent.pos[0], other_agent[1] - agent.pos[1]])

                        norm_G_diff = self.normalizer.nmlz_pos_diff(
                            [other_agent[-2] - other_agent[0], other_agent[-1] - other_agent[1]])

                        norm_vel = tuple(self.normalizer.nmlz_vel([other_agent[2], other_agent[3]]))
                        # norm_vel = self.normalizer.nmlz_vel([other_agent[2], other_agent[3]])
                        norm_surround_agent = np.array([list(norm_pos_diff + norm_G_diff + norm_vel)])

                        other_agents.append(surround_agent)
                        norm_other_agents.append(norm_surround_agent)
                        p1_other_agents.append(p1_surround_agent)
                        p1_norm_other_agents.append(p1_norm_surround_agent)
                        # p2_just_euclidean_delta.append(euclidean_dist)
                        p2_just_neighbour.append(p1_surround_agent)
                        p2_norm_just_neighbour.append(p1_norm_surround_agent)
                        include_neigh_count = include_neigh_count + 1
                        # if include_neigh_count > 0:  # only include 2 nearest agents
                        #     break
                overall_state_p3.append(other_agents)
                norm_overall_state_p3.append(norm_other_agents)
            else:
                overall_state_p3.append([np.zeros((1, 6))])
                norm_overall_state_p3.append([np.zeros((1, 6))])

            max_neigh_count = len(self.all_agents) - 1
            filling_required = max_neigh_count - len(agent.surroundingNeighbor)
            # filling_value = -2
            filling_value = 0
            # filling_dim = 5
            filling_dim = 4
            for _ in range(filling_required):
                p1_other_agents.append(np.array([filling_value]*filling_dim))
                p1_norm_other_agents.append(np.array([filling_value]*filling_dim))
            all_other_agents = np.concatenate(p1_other_agents)
            norm_all_other_agents = np.concatenate(p1_norm_other_agents)

            all_neigh_agents = np.concatenate(p2_just_neighbour)
            norm_all_neigh_agents = np.concatenate(p2_norm_just_neighbour)

            # agent_own = np.array([agent.pos[0], agent.pos[1], agent.vel[0], agent.vel[1], x_error, y_error,
            #                       agent.goal[-1][0]-agent.pos[0], agent.goal[-1][1]-agent.pos[1],
            #                       tcpa, d_tcpa, pre_total_possible_conflict, cur_total_possible_conflict])

            # self_obs = np.array([agent.pos[0], agent.pos[1], agent.vel[0], agent.vel[1], x_error, y_error,
            #                       agent.goal[-1][0]-agent.pos[0], agent.goal[-1][1]-agent.pos[1],
            #                       pre_total_possible_conflict, cur_total_possible_conflict])

            # self_obs = np.array([agent.pos[0], agent.pos[1], agent.vel[0], agent.vel[1],
            #                       agent.goal[-1][0]-agent.pos[0], agent.goal[-1][1]-agent.pos[1],
            #                       pre_total_possible_conflict, cur_total_possible_conflict])

            # self_obs = np.array([agent.pos[0], agent.pos[1], agent.vel[0], agent.vel[1],
            #                       agent.goal[-1][0]-agent.pos[0], agent.goal[-1][1]-agent.pos[1]])

            # self_obs = np.array([agent.pos[0], agent.pos[1], agent.vel[0], agent.vel[1],
            #                       agent.goal[-1][0]-agent.pos[0], agent.goal[-1][1]-agent.pos[1], agent.heading])

            # self_obs = np.array([agent.pos[0], agent.pos[1], agent.vel[0], agent.vel[1],
            #                       agent.goal[-1][0]-agent.pos[0], agent.goal[-1][1]-agent.pos[1],
            #                      agent.acc[0], agent.acc[1], agent.heading])

            self_obs = np.array([agent.pos[0], agent.pos[1], agent.vel[0], agent.vel[1],
                                  agent.goal[-1][0]-agent.pos[0], agent.goal[-1][1]-agent.pos[1], agent.heading])

            # self_obs = np.array([agent.pos[0], agent.pos[1], agent.vel[0], agent.vel[1],
            #                       agent.goal[-1][0]-agent.pos[0], agent.goal[-1][1]-agent.pos[1], agent.heading, delta_nei[0], delta_nei[1]])

            # self_obs = np.array([agent.vel[0], agent.vel[1],
            #                       agent.goal[-1][0]-agent.pos[0], agent.goal[-1][1]-agent.pos[1],
            #                       pre_total_possible_conflict, cur_total_possible_conflict])

            # agent_own = np.concatenate((self_obs, all_other_agents), axis=0)
            agent_own = self_obs
            # agent_own = np.concatenate((self_obs, nearest_neight), axis=0)

            # norm_agent_own = np.concatenate([norm_pos, norm_vel, norm_cross, norm_deltaG,
            #                                  (tcpa, d_tcpa, pre_total_possible_conflict, cur_total_possible_conflict)], axis=0)

            # norm_self_obs = np.concatenate([norm_pos, norm_vel, norm_cross, norm_deltaG,
            #                                  (pre_total_possible_conflict, cur_total_possible_conflict)], axis=0)

            # norm_self_obs = np.concatenate([norm_pos, norm_vel, norm_deltaG,
            #                                  (pre_total_possible_conflict, cur_total_possible_conflict)], axis=0)

            # norm_self_obs = np.concatenate([norm_pos, norm_vel, norm_deltaG], axis=0)
            # norm_self_obs = np.append(norm_self_obs, agent.heading)  # we have to do this because heading dim=1

            # norm_self_obs = np.concatenate([norm_pos, norm_vel, norm_deltaG, norm_acc], axis=0)
            # norm_self_obs = np.append(norm_self_obs, agent.heading)  # we have to do this because heading dim=1

            norm_self_obs = np.concatenate([norm_pos, norm_vel, norm_deltaG], axis=0)
            norm_self_obs = np.append(norm_self_obs, agent.heading)  # we have to do this because heading dim=1
            # norm_self_obs = np.append(norm_self_obs, norm_delta_nei)  # we have to do this because heading dim=1

            # norm_self_obs = np.append(norm_self_obs, norm_nearest_neigh)

            # norm_self_obs = np.concatenate([norm_vel, norm_deltaG,
            #                                  (pre_total_possible_conflict, cur_total_possible_conflict)], axis=0)

            # norm_agent_own = np.concatenate((norm_self_obs, norm_all_other_agents), axis=0)
            norm_agent_own = norm_self_obs

            overall_state_p1.append(agent_own)
            # overall_state_p2.append(agent.observableSpace)
            overall_state_p2_radar.append(agent.observableSpace)
            overall_state_p2.append(all_neigh_agents)

            # distances_list = [dist_element[0] for dist_element in agent.observableSpace]
            # mini_index = find_index_of_min_first_element(agent.observableSpace)
            # # distances_list.append(agent.observableSpace[mini_index][1])  # append the one-hot, -1 meaning no detection, 1 is building, 0 is drone
            # overall_state_p2.append(distances_list)

            norm_overall_state_p1.append(norm_agent_own)
            # norm_overall_state_p2.append(agent.observableSpace)
            norm_overall_state_p2_radar.append(agent.observableSpace)
            norm_overall_state_p2.append(norm_all_neigh_agents)

            # norm_overall_state_p2.append(distances_list)

        overall.append(overall_state_p1)
        overall.append(overall_state_p2)
        overall.append(overall_state_p2_radar)
        overall.append(overall_state_p3)
        for list_ in overall_state_p3:
            if len(list_) == 0:
                print("check")
        norm_overall.append(norm_overall_state_p1)
        norm_overall.append(norm_overall_state_p2)
        norm_overall.append(norm_overall_state_p2_radar)
        norm_overall.append(norm_overall_state_p3)
        # print("rest compute time is {} milliseconds".format((time.time() - rest_compu_time) * 1000))
        return overall, norm_overall, polygons_list_wBound, all_agent_st_pos, all_agent_ed_pos, all_agent_intersection_point_list, all_agent_line_collection, all_agent_mini_intersection_list

    def cur_state_norm_state_fully_observable(self, agentRefer_dict):
        overall = []
        norm_overall = []
        # prepare for output states
        overall_state_p1 = []
        fullyObs_p1 = []
        overall_state_p2 = []
        overall_state_p3 = []
        # prepare normalized output states
        norm_overall_state_p1 = []
        norm_fullyObs_p1 = []
        norm_overall_state_p2 = []
        norm_overall_state_p3 = []
        # loop over all agent again to obtain each agent's detectable neighbor
        # second loop is required, because 1st loop is used to create the STR-tree of all agents
        # circle centre at their position
        for agentIdx, agent in self.all_agents.items():

            # get current agent's name in term of integer
            match = re.search(r'\d+(\.\d+)?', agent.agent_name)
            if match:
                agent_idx = int(match.group())
            else:
                agent_idx = None
                raise ValueError('No number found in string')
            # get agent's observable space around it
            self.all_agents[agentIdx].observableSpace = self.current_observable_space_fixedLength_fromv2_flow(self.all_agents[agentIdx])
            # self.all_agents[agentIdx].observableSpace = self.current_observable_space_fixedLength(self.all_agents[agentIdx])
            # identify neighbors (use distance)
            # update the "surroundingNeighbor" attribute
            agent.surroundingNeighbor = self.get_current_agent_nei_V2(agent, agentRefer_dict)

            # all_other_posdiff = []
            # pair_posdiff = []
            # for other_agentIdx, other_agent in self.all_agents.items():
            #     if other_agentIdx == agentIdx:
            #         continue
            #     all_other_posdiff.append(other_agent.pos[0]-agent.pos[0])
            #     all_other_posdiff.append(other_agent.pos[1]-agent.pos[1])
            #     pair_posdiff.append(other_agent.pos-agent.pos)
            # all_other_posdiff = np.array(all_other_posdiff)


            agent_own = np.array(
                [agent.pos[0], agent.pos[1], agent.goal[0][0] - agent.pos[0], agent.goal[0][1] - agent.pos[1],
                 agent.vel[0], agent.vel[1], agent.acc[0], agent.acc[1]])

            # agent_own = np.concatenate((agent_own, all_other_posdiff))
            # populate normalized agent_own
            # norm_agent_own = []
            norm_pos = self.normalizer.nmlz_pos([agent.pos[0], agent.pos[1]])

            norm_G_diff = self.normalizer.nmlz_pos_diff(
                [agent.goal[0][0] - agent.pos[0], agent.goal[0][1] - agent.pos[1]])

            # norm_other_diff = tuple(self.normalizer.nmlz_pos_diff(pos_diff_pair) for pos_diff_pair in pair_posdiff)

            norm_vel = self.normalizer.nmlz_vel([agent.vel[0], agent.vel[1]])
            norm_acc = self.normalizer.nmlz_acc([agent.acc[0], agent.acc[1]])
            norm_agent_own = np.array(list(norm_pos + norm_G_diff + norm_vel+norm_acc))

            other_agents = []
            combined_other_agent = []
            norm_combined_other_agent = []
            norm_other_agents = []
            if len(agent.surroundingNeighbor) > 0:  # meaning there is surrounding neighbors around the current agent
                for other_agentIdx, other_agent in agent.surroundingNeighbor.items():
                    if other_agentIdx != agent_idx:
                        surround_agent = np.array([[other_agent[0] - agent.pos[0],
                                                   other_agent[1] - agent.pos[1],
                                                   other_agent[-2] - other_agent[0],
                                                   other_agent[-1] - other_agent[1],
                                                   other_agent[2], other_agent[3]]])

                        norm_pos_diff = self.normalizer.nmlz_pos_diff(
                            [other_agent[0] - agent.pos[0], other_agent[1] - agent.pos[1]])

                        norm_G_diff = self.normalizer.nmlz_pos_diff(
                            [other_agent[-2] - other_agent[0], other_agent[-1] - other_agent[1]])

                        norm_vel = self.normalizer.nmlz_vel([other_agent[2], other_agent[3]])
                        norm_surround_agent = np.array([list(norm_pos_diff + norm_G_diff + norm_vel)])

                        combined_other_agent.append([other_agent[0] - agent.pos[0], other_agent[1] - agent.pos[1], other_agent[2], other_agent[3]])  # position difference with host drone + own velocity, so is nx4
                        norm_combined_other_agent.append(list(norm_pos_diff+norm_vel))

                        other_agents.append(surround_agent)
                        norm_other_agents.append(norm_surround_agent)
                combined_other_agent = list(itertools.chain.from_iterable(combined_other_agent))
                norm_combined_other_agent = list(itertools.chain.from_iterable(norm_combined_other_agent))
                overall_state_p3.append(other_agents)
                norm_overall_state_p3.append(norm_other_agents)
            else:
                overall_state_p3.append([np.zeros((1, 6))])
                norm_overall_state_p3.append([np.zeros((1, 6))])

            # overall_state_p1.append(agent_own)
            fullyObs_p1.append(np.concatenate((agent_own, np.array(combined_other_agent))))
            overall_state_p2.append(agent.observableSpace)

            # norm_overall_state_p1.append(norm_agent_own)
            norm_fullyObs_p1.append(np.concatenate((norm_agent_own, np.array(norm_combined_other_agent))))
            norm_overall_state_p2.append(agent.observableSpace)

        overall.append(fullyObs_p1)
        overall.append(overall_state_p2)
        overall.append(overall_state_p3)
        norm_overall.append(norm_fullyObs_p1)
        norm_overall.append(norm_overall_state_p2)
        norm_overall.append(norm_overall_state_p3)
        return overall, norm_overall

    def current_observable_space(self, cur_agent):
        occupied_building_val = 10
        occupied_drone_val = 50
        non_occupied_val = 1
        currentObservableState = []
        cur_hostPos_from_input = np.array([cur_agent.pos[0], cur_agent.pos[1]])
        t_x = cur_hostPos_from_input[0]
        t_y = cur_hostPos_from_input[1]
        polygonSet = []  # this polygonSet including the polygon that intersect with the "self_circle"
        self_circle_inter = []
        worldGrid_polyCombine = []
        # self.world_map_2D_polyList[0][0] is all grid=1, or list of occupied grids
        worldGrid_polyCombine.append(self.world_map_2D_polyList[0][0] + self.world_map_2D_polyList[0][1])
        world_STRtree = STRtree(worldGrid_polyCombine[0])
        detection_circle = Point(t_x, t_y).buffer(cur_agent.detectionRange / 2, cap_style='round')
        self_circle = Point(t_x, t_y).buffer(cur_agent.protectiveBound, cap_style='round')
        possible_matches = world_STRtree.query(detection_circle)

        for poly in world_STRtree.geometries.take(possible_matches):
            if detection_circle.intersects(poly):
                polygonSet.append(poly)
            if self_circle.intersects(poly):
                self_circle_inter.append(poly)

        # all detectable grids (not arranged)
        no_sorted_polySet = polygonSet

        # all detectable grids (arranged)
        sorted_polySet = sort_polygons(polygonSet)
        for poly in sorted_polySet:
            if self_circle.intersects(poly):
                currentObservableState.append(occupied_drone_val)
                continue
            if poly in self.world_map_2D_polyList[0][0]:
                currentObservableState.append(occupied_building_val)
            else:
                currentObservableState.append(non_occupied_val)
        # currently we are using arranged polygonSet and 1D array
        return currentObservableState

    def get_current_agent_nei_V2(self, cur_agent, agentRefer_dict):
        # identify neighbors (use distance)
        point_to_search = cur_agent.pos
        # subtract a small value to exclude point at exactly "search_distance"
        # search_distance = (cur_agent.detectionRange / 2) + cur_agent.protectiveBound
        search_distance = (300000 / 2) + cur_agent.protectiveBound

        for agent_pos, agent_idx_string in agentRefer_dict.items():
            if cur_agent.agent_name == agent_idx_string:
                continue
            if np.linalg.norm(np.array(point_to_search) - np.array(agent_pos))<=search_distance:
                other_agent_idx = int(re.search(r'\d+(\.\d+)?', agent_idx_string).group())
                cur_agent.surroundingNeighbor[other_agent_idx] = np.array([self.all_agents[other_agent_idx].pos[0],
                                                                           self.all_agents[other_agent_idx].pos[1],
                                                                           self.all_agents[other_agent_idx].vel[0],
                                                                           self.all_agents[other_agent_idx].vel[1],
                                                                           self.all_agents[other_agent_idx].goal[0][0],
                                                                           self.all_agents[other_agent_idx].goal[0][1]])

        return cur_agent.surroundingNeighbor

    def current_observable_space_fixedLength(self, cur_agent):
        occupied_building_val = -10
        occupied_drone_val = 50
        non_occupied_val = 1
        host_polygon = None
        worldGrid_polyCombine = []
        # self.world_map_2D_polyList[0][0] is all grid=1, or list of occupied grids
        worldGrid_polyCombine.append(self.world_map_2D_polyList[0][0] + self.world_map_2D_polyList[0][1])
        world_STRtree = STRtree(worldGrid_polyCombine[0])
        currentObservableState = []
        cur_hostPos_from_input = np.array([cur_agent.pos[0], cur_agent.pos[1]])
        t_x = cur_hostPos_from_input[0]
        t_y = cur_hostPos_from_input[1]
        curPt = Point(t_x, t_y)
        no_sorted_polySet = []  # this polygonSet including the polygon that intersect with the "self_circle"
        possible_poly_idx = world_STRtree.query(curPt)
        containList = []
        for poly in world_STRtree.geometries.take(possible_poly_idx).tolist():
            if poly.contains(curPt):
                containList.append(poly)
        if len(containList)!=1:
            containList = [None]  # we clear the list.
            detection_circle = Point(t_x, t_y).buffer(cur_agent.detectionRange / 2, cap_style='round')
            possible_matches = world_STRtree.query(detection_circle)
            highest_overlap_area = 0
            for poly in world_STRtree.geometries.take(possible_matches):
                intersect = poly.intersection(detection_circle)
                intersection_area = intersect.area
                if intersection_area > highest_overlap_area:
                    if intersection_area == highest_overlap_area:
                        print("There are two surrounding polygon has the same overlap area on the host drone's protective area please debug!")
                    highest_overlap_area = intersection_area
                    containList[0] = poly
            if len(containList) == 0:
                print("none intersection detected please debug!")

        larger_polygon = scale(containList[0], xfact=3, yfact=3,
                               origin='center')  # when xfact and yfact both equals to 3, my bigger polygon is of square shape, edge length is 3 times compared to the input square polygon

        poly_idx = world_STRtree.query(larger_polygon)
        for poly in world_STRtree.geometries.take(poly_idx).tolist():
            if larger_polygon.contains(poly):
                no_sorted_polySet.append(poly)


        # for poly in world_STRtree.geometries.take(possible_matches):
        #     if detection_circle.intersects(poly):
        #         polygonSet.append(poly)
        #     if self_circle.intersects(poly):
        #         self_circle_inter.append(poly)


        # all detectable grids (arranged)
        sorted_polySet = sort_polygons(no_sorted_polySet)
        for poly in sorted_polySet:
            if poly.equals(containList[0]):
                currentObservableState.append(occupied_drone_val)
            if poly in self.world_map_2D_polyList[0][0]:  # if polygon is an element of occupied polygon
                currentObservableState.append(occupied_building_val)
            else:
                currentObservableState.append(non_occupied_val)
        # currently we are using arranged polygonSet and 1D array
        return np.array(currentObservableState)

    def current_observable_space_fixedLength_fromv2_flow(self, cur_agent):
        # This function should output an array of length 9. In case that, when agent is at edge of the 2D map, we just patch with 0.
        occupied_building_val = -10
        occupied_drone_val = 50
        non_occupied_val = 1
        max_out_length = 9
        host_polygon = None

        currentObservableState = []
        cur_hostPos_from_input = np.array([cur_agent.pos[0], cur_agent.pos[1]])
        t_x = cur_hostPos_from_input[0]
        t_y = cur_hostPos_from_input[1]
        curPt = Point(t_x, t_y)
        no_sorted_polySet = []  # this polygonSet including the polygon that intersect with the "self_circle"
        possible_poly_idx = self.world_STRtree.query(curPt)
        containList = []
        for poly in self.world_STRtree.geometries.take(possible_poly_idx).tolist():
            if poly.contains(curPt):
                containList.append(poly)
        if len(containList)!=1:
            containList = [None]  # we clear the list.
            detection_circle = Point(t_x, t_y).buffer(cur_agent.detectionRange / 2, cap_style='round')
            possible_matches = self.world_STRtree.query(detection_circle)
            highest_overlap_area = 0
            for poly in self.world_STRtree.geometries.take(possible_matches):
                intersect = poly.intersection(detection_circle)
                intersection_area = intersect.area
                if intersection_area > highest_overlap_area:
                    if intersection_area == highest_overlap_area:
                        print("There are two surrounding polygon has the same overlap area on the host drone's protective area please debug!")
                    highest_overlap_area = intersection_area
                    containList[0] = poly
            if len(containList) == 0:
                print("none intersection detected please debug!")
        if containList[0] == None:
            print("debug")
        larger_polygon = scale(containList[0], xfact=3, yfact=3,
                               origin='center')  # when xfact and yfact both equals to 3, my bigger polygon is of square shape, edge length is 3 times compared to the input square polygon

        poly_idx = self.world_STRtree.query(larger_polygon)
        for poly in self.world_STRtree.geometries.take(poly_idx).tolist():
            if larger_polygon.contains(poly):
                no_sorted_polySet.append(poly)


        # for poly in world_STRtree.geometries.take(possible_matches):
        #     if detection_circle.intersects(poly):
        #         polygonSet.append(poly)
        #     if self_circle.intersects(poly):
        #         self_circle_inter.append(poly)


        # all detectable grids (arranged)
        sorted_polySet = sort_polygons(no_sorted_polySet)
        for poly in sorted_polySet:
            if poly.equals(containList[0]):
                currentObservableState.append(occupied_drone_val)
            elif poly in self.world_map_2D_polyList[0][0]:  # check if polygon is an element of occupied polygon
                currentObservableState.append(occupied_building_val)
            else:
                currentObservableState.append(non_occupied_val)
        # currently we are using arranged polygonSet and 1D array
        if len(currentObservableState) < max_out_length:
            currentObservableState.extend([0] * (max_out_length - len(currentObservableState)))
        if not currentObservableState:
            print("check")
        return np.array(currentObservableState)

    def get_actions_noCR(self):
        outActions = {}
        noCR = 1
        vel = [None] * 2
        for agent_idx, agent in self.all_agents.items():
            # heading in rad must be goal_pos-intruder_pos, and y2-y1, x2-x1
            agent.heading = math.atan2(agent.goal[0][1] - agent.pos[1],
                                       agent.goal[0][0] - agent.pos[0])
            vel[0] = (agent.maxSpeed/2) * math.cos(agent.heading)
            vel[1] = (agent.maxSpeed/2) * math.sin(agent.heading)
            outActions[agent_idx] = np.array([vel[0], vel[1]])
        return outActions

    def get_actions_NN(self, combine_state, eps):  # decentralized execution, only actor net is used here
        outActions = {}
        for agent_idx, agent in self.all_agents.items():
            # obtain the observation for each individual actor
            individual_obs = extract_individual_obs(combine_state, agent_idx)
            # chosen_action = agent.choose_actions(individual_obs)  # when deals with three different section of inputs
            input_tensor = T.tensor(individual_obs.reshape(1, -1), dtype=T.float).to(agent.actorNet.device)
            input_tensor_d = input_tensor.detach()
            chosen_action = agent.actorNet(input_tensor_d)
            # squeeze(0) is to remove the batch information
            # then convert to numpy
            # chosen_action = chosen_action.squeeze(0).detach().numpy()
            chosen_action = chosen_action + T.tensor(self.OU_noise.noise()) # add noise for exploration first, before clamp
            # clip the action
            chosen_action = T.clamp(chosen_action, -1, 1)
            # for ea_idx, ea_a in enumerate(chosen_action):
            #     if ea_a < -1:
            #         chosen_action[ea_idx] = -1
            #     if ea_a > 1:
            #         chosen_action[ea_idx] = 1

            # update current sigma used for the exploration
            self.OU_noise.sigma = self.OU_noise.largest_sigma * eps + (1 - eps) * self.OU_noise.smallest_sigma
            outActions[agent_idx] = np.squeeze(chosen_action.data.cpu().numpy())  # load to output dict

        return outActions

    def get_step_reward_5_v3(self, current_ts, step_reward_record):  # this is for individual drones, current_ts = current time step
        reward, done = [], []
        agent_to_remove = []
        one_step_reward = []
        check_goal = [False] * len(self.all_agents)
        reward_record_idx = 0  # this is used as a list index, increase with for loop. No need go with agent index, this index is also shared by done checking
        # crash_penalty = -200
        crash_penalty = -300
        reach_target = 300
        potential_conflict_count = 0
        final_goal_toadd = 0
        fixed_domino_reward = 1
        x_left_bound = LineString([(self.bound[0], -9999), (self.bound[0], 9999)])
        x_right_bound = LineString([(self.bound[1], -9999), (self.bound[1], 9999)])
        y_bottom_bound = LineString([(-9999, self.bound[2]), (9999, self.bound[2])])
        y_top_bound = LineString([(-9999, self.bound[3]), (9999, self.bound[3])])

        for drone_idx, drone_obj in self.all_agents.items():
            one_agent_reward_record = []
            # re-initialize these two list for individual agents at each time step,this is to ensure collision
            # condition is reset for each agent at every time step
            collision_drones = []
            collide_building = 0
            pc_before, pc_after = [], []
            dist_toHost = []
            # we assume the maximum potential conflict the current drone could have at each time step is equals
            # to the total number of its neighbour at each time step
            pc_max_before = len(drone_obj.pre_surroundingNeighbor)
            pc_max_after = len(drone_obj.surroundingNeighbor)

            # calculate the deviation from the reference path after an action has been taken
            curPoint = Point(self.all_agents[drone_idx].pos)
            if isinstance(self.all_agents[drone_idx].removed_goal, np.ndarray):
                host_refline = LineString([self.all_agents[drone_idx].removed_goal, self.all_agents[drone_idx].goal[0]])
            else:
                host_refline = LineString([self.all_agents[drone_idx].ini_pos, self.all_agents[drone_idx].goal[0]])
            cross_track_deviation = curPoint.distance(host_refline)  # deviation from the reference line, cross track error

            host_pass_line = LineString([self.all_agents[drone_idx].pre_pos, self.all_agents[drone_idx].pos])
            host_passed_volume = host_pass_line.buffer(self.all_agents[drone_idx].protectiveBound, cap_style='round')

            # neigh_keys is the drone_idx for current neighbors
            # loop through neighbors from previous time step
            for neigh_keys in self.all_agents[drone_idx].pre_surroundingNeighbor:
                # compute potential conflicts before and after the action for the current drone with its neighbours
                try:
                    one_pc_before = compute_potential_conflict(drone_obj.pre_pos, drone_obj.pre_vel,
                                                           drone_obj.protectiveBound, self.all_agents[neigh_keys].pre_pos,
                                                           self.all_agents[neigh_keys].pre_vel,
                                                           self.all_agents[neigh_keys].protectiveBound, neigh_keys,
                                                           current_ts)
                except:
                    print("pause")
                if len(one_pc_before) > 0:
                    pc_before.append(one_pc_before)
            # loop through neighbors from current time step
            for neigh_keys in self.all_agents[drone_idx].surroundingNeighbor:
                # compute potential conflicts before and after the action for the current drone with its neighbours
                one_pc_after = compute_potential_conflict(drone_obj.pos, drone_obj.vel,
                                                      drone_obj.protectiveBound, self.all_agents[neigh_keys].pos,
                                                      self.all_agents[neigh_keys].vel,
                                                      self.all_agents[neigh_keys].protectiveBound, neigh_keys,
                                                      current_ts)
                if len(one_pc_after) > 0:
                    pc_after.append(one_pc_after)

                # get distance from host to all the surrounding vehicles
                diff_dist_vec = drone_obj.pos - self.all_agents[neigh_keys].pos  # host pos vector - intruder pos vector
                dist_toHost.append(np.linalg.norm(diff_dist_vec))
                # check whether the current drone has collides with any surrounding neighbors due to current action
                neigh_pass_line = LineString([self.all_agents[neigh_keys].pre_pos, self.all_agents[neigh_keys].pos])
                neigh_passed_volume = neigh_pass_line.buffer(self.all_agents[neigh_keys].protectiveBound,
                                                             cap_style='round')
                if host_passed_volume.intersects(neigh_passed_volume):
                    print("drone_{} collide with drone_{} at time step {}".format(drone_idx, neigh_keys, current_ts))
                    collision_drones.append(neigh_keys)



            # if pc_max_after == 0:  # upper bound in terms of value case for dominoTerm
            #     dominoTerm = fixed_domino_reward
            # elif pc_max_before == 0:  # lower bound in terms of value case for dominoTerm
            #     dominoTerm = -1
            # elif (len(pc_after)/pc_max_after) == 0:  # check if denominator of the dominoTerm equals to 0
            #     # if denominator equals to 0, meaning initial velocity is 0,
            #     # so is like initial condition, then we can just assign this dominoTerm as 0.
            #     dominoTerm = 0
            # else:
            #     dominoTerm = ((len(pc_before)/pc_max_before) -
            #                   (len(pc_after)/pc_max_after)) / (len(pc_after)/pc_max_after)

            # if len(pc_after) == 0:
            #     dominoTerm = fixed_domino_reward
            # else:
            #     dominoTerm = (len(pc_before) - len(pc_after)) / len(pc_after)
            dominoTerm = []  # reset for every new decision making drone
            for neigh_keys, t_cpa, d_cpa in pc_after:
                # td_cpa[1] is d_cpa, td_cpa[0] is t_cpa
                # dominoValue = ((2*drone_obj.protectiveBound)/math.exp(td_cpa[1])) * (math.exp(-td_cpa[0]))
                dominoValue = ((5/math.exp((d_cpa-5)/3))+1) * (1-(1/(5**(3-t_cpa))))
                dominoTerm.append(dominoValue)
            dominoTerm_sum = -sum(dominoTerm)  # use -ve to indicate a penalty

            # check whether current actions leads to a collision with any buildings in the airspace
            allBuildingSTR = STRtree(self.world_map_2D_polyList[0][0])
            possiblePoly = allBuildingSTR.query(host_passed_volume)
            for element in possiblePoly:
                if allBuildingSTR.geometries.take(element).intersection(host_passed_volume):
                    collide_building = 1
                    print("drone_{} crash into building when moving from {} to {} at time step {}".format(drone_idx, self.all_agents[drone_idx].pre_pos, self.all_agents[drone_idx].pos, current_ts))
                    break

            tar_circle = Point(self.all_agents[drone_idx].goal[0]).buffer(1, cap_style='round')
            # when there is no intersection between two geometries, "RuntimeWarning" will appear
            # RuntimeWarning is, "invalid value encountered in intersection"
            goal_cur_intru_intersect = host_passed_volume.intersection(tar_circle)

            # if len(pc_after) > 0:
            #     print("debug")
            #     matplotlib.use('TkAgg')
            #     plt.ion()
            #     # Create figure and axis objects
            #     fig, ax = plt.subplots()
            #     ax.set_xlim(self.bound[0], self.bound[1])  # Set x-axis limits from 0 to 5
            #     ax.set_ylim(self.bound[2], self.bound[3])  # Set y-axis limits from 0 to 5
            #     for one_poly in self.world_map_2D_polyList[0][0]:
            #         one_poly_mat = shapelypoly_to_matpoly(one_poly, True, 'y', 'b')
            #         ax.add_patch(one_poly_mat)
            #     # plot current host drone
            #     ax.text(drone_obj.pos[0], drone_obj.pos[1], drone_obj.agent_name)
            #     self_circle = Point(drone_obj.pos[0],
            #                         drone_obj.pos[1]).buffer(drone_obj.protectiveBound, cap_style='round')
            #     ax.arrow(drone_obj.pre_pos[0], drone_obj.pre_pos[1], drone_obj.pos[0]-drone_obj.pre_pos[0], drone_obj.pos[1]-drone_obj.pre_pos[1], head_width=0.05, head_length=0.1, fc='r', ec='k')
            #     grid_mat_Scir = shapelypoly_to_matpoly(self_circle, inFill=False, Edgecolor='r')
            #
            #     ax.add_patch(grid_mat_Scir)
            #     # for drones that are possible having conflict
            #     for neigh_key in pc_after.keys():
            #         ax.text(self.all_agents[neigh_key].pos[0], self.all_agents[neigh_key].pos[1],
            #                 self.all_agents[neigh_key].agent_name)
            #         possible_conflict_circle = Point(self.all_agents[neigh_key].pos[0],
            #                                          self.all_agents[neigh_key].pos[1]).buffer(
            #             self.all_agents[neigh_key].protectiveBound, cap_style='round')
            #         ax.arrow(self.all_agents[neigh_key].pre_pos[0], self.all_agents[neigh_key].pre_pos[1], self.all_agents[neigh_key].pos[0]-self.all_agents[neigh_key].pre_pos[0], self.all_agents[neigh_key].pos[1]-self.all_agents[neigh_key].pre_pos[1],
            #                  head_width=0.05, head_length=0.1, fc='k', ec='k')
            #         grid_mat_Scir = shapelypoly_to_matpoly(possible_conflict_circle, inFill=False, Edgecolor='k')
            #         ax.add_patch(grid_mat_Scir)
            #     plt.show()
            #     print("done show")

            # ------------- pre-processed condition for a normal step -----------------
            # crossCoefficient = 0.1
            crossCoefficient = 1
            # goalCoefficient = 6
            goalCoefficient = 8
            dominoCoefficient = 1
            # cross track error term
            # cross_track_error = (20 / ((cross_track_deviation * cross_track_deviation) / 200 + 1)) - 3.5  # original
            cross_track_error = (math.e ** (5 - cross_track_deviation / 7) / 5) - 0.5  # original on 24_07
            # cross_track_error = (5 * math.e ** ((5 - cross_track_deviation) / 7)) - 1  # cross_track_deviation>16.266 -> <0, cross_track_deviation=0 -> 9.2
            # Distance between drone and its goal for two consecutive time step
            before_dist_hg = np.linalg.norm(drone_obj.pre_pos - drone_obj.goal[0])
            after_dist_hg = np.linalg.norm(drone_obj.pos - drone_obj.goal[0])  # distance to goal after action
            delta_hg = goalCoefficient * (before_dist_hg - after_dist_hg)
            if len(dist_toHost) == 0:  # meaning there is no neighbouring drone goes into host drone's detection range
                slowChanging_dist_penalty_others = 0
            else:
                # sort dist_toHost()
                # nearest
                dist_to_host_minimum = sorted(dist_toHost)[0]
                # slowChanging_dist_penalty_others = 1 * (-10 * math.exp((5 - dist_to_host_minimum) / 2))
                slowChanging_dist_penalty_others = -(math.e ** (5 - dist_to_host_minimum / 7) / 5)  # from -14.5 to 0

            # a small penalty for discourage the agent to stay in one single spot
            if (before_dist_hg - after_dist_hg) <= 2:
                small_step_penalty = 50
            else:
                small_step_penalty = 0
            alive_penalty = -60
            # -------------end of pre-processed condition for a normal step -----------------

            # exceed bound or crash into buildings or crash with other neighbors
            if collide_building == 1:
                reward.append(np.array(crash_penalty))
                done.append(True)
                # done.append(False)
            elif len(collision_drones) > 0:
                reward.append(np.array(crash_penalty))
                done.append(True)

            # exceed bound condition, don't use current point, use current circle or else will have condition that
            elif x_left_bound.intersects(host_passed_volume) or x_right_bound.intersects(host_passed_volume) or y_bottom_bound.intersects(host_passed_volume) or y_top_bound.intersects(host_passed_volume):
                print("drone_{} has crash into boundary at time step {}".format(drone_idx, current_ts))
                reward.append(np.array(crash_penalty))
                done.append(True)
                # done.append(False)

            elif not goal_cur_intru_intersect.is_empty:  # reached goal?

                if len(drone_obj.goal) > 1:  # meaning the current agent has more than one target/goal
                    print("drone_{} has reached its way point at time step {}".format(drone_idx, current_ts))
                    drone_obj.reach_target = False  # reset this flag
                    drone_obj.removed_goal = drone_obj.goal.pop(0)
                    # normal_step_rw = crossCoefficient*cross_track_error + slowChanging_dist_penalty_others + alive_penalty
                    # normal_step_rw = crossCoefficient * cross_track_error + delta_hg + alive_penalty
                    # normal_step_rw = crossCoefficient*cross_track_error + delta_hg + alive_penalty + slowChanging_dist_penalty_others
                    normal_step_rw = crossCoefficient*cross_track_error + delta_hg + alive_penalty + dominoCoefficient*dominoTerm_sum
                    reward.append(np.array(normal_step_rw))
                    # below two lines are used to debug
                    # reward.append(np.array(reach_target))
                    # agent_to_remove.append(drone_idx)  # NOTE: drone_idx is the key value.
                else:
                    check_goal[reward_record_idx] = True  # drone_obj.reach_target = True, and "check_goal" list also don't track agent's index
                    # normal_step_rw = crossCoefficient * cross_track_error + delta_hg
                    reward.append(np.array(reach_target))
                    # reward.append(np.array(normal_step_rw))
                    print("drone_{} has reached its final goal at time step {}".format(drone_idx, current_ts))
                    agent_to_remove.append(drone_idx)  # NOTE: drone_idx is the key value.
                if all(check_goal):
                    done.append(True)
                else:
                    done.append(False)
                # done.append(False)
            else:  # a normal step taken
                step_reward =crossCoefficient*cross_track_error + delta_hg + alive_penalty + dominoCoefficient*dominoTerm_sum

                # we remove the above termination condition
                done.append(False)

                step_reward = np.array(step_reward)
                reward.append(step_reward)
                # for debug, record the reward
                one_step_reward = [crossCoefficient*cross_track_error, delta_hg, alive_penalty, dominoCoefficient*dominoTerm_sum]
                step_reward_record[reward_record_idx] = one_step_reward
            reward_record_idx = reward_record_idx + 1

        # # we remove the reached agent here
        # for remove_idx in agent_to_remove:  # removed_idx is the key for the removed agent
        #     removed_value = self.all_agents.pop(remove_idx)
        #     # check all other agent's current surrounding neighbours, remove the current agent from their surroundingNeighbors
        #     for drone_idx, drone_obj in self.all_agents.items():  # this for loop already exclude the removed agent
        #         if remove_idx in drone_obj.surroundingNeighbor:
        #             del drone_obj.surroundingNeighbor[remove_idx]
        #         if remove_idx in drone_obj.pre_surroundingNeighbor:  # remember to remove the previous surrounding neighbours as well.
        #             del drone_obj.pre_surroundingNeighbor[remove_idx]
        #
        # # immediately add a dummy agent to recover the removed agent, we need to prevent the case where all agents reaches their destination, then the environmnet become empty
        # num_lack = len(agent_to_remove)
        # agent_filled = []  # contains the key of the newly added agents
        # for i in range(num_lack):  # the num_lack equals the number of times this for loop will execute
        #     # when both drones / all drones reaches the goal at same ts. Will have error. self.all_agents.keys() will become zeros. Debug this problem.
        #     agent = deepcopy(self.dummy_agent)
        #     if len(self.all_agents) == 0:
        #         current_max = max(agent_to_remove)  # all agents have reaches their destination at same time
        #     else:
        #         try:
        #             current_max = max(max(list(self.all_agents.keys())), max(agent_to_remove))  # new agent_idx to be added can the maximum of the remove agent or the maximum of the remining agent, we take whichever is higher.
        #         except:
        #             print("pause")
        #     agent.agent_name = 'agent_%s' % str(current_max + 1)
        #     self.all_agents[current_max + 1] = agent
        #     agent_filled.append(current_max + 1)
        # return reward, done, check_goal, step_reward_record, agent_filled
        return reward, done, check_goal, step_reward_record

    def ss_reward(self, current_ts, step_reward_record, step_collision_record, xy, full_observable_critic_flag, args, evaluation_by_episode, own_obs_only):
        bound_building_check = [False] * 4
        eps_status_holder = [{} for _ in range(len(self.all_agents))]
        reward, done = [], []
        agent_to_remove = []
        one_step_reward = []
        check_goal = [False] * len(self.all_agents)
        # previous_ever_reached = [agent.reach_target for agent in self.all_agents.values()]
        reward_record_idx = 0  # this is used as a list index, increase with for loop. No need go with agent index, this index is also shared by done checking
        # crash_penalty_wall = 5
        # crash_penalty_wall = 15
        crash_penalty_wall = 20
        # crash_penalty_wall = 100
        big_crash_penalty_wall = 200
        crash_penalty_drone = 1
        # reach_target = 1
        # reach_target = 5
        reach_target = 20
        survival_penalty = 0
        move_after_reach = -2

        potential_conflict_count = 0
        final_goal_toadd = 0
        fixed_domino_reward = 1
        x_left_bound = LineString([(self.bound[0], -9999), (self.bound[0], 9999)])
        x_right_bound = LineString([(self.bound[1], -9999), (self.bound[1], 9999)])
        y_bottom_bound = LineString([(-9999, self.bound[2]), (9999, self.bound[2])])
        y_top_bound = LineString([(-9999, self.bound[3]), (9999, self.bound[3])])
        dist_to_goal = 0  # initialize
        for drone_idx, drone_obj in self.all_agents.items():
            if xy[0] is not None and xy[1] is not None and drone_idx > 0:
                continue
            if xy[0] is not None and xy[1] is not None:
                drone_obj.pos = np.array([xy[0], xy[1]])
                drone_obj.pre_pos = drone_obj.pos

            # ------- small step penalty calculation -------
            # no penalty if current spd is larger than drone's radius per time step.
            # norm_rx = (drone_obj.protectiveBound*math.cos(drone_obj.heading))*self.normalizer.x_scale
            # norm_ry = (drone_obj.protectiveBound*math.sin(drone_obj.heading))*self.normalizer.y_scale
            # norm_r = math.sqrt(norm_rx**2 + norm_ry**2)

            drone_status_record = []
            one_agent_reward_record = []
            # re-initialize these two list for individual agents at each time step,this is to ensure collision
            # condition is reset for each agent at every time step
            collision_drones = []
            collide_building = 0
            pc_before, pc_after = [], []
            dist_toHost = []
            # we assume the maximum potential conflict the current drone could have at each time step is equals
            # to the total number of its neighbour at each time step
            pc_max_before = len(drone_obj.pre_surroundingNeighbor)
            pc_max_after = len(drone_obj.surroundingNeighbor)

            # calculate the deviation from the reference path after an action has been taken
            curPoint = Point(self.all_agents[drone_idx].pos)
            if isinstance(self.all_agents[drone_idx].removed_goal, np.ndarray):
                host_refline = LineString([self.all_agents[drone_idx].removed_goal, self.all_agents[drone_idx].goal[0]])
            else:
                host_refline = LineString([self.all_agents[drone_idx].ini_pos, self.all_agents[drone_idx].goal[0]])

            cross_track_deviation = curPoint.distance(host_refline)  # THIS IS WRONG
            # cross_track_deviation_x = abs(cross_track_deviation*math.cos(drone_obj.heading))
            # cross_track_deviation_y = abs(cross_track_deviation*math.sin(drone_obj.heading))
            # norm_cross_track_deviation_x = cross_track_deviation_x * self.normalizer.x_scale
            # norm_cross_track_deviation_y = cross_track_deviation_y * self.normalizer.y_scale

            host_pass_line = LineString([self.all_agents[drone_idx].pre_pos, self.all_agents[drone_idx].pos])
            host_passed_volume = host_pass_line.buffer(self.all_agents[drone_idx].protectiveBound, cap_style='round')
            host_current_circle = Point(self.all_agents[drone_idx].pos[0], self.all_agents[drone_idx].pos[1]).buffer(
                self.all_agents[drone_idx].protectiveBound)
            host_current_point = Point(self.all_agents[drone_idx].pos[0], self.all_agents[drone_idx].pos[1])

            # loop through neighbors from current time step, and search for the nearest neighbour and its neigh_keys
            nearest_neigh_key = None
            immediate_collision_neigh_key = None
            immediate_tcpa = math.inf
            immediate_d_tcpa = math.inf
            shortest_neigh_dist = math.inf
            cur_total_possible_conflict = 0
            pre_total_possible_conflict = 0
            all_neigh_dist = []
            for neigh_keys in self.all_agents[drone_idx].surroundingNeighbor:
                # calculate current t_cpa/d_cpa
                tcpa, d_tcpa, cur_total_possible_conflict = compute_t_cpa_d_cpa_potential_col(
                    self.all_agents[neigh_keys].pos, drone_obj.pos, self.all_agents[neigh_keys].vel, drone_obj.vel,
                    self.all_agents[neigh_keys].protectiveBound, drone_obj.protectiveBound, cur_total_possible_conflict)
                # calculate previous t_cpa/d_cpa
                pre_tcpa, pre_d_tcpa, pre_total_possible_conflict = compute_t_cpa_d_cpa_potential_col(
                    self.all_agents[neigh_keys].pre_pos, drone_obj.pre_pos, self.all_agents[neigh_keys].pre_vel,
                    drone_obj.pre_vel, self.all_agents[neigh_keys].protectiveBound, drone_obj.protectiveBound,
                    pre_total_possible_conflict)

                # find the neigh that has the highest collision probability at current step
                if tcpa >= 0 and tcpa < immediate_tcpa:  # tcpa -> +ve
                    immediate_tcpa = tcpa
                    immediate_d_tcpa = d_tcpa
                    immediate_collision_neigh_key = neigh_keys
                elif tcpa == -10:  # tcpa equals to special number, -10, meaning two drone relative velocity equals to 0
                    if d_tcpa < immediate_tcpa: # if currently relative velocity equals to 0, we move on to check their current relative distance
                        immediate_tcpa = tcpa  # indicate current neigh has a 0 relative velocity
                        immediate_d_tcpa = d_tcpa
                        immediate_collision_neigh_key = neigh_keys
                else:  # tcpa -> -ve, don't have collision risk, no need to update "immediate_tcpa"
                    pass

                # ---- start of make nei invis when nei has reached their goal ----
                # check if this drone reached their goal yet
                cur_nei_circle = Point(self.all_agents[neigh_keys].pos[0],
                                            self.all_agents[neigh_keys].pos[1]).buffer(self.all_agents[neigh_keys].protectiveBound)

                cur_nei_tar_circle = Point(self.all_agents[neigh_keys].goal[-1]).buffer(1,
                                                                               cap_style='round')  # set to [-1] so there are no more reference path
                # when there is no intersection between two geometries, "RuntimeWarning" will appear
                # RuntimeWarning is, "invalid value encountered in intersection"
                neigh_goal_intersect = cur_nei_circle.intersection(cur_nei_tar_circle)
                if args.mode == 'eval' and evaluation_by_episode == False:
                    if not neigh_goal_intersect.is_empty:  # current neigh has reached their goal
                        continue  # straight away pass this neigh which has already reached.

                # ---- end of make nei invis when nei has reached their goal ----

                # get distance from host to all the surrounding vehicles
                diff_dist_vec = drone_obj.pos - self.all_agents[neigh_keys].pos  # host pos vector - intruder pos vector
                euclidean_dist_diff = np.linalg.norm(diff_dist_vec)
                all_neigh_dist.append(euclidean_dist_diff)
                if euclidean_dist_diff < shortest_neigh_dist:
                    shortest_neigh_dist = euclidean_dist_diff
                    nearest_neigh_key = neigh_keys
                if np.linalg.norm(diff_dist_vec) <= drone_obj.protectiveBound * 2:
                    if args.mode == 'eval' and evaluation_by_episode == False:
                        if self.all_agents[neigh_keys].drone_collision == True \
                                or self.all_agents[neigh_keys].building_collision == True \
                                or self.all_agents[neigh_keys].bound_collision == True:
                            continue  # pass this neigh if this neigh is at its terminal condition
                        else:
                            print("host drone_{} collide with drone_{} at time step {}".format(drone_idx, neigh_keys,
                                                                                               current_ts))
                            collision_drones.append(neigh_keys)
                            drone_obj.drone_collision = True
                            self.all_agents[neigh_keys].drone_collision = True
                    else:
                        print("host drone_{} collide with drone_{} at time step {}".format(drone_idx, neigh_keys, current_ts))
                        collision_drones.append(neigh_keys)
                        drone_obj.drone_collision = True
            # loop over all previous step neighbour, check if the collision at current step, is done by the drones that is previous within the closest two neighbors
            neigh_count = 0
            flag_previous_nearest_two = 0
            for neigh_keys in self.all_agents[drone_idx].pre_surroundingNeighbor:
                for collided_drone_keys in collision_drones:
                    if collided_drone_keys == neigh_keys:
                        flag_previous_nearest_two = 1
                        break
                neigh_count = neigh_count + 1
                if neigh_count > 1:
                    break

            # check whether current actions leads to a collision with any buildings in the airspace

            # -------- check collision with building V1-------------
            start_of_v1_time = time.time()
            v1_decision = 0
            possiblePoly = self.allbuildingSTR.query(host_current_circle)
            for element in possiblePoly:
                if self.allbuildingSTR.geometries.take(element).intersection(host_current_circle):
                    collide_building = 1
                    v1_decision = collide_building
                    drone_obj.collide_wall_count = drone_obj.collide_wall_count + 1
                    drone_obj.building_collision = True
                    # print("drone_{} crash into building when moving from {} to {} at time step {}".format(drone_idx, self.all_agents[drone_idx].pre_pos, self.all_agents[drone_idx].pos, current_ts))
                    break
            end_v1_time = (time.time() - start_of_v1_time)*1000*1000
            # print("check building collision V1 time used is {} micro".format(end_v1_time))
            # -----------end of check collision with building v1 ---------

            end_v2_time, end_v3_time, v2_decision, v3_decision = 0, 0, 0, 0,
            step_collision_record[drone_idx].append([end_v1_time, end_v2_time, end_v3_time,
                                                     v1_decision, v2_decision, v3_decision])
            # if step_collision_record[drone_idx] == None:
            #     step_collision_record[drone_idx] = [[end_v1_time, end_v2_time, end_v3_time,
            #                                          v1_decision, v2_decision, v3_decision]]
            # else:
            #     step_collision_record[drone_idx].append([end_v1_time, end_v2_time, end_v3_time,
            #                                              v1_decision, v2_decision, v3_decision])

            # tar_circle = Point(self.all_agents[drone_idx].goal[0]).buffer(1, cap_style='round')
            tar_circle = Point(self.all_agents[drone_idx].goal[-1]).buffer(1, cap_style='round')  # set to [-1] so there are no more reference path
            # when there is no intersection between two geometries, "RuntimeWarning" will appear
            # RuntimeWarning is, "invalid value encountered in intersection"
            goal_cur_intru_intersect = host_current_circle.intersection(tar_circle)

            # wp_circle = Point(self.all_agents[drone_idx].goal[0]).buffer(1, cap_style='round')
            # wp_circle = Point(self.all_agents[drone_idx].goal[0]).buffer(drone_obj.protectiveBound,
            #                                                              cap_style='round')
            # wp_circle = Point(self.all_agents[drone_idx].goal[0]).buffer(3.5, cap_style='round')
            # wp_intersect = host_current_circle.intersection(wp_circle)
            wp_reach_threshold_dist = 5
            # --------------- a new way to check for the next wp --------------------
            # smallest_dist = math.inf
            # wp_intersect_flag = False
            # for wpidx, wp in enumerate(self.all_agents[drone_idx].goal):
            #     cur_dist_to_wp = curPoint.distance(Point(wp))
            #     if cur_dist_to_wp < smallest_dist:
            #         smallest_dist = cur_dist_to_wp
            #         next_wp = np.array(wp)
            #         if smallest_dist < wp_reach_threshold_dist:
            #             wp_intersect_flag = True
            #             # we find the next wp, as long as it is not the last wp
            #             if len(self.all_agents[drone_idx].goal) > 1:
            #                 drone_obj.removed_goal = drone_obj.goal.pop(wpidx)  # remove current wp
            #                 points_list = [Point(coord) for coord in self.all_agents[drone_idx].goal]
            #                 next_wPoint = min(points_list, key=lambda point: point.distance(curPoint))
            #                 next_wp = np.array([next_wPoint.x, next_wPoint.y])
            #             break  # once the nearest wp is found we break out of the loop
            # ---------------end of a new way to check for the next wp --------------------

            #  ------  using sequence wp reaching method ----------
            cur_dist_to_wp = curPoint.distance(Point(self.all_agents[drone_idx].waypoints[0]))
            next_wp = np.array(self.all_agents[drone_idx].waypoints[0])

            if cur_dist_to_wp < wp_reach_threshold_dist:
                wp_intersect_flag = True
            else:
                wp_intersect_flag = False
            # ------ end of using sequence wp reaching method ----------

            # ------------- pre-processed condition for a normal step -----------------
            # rew = 3
            rew = 0
            dist_to_goal_coeff = 1
            # dist_to_goal_coeff = 3
            # dist_to_goal_coeff = 1
            # dist_to_goal_coeff = 0
            # dist_to_goal_coeff = 2

            x_norm, y_norm = self.normalizer.nmlz_pos(drone_obj.pos)
            tx_norm, ty_norm = self.normalizer.nmlz_pos(drone_obj.goal[-1])
            # dist_to_goal = dist_to_goal_coeff * math.sqrt(((x_norm-tx_norm)**2 + (y_norm-ty_norm)**2))  # 0~2.828 at each step

            # ---- leading to goal reward V4 ---- 
            # before_dist_hg = np.linalg.norm(drone_obj.pre_pos - drone_obj.goal[-1])  # distance to goal before action
            # # before_dist_hg = np.linalg.norm(drone_obj.pre_pos - next_wp)  # distance to goal before action
            # after_dist_hg = np.linalg.norm(drone_obj.pos - drone_obj.goal[-1])  # distance to goal after action
            # # after_dist_hg = np.linalg.norm(drone_obj.pos - next_wp)  # distance to goal after action
            # dist_to_goal = dist_to_goal_coeff * (before_dist_hg - after_dist_hg)
            # dist_to_goal = dist_to_goal / drone_obj.maxSpeed  # perform a normalization
            # ---- end of leading to goal reward V4 ----

            # ---- V5 euclidean distance ----
            # dist_away = np.linalg.norm(drone_obj.ini_pos - drone_obj.goal[-1])
            # after_dist_hg = np.linalg.norm(drone_obj.pos - drone_obj.goal[-1])  # distance to goal after action
            # if after_dist_hg > dist_away:
            #     dist_to_goal = dist_to_goal_coeff * 0
            # else:
            #     dist_to_goal = dist_to_goal_coeff * (1-after_dist_hg/dist_away)
            # ---- end of V5 -------

            # ---- v6 denominator is diagonal of the map ---
            dist_away = 2*math.sqrt(2)  # maximum diagonal distance after normalization
            norm_pos = self.normalizer.nmlz_pos(drone_obj.pos)
            norm_goal = self.normalizer.nmlz_pos(drone_obj.goal[-1])
            after_dist_hg = np.linalg.norm(norm_pos - norm_goal)  # distance to goal after action
            dist_to_goal = dist_to_goal_coeff * (1 - after_dist_hg / dist_away)
            # ---- end of v6 -----

            # ----- v4 accumulative ---
            # one_drone_dist_to_goal = dist_to_goal_coeff * (before_dist_hg - after_dist_hg)  # (before_dist_hg - after_dist_hg) -max_vel - max_vel
            # one_drone_dist_to_goal = one_drone_dist_to_goal / drone_obj.maxSpeed  # perform a normalization
            # dist_to_goal = dist_to_goal + one_drone_dist_to_goal
            # ------ end of v4 accumulative----


            # dist_left = total_length_to_end_of_line(drone_obj.pos, drone_obj.ref_line)
            # dist_to_goal = dist_to_goal_coeff * (1 - (dist_left / drone_obj.ref_line.length))  # v1

            # ---- v2 leading to goal reward, based on compute_projected_velocity ---
            # projected_velocity = compute_projected_velocity(drone_obj.vel, drone_obj.ref_line, Point(drone_obj.pos))
            # get the norm as the projected_velocity.
            # dist_to_goal = dist_to_goal_coeff * np.linalg.norm(projected_velocity)
            # ---- end of v2 leading to goal reward, based on compute_projected_velocity ---

            # ---- v3 leading to goal reward, based on remained distance to travel only ---
            # dist_left = total_length_to_end_of_line_without_cross(drone_obj.pos, drone_obj.ref_line)
            # dist_to_goal = dist_to_goal_coeff * (1 - (dist_left / drone_obj.ref_line.length))  # v3
            # ---- end of v3 leading to goal reward, based on remained distance to travel only ---

            if dist_to_goal > drone_obj.maxSpeed:
                print("dist_to_goal reward out of range")

            # ------- small segment reward ------------
            # dist_to_seg_coeff = 10
            # dist_to_seg_coeff = 1
            dist_to_seg_coeff = 0

            # if drone_obj.removed_goal == None:
            #     total_delta_seg_vector = np.linalg.norm((drone_obj.ini_pos - np.array(drone_obj.goal[0])))
            # else:
            #     total_delta_seg_vector = np.linalg.norm((np.array(drone_obj.removed_goal) - np.array(drone_obj.goal[0])))
            # delta_seg_vector = drone_obj.pos - drone_obj.goal[0]
            # dist_seg_vector = np.linalg.norm(delta_seg_vector)
            # if dist_seg_vector / total_delta_seg_vector <= 1:  # we reward the agent
            #     seg_reward = dist_to_seg_coeff * (dist_seg_vector / total_delta_seg_vector)
            # else:
            #     seg_reward = dist_to_seg_coeff * (-1)*(dist_seg_vector / total_delta_seg_vector)

            # s_tx_norm, s_ty_norm = self.normalizer.nmlz_pos(drone_obj.goal[0])
            # seg_reward = dist_to_seg_coeff * math.sqrt(((x_norm-s_tx_norm)**2 + (y_norm-s_ty_norm)**2))  # 0~2.828 at each step
            seg_reward = dist_to_seg_coeff * 0
            # -------- end of small segment reward ----------

            # dist_to_goal = 0
            # coef_ref_line = 0.5
            # coef_ref_line = -10
            # coef_ref_line = 3
            # coef_ref_line = 1
            # coef_ref_line = 2
            # coef_ref_line = 1.5
            coef_ref_line = 0
            cross_err_distance, x_error, y_error, nearest_pt = self.cross_track_error(host_current_point, drone_obj.ref_line)  # deviation from the reference line, cross track error
            norm_cross_track_deviation_x = x_error * self.normalizer.x_scale
            norm_cross_track_deviation_y = y_error * self.normalizer.y_scale
            # dist_to_ref_line = coef_ref_line*math.sqrt(norm_cross_track_deviation_x ** 2 +
            #                                            norm_cross_track_deviation_y ** 2)

            if cross_err_distance <= drone_obj.protectiveBound:
                # linear increase in reward
                m = (0 - 1) / (drone_obj.protectiveBound - 0)
                dist_to_ref_line = coef_ref_line*(m * cross_err_distance + 1)  # 0~1*coef_ref_line
                # dist_to_ref_line = (coef_ref_line*(m * cross_err_distance + 1)) + coef_ref_line  # 0~1*coef_ref_line, with a fixed reward
            else:
                dist_to_ref_line = -coef_ref_line*1
                # dist_to_ref_line = -coef_ref_line*3
                # dist_to_ref_line = -coef_ref_line*0

            # ------- penalty for surrounding agents as a whole -----
            surrounding_collision_penalty = 0
            # if pre_total_possible_conflict < cur_total_possible_conflict:
            #     surrounding_collision_penalty = 2
            # ------- end of reward for surrounding agents as a whole ----

            # ----- start of near drone penalty ----------------
            # near_drone_penalty_coef = 10
            # # near_drone_penalty_coef = 5
            # # near_drone_penalty_coef = 1
            # # near_drone_penalty_coef = 3
            # # near_drone_penalty_coef = 0
            # # dist_to_penalty_upperbound = 6
            # dist_to_penalty_upperbound = 10
            # dist_to_penalty_lowerbound = 2.5
            # # assume when at lowerbound, y = 1
            # c_drone = 1 + (dist_to_penalty_lowerbound / (dist_to_penalty_upperbound - dist_to_penalty_lowerbound))
            # m_drone = (0 - 1) / (dist_to_penalty_upperbound - dist_to_penalty_lowerbound)
            # if nearest_neigh_key is not None:
            #     if shortest_neigh_dist >= dist_to_penalty_lowerbound and shortest_neigh_dist <= dist_to_penalty_upperbound:
            #         near_drone_penalty = near_drone_penalty_coef * (m_drone * shortest_neigh_dist + c_drone)
            #     else:
            #         near_drone_penalty = near_drone_penalty_coef * 0
            # else:
            #     near_drone_penalty = near_drone_penalty_coef * 0
            # -----end of near drone penalty ----------------

            # ----- start of SUM near drone penalty ----------------
            # near_drone_penalty_coef = 10
            if own_obs_only:
                near_drone_penalty_coef = 0
            else:
                near_drone_penalty_coef = 1
            # near_drone_penalty_coef = 5
            # near_drone_penalty_coef = 1
            # near_drone_penalty_coef = 3
            # near_drone_penalty_coef = 0
            # dist_to_penalty_upperbound = 6
            dist_to_penalty_upperbound = 10
            # dist_to_penalty_upperbound = 20
            dist_to_penalty_lowerbound = drone_obj.protectiveBound
            # assume when at lowerbound, y = 1
            near_drone_penalty = 0  # initialize
            c_drone = 1 + (dist_to_penalty_lowerbound / (dist_to_penalty_upperbound - dist_to_penalty_lowerbound))
            m_drone = (0 - 1) / (dist_to_penalty_upperbound - dist_to_penalty_lowerbound)
            if len(all_neigh_dist) == 0:
                near_drone_penalty = near_drone_penalty + near_drone_penalty_coef * 0
            else:
                for individual_nei_dist in all_neigh_dist:
                    if individual_nei_dist >= dist_to_penalty_lowerbound and individual_nei_dist <= dist_to_penalty_upperbound:
                        # normalize distance to 0-1
                        norm_ind_nei_dist = (individual_nei_dist-dist_to_penalty_lowerbound) / (dist_to_penalty_upperbound-dist_to_penalty_lowerbound)
                        near_drone_penalty = near_drone_penalty + (norm_ind_nei_dist-1)**2
                    else:
                        near_drone_penalty = near_drone_penalty + near_drone_penalty_coef * 0

                    # if individual_nei_dist >= dist_to_penalty_lowerbound and individual_nei_dist <= dist_to_penalty_upperbound:
                    #     near_drone_penalty = near_drone_penalty + (near_drone_penalty_coef * (m_drone * individual_nei_dist + c_drone))
                    # else:
                    #     near_drone_penalty = near_drone_penalty + near_drone_penalty_coef * 0
            # -----end of near SUM drone penalty ----------------

            # ----- start of V2 nearest drone penalty ----------------
            # near_drone_penalty_coef = 1
            # dist_to_penalty_upperbound = 10
            # dist_to_penalty_lowerbound = 2.5
            # nearest_drone_dist = min(all_neigh_dist)
            # if nearest_drone_dist >= dist_to_penalty_lowerbound and nearest_drone_dist <= dist_to_penalty_upperbound:
            #     # normalize distance to 0-1
            #     norm_ind_nei_dist = (nearest_drone_dist - dist_to_penalty_lowerbound) / (
            #                 dist_to_penalty_upperbound - dist_to_penalty_lowerbound)
            #     near_drone_penalty = (norm_ind_nei_dist - 1) ** 2
            # else:
            #     near_drone_penalty = near_drone_penalty_coef * 0
            # -----end of V2 nearest drone penalty ----------------

            # ---- start of V3 near drone penalty -------
            # if immediate_collision_neigh_key is None:
            #     near_drone_penalty = near_drone_penalty_coef * 0
            # else:
            #     if immediate_tcpa >= 0:
            #         near_drone_penalty = near_drone_penalty_coef * math.exp(-(immediate_tcpa-1)/2)  # 10: 0~16.487
            #     elif immediate_tcpa == -10:
            #         near_drone_penalty = near_drone_penalty_coef * math.exp((5 - (2 * immediate_d_tcpa)) / 5)  # 10: 0~27.183
            # ----- end of V3 near drone penalty -------


            # small_step_penalty_coef = 5
            small_step_penalty_coef = 0
            spd_penalty_threshold = 2*drone_obj.protectiveBound
            # spd_penalty_threshold = drone_obj.protectiveBound
            small_step_penalty_val = (spd_penalty_threshold -
                                  np.clip(np.linalg.norm(drone_obj.vel), 0, spd_penalty_threshold))*\
                                 (1.0 / spd_penalty_threshold)  # between 0-1.
            small_step_penalty = small_step_penalty_coef * small_step_penalty_val

            # dist_moved = np.linalg.norm(drone_obj.pos - drone_obj.pre_pos)
            # if dist_moved <= 1:
            #     small_step_penalty = small_step_penalty_coef * 1
            # else:
            #     small_step_penalty = small_step_penalty_coef * 0

            # near_goal_coefficient = 3  # so that near_goal_reward will become 0-3 instead of 0-1
            near_goal_coefficient = 0
            near_goal_threshold = drone_obj.detectionRange
            actual_after_dist_hg = math.sqrt(((drone_obj.pos[0] - drone_obj.goal[-1][0]) ** 2 +
                                              (drone_obj.pos[1] - drone_obj.goal[-1][1]) ** 2))
            near_goal_reward = near_goal_coefficient * ((near_goal_threshold -
                                np.clip(actual_after_dist_hg, 0, near_goal_threshold)) * 1.0/near_goal_threshold)

            # penalty for any buildings are getting too near to the host agent
            turningPtConst = drone_obj.detectionRange/2-drone_obj.protectiveBound  # this one should be 12.5
            # dist_array = np.array([dist_info[0] for dist_info in drone_obj.observableSpace])  # used when radar detect other uavs
            if len(drone_obj.observableSpace) == 18:
                dist_array = np.array([dist_info for dist_info in drone_obj.observableSpace])
            else:
                probe_dist = []
                for i in range(0, len(drone_obj.observableSpace), 2):  # we are store position coordinate of the distance probe
                    # Access the current pair of elements, these elements are normalized difference position, is obtained using: norm_prob_delta_coord = norm_pos - norm_intersection_obstacle
                    norm_prob_delta_coord = drone_obj.observableSpace[i:i + 2]
                    if (norm_prob_delta_coord == [-2, -2]).all():
                        probe_dist.append(drone_obj.detectionRange/2)
                        continue
                    norm_intersection_obstacle = norm_pos - norm_prob_delta_coord
                    actual_prob_coord = self.normalizer.reverse_nmlz_pos(norm_intersection_obstacle)
                    dist = np.linalg.norm(actual_prob_coord - drone_obj.pos)
                    probe_dist.append(dist)
                dist_array = np.array(probe_dist)
            # dist_array = np.array([dist_info for dist_info in drone_obj.observableSpace])

            ascending_array = np.sort(dist_array)
            min_index = np.argmin(dist_array)
            min_dist = dist_array[min_index]
            # radar_status = drone_obj.observableSpace[min_index][-1]  # radar status for now not required

            # the distance is based on the minimum of the detected distance to surrounding buildings.
            # near_building_penalty_coef = 4
            if own_obs_only:
                near_building_penalty_coef = 0
            else:
                near_building_penalty_coef = 10
            # near_building_penalty_coef = 3
            # near_building_penalty_coef = 0

            near_building_penalty = 0  # initialize
            prob_counter = 0  # initialize
            # turningPtConst = 12.5
            # turningPtConst = 5
            turningPtConst = 10
            if turningPtConst == 12.5:
                c = 1.25
            elif turningPtConst == 5:
                c = 2

            c = 1 + (drone_obj.protectiveBound / (turningPtConst - drone_obj.protectiveBound))

            for dist_idx, dist in enumerate(ascending_array):
                # only consider the nearest 4 prob
                if dist_idx > 3:
                    continue
                # # linear building penalty
                # makesure only when min_dist is >=0 and <= turningPtConst, then we activate this penalty
                m = (0-1)/(turningPtConst-drone_obj.protectiveBound)  # we must consider drone's circle, because when min_distance is less than drone's radius, it is consider collision.
                # if dist>=drone_obj.protectiveBound and dist<=turningPtConst:  # only when min_dist is between 2.5~5, this penalty is working.
                #     near_building_penalty = near_building_penalty + near_building_penalty_coef*(m*dist+c)  # at each step, penalty from 3 to 0.
                # else:
                #     near_building_penalty = near_building_penalty + 0.0  # if min_dist is outside of the bound, other parts of the reward will be taking care.
                # non-linear building penalty
                if dist >= drone_obj.protectiveBound and dist <= turningPtConst:
                    norm_ind_nei_dist = (dist - drone_obj.protectiveBound) / (
                                turningPtConst - drone_obj.protectiveBound)
                    near_building_penalty = near_building_penalty + near_building_penalty_coef * \
                                            (1-norm_ind_nei_dist)**3
                else:
                    near_building_penalty = near_building_penalty + 0.0

            # if min_dist < drone_obj.protectiveBound:
            #     print("check for collision")
            # # (linear building penalty) same thing, another way of express
            # if min_dist < 2.5 or min_dist > turningPtConst:  # when min_dist is less than 2.5m, is consider collision, the collision penalty will take care of that
            #     near_building_penalty = 0
            # else:
            #     near_building_penalty = near_building_penalty_coef * \
            #                             ((min_dist-drone_obj.protectiveBound)/(turningPtConst-drone_obj.protectiveBound))

            # -------------end of pre-processed condition for a normal step -----------------
            #
            # Always check the boundary as the 1st condition, or else will encounter error where the agent crash into wall but also exceed the bound, but crash into wall did not stop the episode. So, we must put the check boundary condition 1st, so that episode can terminate in time and does not leads to exceed boundary with error in no polygon found.
            # exceed bound condition, don't use current point, use current circle or else will have condition that
            # must use "host_passed_volume", or else, we unable to confirm whether the host's circle is at left or right of the boundary lines
            if x_left_bound.intersects(host_passed_volume) or x_right_bound.intersects(
                    host_passed_volume) or y_bottom_bound.intersects(host_passed_volume) or y_top_bound.intersects(
                    host_passed_volume):
                bound_collision = True
            else:
                bound_collision = False

            if own_obs_only:
                collide_building = 0
                collision_drones = []
            
            if bound_collision:
                print("drone_{} has crash into boundary at time step {}".format(drone_idx, current_ts))
                drone_obj.bound_collision = True
                rew = rew - crash_penalty_wall
                if args.mode == 'eval' and evaluation_by_episode == False:
                    done.append(False)
                else:  # during training or evaluation by episode is TRUE
                    done.append(True)
                bound_building_check[0] = True
                # done.append(False)
                reward.append(np.array(rew))
            # # crash into buildings or crash with other neighbors
            elif collide_building == 1:
                if args.mode == 'eval' and evaluation_by_episode == False:
                    done.append(False)
                else:  # during training or evaluation by episode is TRUE
                    done.append(True)
                bound_building_check[1] = True
                rew = rew - crash_penalty_wall
                # rew = rew - big_crash_penalty_wall
                reward.append(np.array(rew))
            # # ---------- Termination only during collision to wall on the 3rd time -----------------------
            # elif drone_obj.collide_wall_count >0:
            #     if drone_obj.collide_wall_count == 1:
            #         done.append(False)
            #         rew = rew - dist_to_ref_line - crash_penalty_wall - dist_to_goal - small_step_penalty + near_goal_reward -5
            #         reward.append(np.array(rew))
            #     elif drone_obj.collide_wall_count == 2:
            #         done.append(False)
            #         rew = rew - dist_to_ref_line - crash_penalty_wall - dist_to_goal - small_step_penalty + near_goal_reward -15
            #         reward.append(np.array(rew))
            #     else:
            #         done.append(True)
            #         rew = rew - dist_to_ref_line - crash_penalty_wall - dist_to_goal - small_step_penalty + near_goal_reward - 20
            #         reward.append(np.array(rew))
            # # ----------End of termination only during collision to wall on the 3rd time -----------------------
            elif len(collision_drones) > 0:
                if args.mode == 'eval' and evaluation_by_episode == False:
                    done.append(False)
                else:  # during training or evaluation by episode is TRUE
                    done.append(True)
                # done.append(False)
                bound_building_check[2] = True
                rew = rew - crash_penalty_wall
                reward.append(np.array(rew))
                # check if the collision is due to the nearest drone.
                # if collision_drones[-1] == nearest_neigh_key:
                # check if the collision is due to the previous nearest two drone.
                if flag_previous_nearest_two:
                    bound_building_check[3] = True
            elif not goal_cur_intru_intersect.is_empty:  # reached goal?
                # --------------- with way point -----------------------
                drone_obj.reach_target = True
                check_goal[drone_idx] = True

                # print("drone_{} has reached its final goal at time step {}".format(drone_idx, current_ts))
                agent_to_remove.append(drone_idx)  # NOTE: drone_idx is the key value.
                rew = rew + reach_target + near_goal_reward
                reward.append(np.array(rew))
                done.append(False)
                # --------------- end of with way point -----------------------
                # without wap point
                # rew = rew + reach_target
                # reward.append(np.array(rew))
                # print("final goal has reached")
                # done.append(False)
            else:  # a normal step taken
                if xy[0] is None and xy[1] is None:  # we only alter drone's goal during actual training
                    # if (not wp_intersect.is_empty) and len(drone_obj.goal) > 1: # check if wp reached, and this is not the end point
                    if wp_intersect_flag and len(drone_obj.waypoints) > 1: # check if wp reached and don't remove last element
                        drone_obj.removed_goal = drone_obj.waypoints.pop(0)  # remove current wp
                        # we add a wp reached reward, this reward is equals to the maximum of the path deviation reward
                        # rew = rew + coef_ref_line
                        # print("drone {} has reached a WP on step {}, claim additional {} points of reward"
                        #       .format(drone_idx, current_ts, coef_ref_line))
                # if drone_obj.reach_target == False:
                #     rew = rew + dist_to_ref_line + dist_to_goal - \
                #           small_step_penalty + near_goal_reward - near_building_penalty + seg_reward-survival_penalty - near_drone_penalty
                # else:
                #     rew = rew + move_after_reach
                rew = rew + dist_to_ref_line + dist_to_goal - \
                      small_step_penalty + near_goal_reward - near_building_penalty + seg_reward \
                      - survival_penalty - near_drone_penalty - surrounding_collision_penalty
                # we remove the above termination condition
                # if current_ts >= args.episode_length:
                #     done.append(True)
                # else:
                #     done.append(False)
                done.append(False)
                step_reward = np.array(rew)
                reward.append(step_reward)
                # for debug, record the reward
                # one_step_reward = [crossCoefficient*cross_track_error, delta_hg, alive_penalty, dominoCoefficient*dominoTerm_sum]

                # if rew < 1:
                #     print("check")
            # if rew < 0.1 and rew >= 0:
            #     print("check")
            step_reward_record[drone_idx] = [dist_to_ref_line, rew]

            # print("current drone {} actual distance to goal is {}, current reward is {}".format(drone_idx, actual_after_dist_hg, reward[-1]))
            # print("current drone {} actual distance to goal is {}, current reward to gaol is {}, current ref line reward is {}, current step reward is {}".format(drone_idx, actual_after_dist_hg, dist_to_goal, dist_to_ref_line, rew))

            # record status of each step.
            eps_status_holder = self.display_one_eps_status(eps_status_holder, drone_idx, np.array(after_dist_hg),
                                                            [np.array(dist_to_goal), cross_err_distance, dist_to_ref_line,
                                                             np.array(near_building_penalty), small_step_penalty,
                                                             np.linalg.norm(drone_obj.vel), near_goal_reward,
                                                             seg_reward, nearest_pt, drone_obj.observableSpace,
                                                             drone_obj.heading, np.array(near_drone_penalty)])
            # overall_status_record[2].append()  # 3rd is accumulated reward till that step for each agent

        if full_observable_critic_flag:
            reward = [np.sum(reward) for _ in reward]

        # if all(check_goal):
        #     for element_idx, element in enumerate(done):
        #         done[element_idx] = True

        # ever_reached = [agent.reach_target for agent in self.all_agents.values()]
        # if check_goal.count(True) == 1 and ever_reached.count(True) == 0:
        #     reward = [ea_rw + 200 for ea_rw in reward]
        # elif check_goal.count(True) == 2 and ever_reached.count(True) == 1:
        #     reward = [ea_rw + 400 for ea_rw in reward]
        # elif check_goal.count(True) == 3 and ever_reached.count(True) == 2:
        #     reward = [ea_rw + 600 for ea_rw in reward]

        # all_reach_target = all(agent.reach_target == True for agent in self.all_agents.values())
        # if all_reach_target:  # in this episode all agents have reached their target at least one
        #     # we cannot just assign a single True to "done", as it must be a list to output from the function.
        #     done = [True, True, True]

        return reward, done, check_goal, step_reward_record, eps_status_holder, step_collision_record, bound_building_check

    def ss_reward_Mar(self, current_ts, step_reward_record, step_collision_record, xy, full_observable_critic_flag, args, evaluation_by_episode):
        bound_building_check = [False] * 4
        eps_status_holder = [{} for _ in range(len(self.all_agents))]
        reward, done = [], []
        agent_to_remove = []
        one_step_reward = []
        check_goal = [False] * len(self.all_agents)
        # previous_ever_reached = [agent.reach_target for agent in self.all_agents.values()]
        reward_record_idx = 0  # this is used as a list index, increase with for loop. No need go with agent index, this index is also shared by done checking
        # crash_penalty_wall = 5
        # crash_penalty_wall = 15
        crash_penalty_wall = 20
        # crash_penalty_wall = 100
        big_crash_penalty_wall = 200
        crash_penalty_drone = 1
        # reach_target = 1
        # reach_target = 5
        reach_target = 20
        survival_penalty = 0
        move_after_reach = -2

        potential_conflict_count = 0
        final_goal_toadd = 0
        fixed_domino_reward = 1
        x_left_bound = LineString([(self.bound[0], -9999), (self.bound[0], 9999)])
        x_right_bound = LineString([(self.bound[1], -9999), (self.bound[1], 9999)])
        y_bottom_bound = LineString([(-9999, self.bound[2]), (9999, self.bound[2])])
        y_top_bound = LineString([(-9999, self.bound[3]), (9999, self.bound[3])])
        dist_to_goal = 0  # initialize
        for drone_idx, drone_obj in self.all_agents.items():
            host_current_circle = Point(self.all_agents[drone_idx].pos[0], self.all_agents[drone_idx].pos[1]).buffer(
                self.all_agents[drone_idx].protectiveBound)
            tar_circle = Point(self.all_agents[drone_idx].goal[-1]).buffer(1, cap_style='round')
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                goal_cur_intru_intersect = host_current_circle.intersection(tar_circle)
            if not goal_cur_intru_intersect.is_empty:
                drone_obj.reach_target = True

        for drone_idx, drone_obj in self.all_agents.items():
            if xy[0] is not None and xy[1] is not None and drone_idx > 0:
                continue
            if xy[0] is not None and xy[1] is not None:
                drone_obj.pos = np.array([xy[0], xy[1]])
                drone_obj.pre_pos = drone_obj.pos

            # ------- small step penalty calculation -------
            # no penalty if current spd is larger than drone's radius per time step.
            # norm_rx = (drone_obj.protectiveBound*math.cos(drone_obj.heading))*self.normalizer.x_scale
            # norm_ry = (drone_obj.protectiveBound*math.sin(drone_obj.heading))*self.normalizer.y_scale
            # norm_r = math.sqrt(norm_rx**2 + norm_ry**2)

            drone_status_record = []
            one_agent_reward_record = []
            # re-initialize these two list for individual agents at each time step,this is to ensure collision
            # condition is reset for each agent at every time step
            collision_drones = []
            collide_building = 0
            pc_before, pc_after = [], []
            dist_toHost = []
            # we assume the maximum potential conflict the current drone could have at each time step is equals
            # to the total number of its neighbour at each time step
            pc_max_before = len(drone_obj.pre_surroundingNeighbor)
            pc_max_after = len(drone_obj.surroundingNeighbor)

            # calculate the deviation from the reference path after an action has been taken
            curPoint = Point(self.all_agents[drone_idx].pos)
            if isinstance(self.all_agents[drone_idx].removed_goal, np.ndarray):
                host_refline = LineString([self.all_agents[drone_idx].removed_goal, self.all_agents[drone_idx].goal[0]])
            else:
                host_refline = LineString([self.all_agents[drone_idx].ini_pos, self.all_agents[drone_idx].goal[0]])

            cross_track_deviation = curPoint.distance(host_refline)  # THIS IS WRONG
            # cross_track_deviation_x = abs(cross_track_deviation*math.cos(drone_obj.heading))
            # cross_track_deviation_y = abs(cross_track_deviation*math.sin(drone_obj.heading))
            # norm_cross_track_deviation_x = cross_track_deviation_x * self.normalizer.x_scale
            # norm_cross_track_deviation_y = cross_track_deviation_y * self.normalizer.y_scale

            host_pass_line = LineString([self.all_agents[drone_idx].pre_pos, self.all_agents[drone_idx].pos])
            host_passed_volume = host_pass_line.buffer(self.all_agents[drone_idx].protectiveBound, cap_style='round')
            host_current_circle = Point(self.all_agents[drone_idx].pos[0], self.all_agents[drone_idx].pos[1]).buffer(
                self.all_agents[drone_idx].protectiveBound)
            host_current_point = Point(self.all_agents[drone_idx].pos[0], self.all_agents[drone_idx].pos[1])

            # loop through neighbors from current time step, and search for the nearest neighbour and its neigh_keys
            nearest_neigh_key = None
            immediate_collision_neigh_key = None
            immediate_tcpa = math.inf
            immediate_d_tcpa = math.inf
            shortest_neigh_dist = math.inf
            cur_total_possible_conflict = 0
            pre_total_possible_conflict = 0
            all_neigh_dist = []
            neigh_relative_bearing = None
            neigh_collision_bearing = None
            previous_potential_conflict = []
            current_potential_conflict = []
            for neigh_keys in self.all_agents[drone_idx].surroundingNeighbor:
                # calculate current t_cpa/d_cpa
                tcpa, d_tcpa, _ = compute_t_cpa_d_cpa_potential_col(
                    self.all_agents[neigh_keys].pos, drone_obj.pos, self.all_agents[neigh_keys].vel, drone_obj.vel,
                    self.all_agents[neigh_keys].protectiveBound, drone_obj.protectiveBound, cur_total_possible_conflict)
                if tcpa >= 0 and tcpa < (2/self.time_step) and d_tcpa <= drone_obj.protectiveBound*2:
                    current_potential_conflict.append(neigh_keys)

                # calculate previous t_cpa/d_cpa
                pre_tcpa, pre_d_tcpa, _ = compute_t_cpa_d_cpa_potential_col(
                    self.all_agents[neigh_keys].pre_pos, drone_obj.pre_pos, self.all_agents[neigh_keys].pre_vel,
                    drone_obj.pre_vel, self.all_agents[neigh_keys].protectiveBound, drone_obj.protectiveBound,
                    pre_total_possible_conflict)
                if pre_tcpa >= 0 and pre_tcpa < (2/self.time_step) and d_tcpa <= drone_obj.protectiveBound*2:
                    previous_potential_conflict.append(neigh_keys)

                # tcpa is in seconds. Our time step is 0.5 second

                # find the neigh that has the highest collision probability at current step
                # if tcpa >= 0 and tcpa < immediate_tcpa:  # tcpa -> +ve
                #     immediate_tcpa = tcpa
                #     immediate_d_tcpa = d_tcpa
                #     immediate_collision_neigh_key = neigh_keys
                # elif tcpa == -10:  # tcpa equals to special number, -10, meaning two drone relative velocity equals to 0
                #     if d_tcpa < immediate_tcpa: # if currently relative velocity equals to 0, we move on to check their current relative distance
                #         immediate_tcpa = tcpa  # indicate current neigh has a 0 relative velocity
                #         immediate_d_tcpa = d_tcpa
                #         immediate_collision_neigh_key = neigh_keys
                # else:  # tcpa -> -ve, don't have collision risk, no need to update "immediate_tcpa"
                #     pass

                # obtain DEC
                dec_coeff = 1
                if len(previous_potential_conflict) == 0 and len(current_potential_conflict) == 0:
                    dec = 0
                elif len(previous_potential_conflict) == 0 and len(current_potential_conflict) != 0:
                    dec = 1
                else:
                    dec = (len(current_potential_conflict) - len(previous_potential_conflict)) / len(previous_potential_conflict)
                    if dec > 1:
                        dec = 1
                # dec_score
                dec_score = dec_coeff * dec

                # ---- start of make nei invis when nei has reached their goal ----
                # check if this drone reached their goal yet
                cur_nei_circle = Point(self.all_agents[neigh_keys].pos[0],
                                            self.all_agents[neigh_keys].pos[1]).buffer(self.all_agents[neigh_keys].protectiveBound)

                cur_nei_tar_circle = Point(self.all_agents[neigh_keys].goal[-1]).buffer(1,
                                                                               cap_style='round')  # set to [-1] so there are no more reference path
                # when there is no intersection between two geometries, "RuntimeWarning" will appear
                # RuntimeWarning is, "invalid value encountered in intersection"
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', category=RuntimeWarning)
                    neigh_goal_intersect = cur_nei_circle.intersection(cur_nei_tar_circle)
                if args.mode == 'eval' and evaluation_by_episode == False:
                    if not neigh_goal_intersect.is_empty:  # current neigh has reached their goal
                        continue  # straight away pass this neigh which has already reached.

                # ---- end of make nei invis when nei has reached their goal ----

                # get distance from host to all the surrounding vehicles
                diff_dist_vec = drone_obj.pos - self.all_agents[neigh_keys].pos  # host pos vector - intruder pos vector
                euclidean_dist_diff = np.linalg.norm(diff_dist_vec)


                if self.all_agents[neigh_keys].reach_target == True or drone_obj.reach_target == True:
                    euclidean_dist_diff = math.inf
                else:
                    all_neigh_dist.append(euclidean_dist_diff)

                if euclidean_dist_diff < shortest_neigh_dist:
                    shortest_neigh_dist = euclidean_dist_diff
                    neigh_relative_bearing = calculate_bearing(drone_obj.pos[0], drone_obj.pos[1],
                                                               self.all_agents[neigh_keys].pos[0], self.all_agents[neigh_keys].pos[1])
                    nearest_neigh_key = neigh_keys
                if np.linalg.norm(diff_dist_vec) <= drone_obj.protectiveBound * 2:
                    if args.mode == 'eval' and evaluation_by_episode == False:
                        neigh_collision_bearing = calculate_bearing(drone_obj.pos[0], drone_obj.pos[1],
                                                                   self.all_agents[neigh_keys].pos[0],
                                                                   self.all_agents[neigh_keys].pos[1])
                        if self.all_agents[neigh_keys].drone_collision == True \
                                or self.all_agents[neigh_keys].building_collision == True \
                                or self.all_agents[neigh_keys].reach_target == True \
                                or drone_obj.reach_target == True \
                                or drone_obj.building_collision == True \
                                or drone_obj.drone_collision == True \
                                or self.all_agents[neigh_keys].bound_collision == True:
                            continue  # pass this neigh if this neigh is at its terminal condition
                        else:
                            print("host drone_{} collide with drone_{} at time step {}".format(drone_idx, neigh_keys,
                                                                                               current_ts))
                            collision_drones.append(neigh_keys)
                            drone_obj.drone_collision = True
                            self.all_agents[neigh_keys].drone_collision = True
                    else:
                        if self.all_agents[neigh_keys].reach_target == True or drone_obj.reach_target==True:
                            pass
                        else:
                            print("host drone_{} collide with drone_{} at time step {}".format(drone_idx, neigh_keys, current_ts))
                            neigh_collision_bearing = calculate_bearing(drone_obj.pos[0], drone_obj.pos[1],
                                                                       self.all_agents[neigh_keys].pos[0],
                                                                       self.all_agents[neigh_keys].pos[1])
                            collision_drones.append(neigh_keys)
                            drone_obj.drone_collision = True
            # loop over all previous step neighbour, check if the collision at current step, is done by the drones that is previous within the closest two neighbors
            neigh_count = 0
            flag_previous_nearest_two = 0
            for neigh_keys in self.all_agents[drone_idx].pre_surroundingNeighbor:
                for collided_drone_keys in collision_drones:
                    if collided_drone_keys == neigh_keys:
                        flag_previous_nearest_two = 1
                        break
                neigh_count = neigh_count + 1
                if neigh_count > 1:
                    break

            # check whether current actions leads to a collision with any buildings in the airspace

            # -------- check collision with building V1-------------
            start_of_v1_time = time.time()
            v1_decision = 0
            possiblePoly = self.allbuildingSTR.query(host_current_circle)
            for element in possiblePoly:
                if self.allbuildingSTR.geometries.take(element).intersection(host_current_circle):
                    collide_building = 1
                    v1_decision = collide_building
                    drone_obj.collide_wall_count = drone_obj.collide_wall_count + 1
                    drone_obj.building_collision = True
                    # print("drone_{} crash into building when moving from {} to {} at time step {}".format(drone_idx, self.all_agents[drone_idx].pre_pos, self.all_agents[drone_idx].pos, current_ts))
                    break
            end_v1_time = (time.time() - start_of_v1_time)*1000*1000
            # print("check building collision V1 time used is {} micro".format(end_v1_time))
            # -----------end of check collision with building v1 ---------

            end_v2_time, end_v3_time, v2_decision, v3_decision = 0, 0, 0, 0,
            step_collision_record[drone_idx].append([end_v1_time, end_v2_time, end_v3_time,
                                                     v1_decision, v2_decision, v3_decision])
            # if step_collision_record[drone_idx] == None:
            #     step_collision_record[drone_idx] = [[end_v1_time, end_v2_time, end_v3_time,
            #                                          v1_decision, v2_decision, v3_decision]]
            # else:
            #     step_collision_record[drone_idx].append([end_v1_time, end_v2_time, end_v3_time,
            #                                              v1_decision, v2_decision, v3_decision])

            # tar_circle = Point(self.all_agents[drone_idx].goal[0]).buffer(1, cap_style='round')
            tar_circle = Point(self.all_agents[drone_idx].goal[-1]).buffer(1, cap_style='round')  # set to [-1] so there are no more reference path
            # when there is no intersection between two geometries, "RuntimeWarning" will appear
            # RuntimeWarning is, "invalid value encountered in intersection"
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                goal_cur_intru_intersect = host_current_circle.intersection(tar_circle)

            # wp_circle = Point(self.all_agents[drone_idx].goal[0]).buffer(1, cap_style='round')
            # wp_circle = Point(self.all_agents[drone_idx].goal[0]).buffer(drone_obj.protectiveBound,
            #                                                              cap_style='round')
            # wp_circle = Point(self.all_agents[drone_idx].goal[0]).buffer(3.5, cap_style='round')
            # wp_intersect = host_current_circle.intersection(wp_circle)
            wp_reach_threshold_dist = 5
            # --------------- a new way to check for the next wp --------------------
            # smallest_dist = math.inf
            # wp_intersect_flag = False
            # for wpidx, wp in enumerate(self.all_agents[drone_idx].goal):
            #     cur_dist_to_wp = curPoint.distance(Point(wp))
            #     if cur_dist_to_wp < smallest_dist:
            #         smallest_dist = cur_dist_to_wp
            #         next_wp = np.array(wp)
            #         if smallest_dist < wp_reach_threshold_dist:
            #             wp_intersect_flag = True
            #             # we find the next wp, as long as it is not the last wp
            #             if len(self.all_agents[drone_idx].goal) > 1:
            #                 drone_obj.removed_goal = drone_obj.goal.pop(wpidx)  # remove current wp
            #                 points_list = [Point(coord) for coord in self.all_agents[drone_idx].goal]
            #                 next_wPoint = min(points_list, key=lambda point: point.distance(curPoint))
            #                 next_wp = np.array([next_wPoint.x, next_wPoint.y])
            #             break  # once the nearest wp is found we break out of the loop
            # ---------------end of a new way to check for the next wp --------------------

            #  ------  using sequence wp reaching method ----------
            cur_dist_to_wp = curPoint.distance(Point(self.all_agents[drone_idx].waypoints[0]))
            next_wp = np.array(self.all_agents[drone_idx].waypoints[0])

            if cur_dist_to_wp < wp_reach_threshold_dist:
                wp_intersect_flag = True
            else:
                wp_intersect_flag = False
            # ------ end of using sequence wp reaching method ----------

            # ------------- pre-processed condition for a normal step -----------------
            # rew = 3
            rew = 0
            # dist_to_goal_coeff = 1
            # dist_to_goal_coeff = 3
            dist_to_goal_coeff = 6
            # dist_to_goal_coeff = 1
            # dist_to_goal_coeff = 0
            # dist_to_goal_coeff = 2

            x_norm, y_norm = self.normalizer.nmlz_pos(drone_obj.pos)
            tx_norm, ty_norm = self.normalizer.nmlz_pos(drone_obj.goal[-1])
            after_dist_hg = np.linalg.norm(drone_obj.pos - drone_obj.goal[-1])  # distance to goal after action

            # -- original --
            dist_left = total_length_to_end_of_line(drone_obj.pos, drone_obj.ref_line)
            dist_to_goal = dist_to_goal_coeff * (1 - (dist_left / drone_obj.ref_line.length))
            # end of original --

            # ---- leading to goal reward V4 ----
            # before_dist_hg = np.linalg.norm(drone_obj.pre_pos - drone_obj.goal[-1])  # distance to goal before action
            # # before_dist_hg = np.linalg.norm(drone_obj.pre_pos - next_wp)  # distance to goal before action
            # after_dist_hg = np.linalg.norm(drone_obj.pos - drone_obj.goal[-1])  # distance to goal after action
            # # after_dist_hg = np.linalg.norm(drone_obj.pos - next_wp)  # distance to goal after action
            # dist_to_goal = dist_to_goal_coeff * (before_dist_hg - after_dist_hg)
            # dist_to_goal = dist_to_goal / drone_obj.maxSpeed  # perform a normalization
            # ---- end of leading to goal reward V4 ----

            # ---- V5 euclidean distance ----
            # dist_away = np.linalg.norm(drone_obj.ini_pos - drone_obj.goal[-1])
            # after_dist_hg = np.linalg.norm(drone_obj.pos - drone_obj.goal[-1])  # distance to goal after action
            # if after_dist_hg > dist_away:
            #     dist_to_goal = dist_to_goal_coeff * 0
            # else:
            #     dist_to_goal = dist_to_goal_coeff * (1-after_dist_hg/dist_away)
            # ---- end of V5 -------

            # ----- v4 accumulative ---
            # one_drone_dist_to_goal = dist_to_goal_coeff * (before_dist_hg - after_dist_hg)  # (before_dist_hg - after_dist_hg) -max_vel - max_vel
            # one_drone_dist_to_goal = one_drone_dist_to_goal / drone_obj.maxSpeed  # perform a normalization
            # dist_to_goal = dist_to_goal + one_drone_dist_to_goal
            # ------ end of v4 accumulative----


            # dist_left = total_length_to_end_of_line(drone_obj.pos, drone_obj.ref_line)
            # dist_to_goal = dist_to_goal_coeff * (1 - (dist_left / drone_obj.ref_line.length))  # v1

            # ---- v2 leading to goal reward, based on compute_projected_velocity ---
            # projected_velocity = compute_projected_velocity(drone_obj.vel, drone_obj.ref_line, Point(drone_obj.pos))
            # get the norm as the projected_velocity.
            # dist_to_goal = dist_to_goal_coeff * np.linalg.norm(projected_velocity)
            # ---- end of v2 leading to goal reward, based on compute_projected_velocity ---

            # ---- v3 leading to goal reward, based on remained distance to travel only ---
            # dist_left = total_length_to_end_of_line_without_cross(drone_obj.pos, drone_obj.ref_line)
            # dist_to_goal = dist_to_goal_coeff * (1 - (dist_left / drone_obj.ref_line.length))  # v3
            # ---- end of v3 leading to goal reward, based on remained distance to travel only ---

            # if dist_to_goal > drone_obj.maxSpeed:
            #     print("dist_to_goal reward out of range")

            # ------- small segment reward ------------
            # dist_to_seg_coeff = 10
            # dist_to_seg_coeff = 1
            dist_to_seg_coeff = 0

            # if drone_obj.removed_goal == None:
            #     total_delta_seg_vector = np.linalg.norm((drone_obj.ini_pos - np.array(drone_obj.goal[0])))
            # else:
            #     total_delta_seg_vector = np.linalg.norm((np.array(drone_obj.removed_goal) - np.array(drone_obj.goal[0])))
            # delta_seg_vector = drone_obj.pos - drone_obj.goal[0]
            # dist_seg_vector = np.linalg.norm(delta_seg_vector)
            # if dist_seg_vector / total_delta_seg_vector <= 1:  # we reward the agent
            #     seg_reward = dist_to_seg_coeff * (dist_seg_vector / total_delta_seg_vector)
            # else:
            #     seg_reward = dist_to_seg_coeff * (-1)*(dist_seg_vector / total_delta_seg_vector)

            # s_tx_norm, s_ty_norm = self.normalizer.nmlz_pos(drone_obj.goal[0])
            # seg_reward = dist_to_seg_coeff * math.sqrt(((x_norm-s_tx_norm)**2 + (y_norm-s_ty_norm)**2))  # 0~2.828 at each step
            seg_reward = dist_to_seg_coeff * 0
            # -------- end of small segment reward ----------

            # dist_to_goal = 0
            # coef_ref_line = 0.5
            # coef_ref_line = -10
            # coef_ref_line = 3
            # coef_ref_line = 1
            # coef_ref_line = 2
            # coef_ref_line = 1.5
            coef_ref_line = 0
            cross_err_distance, x_error, y_error, nearest_pt = self.cross_track_error(host_current_point, drone_obj.ref_line)  # deviation from the reference line, cross track error
            norm_cross_track_deviation_x = x_error * self.normalizer.x_scale
            norm_cross_track_deviation_y = y_error * self.normalizer.y_scale
            # dist_to_ref_line = coef_ref_line*math.sqrt(norm_cross_track_deviation_x ** 2 +
            #                                            norm_cross_track_deviation_y ** 2)

            if cross_err_distance <= drone_obj.protectiveBound:
                # linear increase in reward
                m = (0 - 1) / (drone_obj.protectiveBound - 0)
                dist_to_ref_line = coef_ref_line*(m * cross_err_distance + 1)  # 0~1*coef_ref_line
                # dist_to_ref_line = (coef_ref_line*(m * cross_err_distance + 1)) + coef_ref_line  # 0~1*coef_ref_line, with a fixed reward
            else:
                dist_to_ref_line = -coef_ref_line*1
                # dist_to_ref_line = -coef_ref_line*3
                # dist_to_ref_line = -coef_ref_line*0

            # ------- penalty for surrounding agents as a whole -----
            surrounding_collision_penalty = 0
            # if pre_total_possible_conflict < cur_total_possible_conflict:
            #     surrounding_collision_penalty = 2
            # ------- end of reward for surrounding agents as a whole ----

            # ----- start of near drone penalty ----------------
            near_drone_penalty_coef = 10
            # near_drone_penalty_coef = 5
            # near_drone_penalty_coef = 1
            # near_drone_penalty_coef = 3
            # near_drone_penalty_coef = 0
            near_drone_penalty = 0
            # dist_to_penalty_upperbound = 6
            dist_to_penalty_upperbound = 10
            dist_to_penalty_lowerbound = 2.5
            # assume when at lowerbound, y = 1
            all_neigh_dist.sort()
            c_drone = 1 + (dist_to_penalty_lowerbound / (dist_to_penalty_upperbound - dist_to_penalty_lowerbound))
            m_drone = (0 - 1) / (dist_to_penalty_upperbound - dist_to_penalty_lowerbound)
            if nearest_neigh_key is not None:
                for neigh_dist_idx, shortest_neigh_dist in enumerate(all_neigh_dist):
                    if neigh_dist_idx == 2:
                    # if neigh_dist_idx == 1:
                        break
                    if shortest_neigh_dist >= dist_to_penalty_lowerbound and shortest_neigh_dist <= dist_to_penalty_upperbound:
                        if neigh_relative_bearing >= 90.0 and neigh_relative_bearing < 270:
                            near_drone_penalty_coef = near_drone_penalty_coef * 2
                        else:
                            pass
                        near_drone_penalty = near_drone_penalty + near_drone_penalty_coef * (m_drone * shortest_neigh_dist + c_drone)
                    else:
                        near_drone_penalty = near_drone_penalty + near_drone_penalty_coef * 0
            else:
                near_drone_penalty = near_drone_penalty_coef * 0
            # -----end of near drone penalty ----------------



            small_step_penalty_coef = 5
            # small_step_penalty_coef = 0
            spd_penalty_threshold = drone_obj.maxSpeed / 2
            # spd_penalty_threshold = drone_obj.protectiveBound
            small_step_penalty_val = (spd_penalty_threshold -
                                  np.clip(np.linalg.norm(drone_obj.vel), 0, spd_penalty_threshold))*\
                                 (1.0 / spd_penalty_threshold)  # between 0-1.
            small_step_penalty = small_step_penalty_coef * small_step_penalty_val

            # dist_moved = np.linalg.norm(drone_obj.pos - drone_obj.pre_pos)
            # if dist_moved <= 1:
            #     small_step_penalty = small_step_penalty_coef * 1
            # else:
            #     small_step_penalty = small_step_penalty_coef * 0

            # near_goal_coefficient = 3  # so that near_goal_reward will become 0-3 instead of 0-1
            near_goal_coefficient = 0
            near_goal_threshold = drone_obj.detectionRange
            actual_after_dist_hg = math.sqrt(((drone_obj.pos[0] - drone_obj.goal[-1][0]) ** 2 +
                                              (drone_obj.pos[1] - drone_obj.goal[-1][1]) ** 2))
            near_goal_reward = near_goal_coefficient * ((near_goal_threshold -
                                np.clip(actual_after_dist_hg, 0, near_goal_threshold)) * 1.0/near_goal_threshold)

            # penalty for any buildings are getting too near to the host agent
            turningPtConst = drone_obj.detectionRange/2-drone_obj.protectiveBound  # this one should be 12.5
            # dist_array = np.array([dist_info[0] for dist_info in drone_obj.observableSpace])  # used when radar detect other uavs
            dist_array = np.array([dist_info for dist_info in drone_obj.observableSpace])

            ascending_array = np.sort(dist_array)
            min_index = np.argmin(dist_array)
            min_dist = dist_array[min_index]
            # radar_status = drone_obj.observableSpace[min_index][-1]  # radar status for now not required

            # ---linear building penalty ---
            # the distance is based on the minimum of the detected distance to surrounding buildings.
            # near_building_penalty_coef = 1
            near_building_penalty_coef = 3
            # near_building_penalty_coef = 0
            # near_building_penalty = near_building_penalty_coef*((1-(1/(1+math.exp(turningPtConst-min_dist))))*
            #
            #                                                     (1-(min_dist/turningPtConst)**2))  # value from 0 ~ 1.
            # turningPtConst = 12.5
            turningPtConst = 5
            if turningPtConst == 12.5:
                c = 1.25
            elif turningPtConst == 5:
                c = 2
            # # linear building penalty
            # makesure only when min_dist is >=0 and <= turningPtConst, then we activate this penalty
            m = (0-1)/(turningPtConst-drone_obj.protectiveBound)  # we must consider drone's circle, because when min_distance is less than drone's radius, it is consider collision.
            if min_dist>=drone_obj.protectiveBound and min_dist<=turningPtConst:  # only when min_dist is between 2.5~5, this penalty is working.
                near_building_penalty = near_building_penalty_coef*(m*min_dist+c)  # at each step, penalty from 3 to 0.
            else:
                near_building_penalty = 0  # if min_dist is outside of the bound, other parts of the reward will be taking care.
            # --- end of linear building penalty ---

            # -------------end of pre-processed condition for a normal step -----------------
            #
            # Always check the boundary as the 1st condition, or else will encounter error where the agent crash into wall but also exceed the bound, but crash into wall did not stop the episode. So, we must put the check boundary condition 1st, so that episode can terminate in time and does not leads to exceed boundary with error in no polygon found.
            # exceed bound condition, don't use current point, use current circle or else will have condition that
            # must use "host_passed_volume", or else, we unable to confirm whether the host's circle is at left or right of the boundary lines
            if x_left_bound.intersects(host_passed_volume) or x_right_bound.intersects(host_passed_volume) or y_bottom_bound.intersects(host_passed_volume) or y_top_bound.intersects(host_passed_volume):
                print("drone_{} has crash into boundary at time step {}".format(drone_idx, current_ts))
                drone_obj.bound_collision = True
                rew = rew - crash_penalty_wall
                if args.mode == 'eval' and evaluation_by_episode == False:
                    done.append(False)
                else:  # during training or evaluation by episode is TRUE
                    done.append(True)
                bound_building_check[0] = True
                # done.append(False)
                reward.append(np.array(rew))
            # # crash into buildings or crash with other neighbors
            elif collide_building == 1:
                if args.mode == 'eval' and evaluation_by_episode == False:
                    done.append(False)
                else:  # during training or evaluation by episode is TRUE
                    done.append(True)
                bound_building_check[1] = True
                rew = rew - crash_penalty_wall
                # rew = rew - big_crash_penalty_wall
                reward.append(np.array(rew))
            # # ---------- Termination only during collision to wall on the 3rd time -----------------------
            # elif drone_obj.collide_wall_count >0:
            #     if drone_obj.collide_wall_count == 1:
            #         done.append(False)
            #         rew = rew - dist_to_ref_line - crash_penalty_wall - dist_to_goal - small_step_penalty + near_goal_reward -5
            #         reward.append(np.array(rew))
            #     elif drone_obj.collide_wall_count == 2:
            #         done.append(False)
            #         rew = rew - dist_to_ref_line - crash_penalty_wall - dist_to_goal - small_step_penalty + near_goal_reward -15
            #         reward.append(np.array(rew))
            #     else:
            #         done.append(True)
            #         rew = rew - dist_to_ref_line - crash_penalty_wall - dist_to_goal - small_step_penalty + near_goal_reward - 20
            #         reward.append(np.array(rew))
            # # ----------End of termination only during collision to wall on the 3rd time -----------------------
            elif len(collision_drones) > 0:
                if args.mode == 'eval' and evaluation_by_episode == False:
                    done.append(False)
                else:  # during training or evaluation by episode is TRUE
                    done.append(True)
                # done.append(False)
                bound_building_check[2] = True
                if neigh_collision_bearing >=90.0 and neigh_collision_bearing <=180:
                    crash_penalty_wall = crash_penalty_wall * 2
                else:
                    pass
                rew = rew - crash_penalty_wall
                reward.append(np.array(rew))
                # check if the collision is due to the nearest drone.
                # if collision_drones[-1] == nearest_neigh_key:
                # check if the collision is due to the previous nearest two drone.
                if flag_previous_nearest_two:
                    bound_building_check[3] = True
            elif not goal_cur_intru_intersect.is_empty:  # reached goal?
                # --------------- with way point -----------------------
                drone_obj.reach_target = True
                check_goal[drone_idx] = True

                # print("drone_{} has reached its final goal at time step {}".format(drone_idx, current_ts))
                agent_to_remove.append(drone_idx)  # NOTE: drone_idx is the key value.
                rew = rew + reach_target + near_goal_reward
                reward.append(np.array(rew))
                done.append(False)
                # --------------- end of with way point -----------------------
                # without wap point
                # rew = rew + reach_target
                # reward.append(np.array(rew))
                # print("final goal has reached")
                # done.append(False)
            else:  # a normal step taken
                if xy[0] is None and xy[1] is None:  # we only alter drone's goal during actual training
                    # if (not wp_intersect.is_empty) and len(drone_obj.goal) > 1: # check if wp reached, and this is not the end point
                    if wp_intersect_flag and len(drone_obj.waypoints) > 1: # check if wp reached and don't remove last element
                        drone_obj.removed_goal = drone_obj.waypoints.pop(0)  # remove current wp
                        # we add a wp reached reward, this reward is equals to the maximum of the path deviation reward
                        # rew = rew + coef_ref_line
                        # print("drone {} has reached a WP on step {}, claim additional {} points of reward"
                        #       .format(drone_idx, current_ts, coef_ref_line))
                # if drone_obj.reach_target == False:
                #     rew = rew + dist_to_ref_line + dist_to_goal - \
                #           small_step_penalty + near_goal_reward - near_building_penalty + seg_reward-survival_penalty - near_drone_penalty
                # else:
                #     rew = rew + move_after_reach
                rew = rew + dist_to_ref_line + dist_to_goal - \
                      small_step_penalty + near_goal_reward - near_building_penalty + seg_reward \
                      - survival_penalty - near_drone_penalty - surrounding_collision_penalty
                # we remove the above termination condition
                # if current_ts >= args.episode_length:
                #     done.append(True)
                # else:
                #     done.append(False)
                done.append(False)
                step_reward = np.array(rew)
                reward.append(step_reward)
                # for debug, record the reward
                # one_step_reward = [crossCoefficient*cross_track_error, delta_hg, alive_penalty, dominoCoefficient*dominoTerm_sum]

                # if rew < 1:
                #     print("check")
            # if rew < 0.1 and rew >= 0:
            #     print("check")
            step_reward_record[drone_idx] = [dist_to_ref_line, rew]

            # print("current drone {} actual distance to goal is {}, current reward is {}".format(drone_idx, actual_after_dist_hg, reward[-1]))
            # print("current drone {} actual distance to goal is {}, current reward to gaol is {}, current ref line reward is {}, current step reward is {}".format(drone_idx, actual_after_dist_hg, dist_to_goal, dist_to_ref_line, rew))

            # record status of each step.
            eps_status_holder = self.display_one_eps_status(eps_status_holder, drone_idx, np.array(after_dist_hg),
                                                            [np.array(dist_to_goal), cross_err_distance, dist_to_ref_line,
                                                             np.array(near_building_penalty), small_step_penalty,
                                                             np.linalg.norm(drone_obj.vel), near_goal_reward,
                                                             seg_reward, nearest_pt, drone_obj.observableSpace,
                                                             drone_obj.heading, np.array(near_drone_penalty)])
            # overall_status_record[2].append()  # 3rd is accumulated reward till that step for each agent

        if full_observable_critic_flag:
            reward = [np.sum(reward) for _ in reward]
            # done = any(done)

        # if all(check_goal):
        #     for element_idx, element in enumerate(done):
        #         done[element_idx] = True

        # ever_reached = [agent.reach_target for agent in self.all_agents.values()]
        # if check_goal.count(True) == 1 and ever_reached.count(True) == 0:
        #     reward = [ea_rw + 200 for ea_rw in reward]
        # elif check_goal.count(True) == 2 and ever_reached.count(True) == 1:
        #     reward = [ea_rw + 400 for ea_rw in reward]
        # elif check_goal.count(True) == 3 and ever_reached.count(True) == 2:
        #     reward = [ea_rw + 600 for ea_rw in reward]

        # all_reach_target = all(agent.reach_target == True for agent in self.all_agents.values())
        # if all_reach_target:  # in this episode all agents have reached their target at least one
        #     # we cannot just assign a single True to "done", as it must be a list to output from the function.
        #     done = [True, True, True]

        return reward, done, check_goal, step_reward_record, eps_status_holder, step_collision_record, bound_building_check


    def display_one_eps_status(self, status_holder, drone_idx, cur_dist_to_goal, cur_step_reward):
        status_holder[drone_idx]['Euclidean_dist_to_goal'] = cur_dist_to_goal
        status_holder[drone_idx]['goal_leading_reward'] = cur_step_reward[0]
        status_holder[drone_idx]['deviation_to_ref_line'] = cur_step_reward[1]
        status_holder[drone_idx]['deviation_to_ref_line_reward'] = cur_step_reward[2]
        status_holder[drone_idx]['near_building_penalty'] = cur_step_reward[3]
        status_holder[drone_idx]['small_step_penalty'] = cur_step_reward[4]
        status_holder[drone_idx]['current_drone_speed'] = cur_step_reward[5]
        status_holder[drone_idx]['addition_near_goal_reward'] = cur_step_reward[6]
        status_holder[drone_idx]['segment_reward'] = cur_step_reward[7]
        status_holder[drone_idx]['neareset_point'] = cur_step_reward[8]
        status_holder[drone_idx]['A'+str(drone_idx)+'_observable space'] = cur_step_reward[9]
        status_holder[drone_idx]['A'+str(drone_idx)+'_heading'] = cur_step_reward[10]
        status_holder[drone_idx]['near_drone_penalty'] = cur_step_reward[11]
        return status_holder

    def step(self, actions, current_ts, acc_max, args, evaluation_by_episode, full_observable_critic_flag):
        next_combine_state = []
        agentCoorKD_list_update = []
        agentRefer_dict = {}  # A dictionary to use agent's current pos as key, their agent name (idx) as value
        # we use 4 here, because the time-step for the simulation is 0.5 sec.
        # hence, 4 here is equivalent to the acceleration of 2m/s^2

        # coe_a = 4  # coe_a is the coefficient of action is 4 because our time step is 0.5 sec
        coe_a = acc_max  # coe_a is the coefficient of action is 4 because our time step is 0.5 sec
        # based on the input stack of actions we propagate all agents forward
        # for drone_idx, drone_act in actions.items():  # this is for evaluation with default action
        count = 1
        for drone_idx_obj, drone_act in zip(self.all_agents.items(), actions):
            drone_idx = drone_idx_obj[0]
            drone_obj = drone_idx_obj[1]
            # let current neighbor become neighbor recorded before action
            start_deepcopy_time = time.time()
            self.all_agents[drone_idx].pre_surroundingNeighbor = deepcopy(self.all_agents[drone_idx].surroundingNeighbor)
            # let current position become position is the previous state, so that new position can be updated
            self.all_agents[drone_idx].pre_pos = deepcopy(self.all_agents[drone_idx].pos)
            # fill previous velocities
            self.all_agents[drone_idx].pre_vel = deepcopy(self.all_agents[drone_idx].vel)
                        # fill previous acceleration
            self.all_agents[drone_idx].pre_acc = deepcopy(self.all_agents[drone_idx].acc)
            # print("deepcopy done, time used {} milliseconds".format((time.time()-start_deepcopy_time)*1000))

            if args.mode == 'eval' and evaluation_by_episode == False:
                if self.all_agents[drone_idx].reach_target == True \
                        or self.all_agents[drone_idx].bound_collision == True \
                        or self.all_agents[drone_idx].building_collision == True \
                        or self.all_agents[drone_idx].drone_collision == True:
                    continue  # we make the drone don't move.

            # --------------- speed & heading angle control for training -------------------- #
            # raw_speed, raw_heading_angle = drone_act[0], drone_act[1]
            # speed = ((raw_speed + 1) / 2) * self.all_agents[drone_idx].maxSpeed  # map from -1 to 1 to 0 to maxSpd of the agent
            # heading_angle = raw_heading_angle * math.pi  # ensure the heading angle is between -pi to pi.
            # delta_x = speed * math.cos(heading_angle) * self.time_step
            # delta_y = speed * math.sin(heading_angle) * self.time_step
            # -------------- end of speed & heading angle control ---------------------#

            # ----------------- acceleration in x and acceleration in y state transition control for training-------------------- #
            ax, ay = drone_act[0], drone_act[1]

            ax = ax * coe_a
            ay = ay * coe_a

            # record current drone's acceleration
            self.all_agents[drone_idx].acc = np.array([ax, ay])

            # check velocity limit
            curVelx = self.all_agents[drone_idx].vel[0] + ax * self.time_step
            curVely = self.all_agents[drone_idx].vel[1] + ay * self.time_step
            next_heading = math.atan2(curVely, curVelx)
            if np.linalg.norm([curVelx, curVely]) >= self.all_agents[drone_idx].maxSpeed:

                # update host velocity when chosen speed has exceeded the max speed
                hvx = self.all_agents[drone_idx].maxSpeed * math.cos(next_heading)
                hvy = self.all_agents[drone_idx].maxSpeed * math.sin(next_heading)
                self.all_agents[drone_idx].vel = np.array([hvx, hvy])
            else:
                # update host velocity when max speed is not exceeded
                self.all_agents[drone_idx].vel = np.array([curVelx, curVely])

            #print("At time step {} the drone_{}'s output speed is {}".format(current_ts, drone_idx, np.linalg.norm(self.all_agents[drone_idx].vel)))

            # update the drone's position based on the update velocities
            if drone_obj.reach_target == True:
                delta_x = 0
                delta_y = 0
            else:
                delta_x = self.all_agents[drone_idx].vel[0] * self.time_step
                delta_y = self.all_agents[drone_idx].vel[1] * self.time_step

            # update current acceleration of the agent after an action
            self.all_agents[drone_idx].acc = np.array([ax, ay])

            counterCheck_heading = math.atan2(delta_y, delta_x)

            self.all_agents[drone_idx].heading = counterCheck_heading
            # ------------- end of acceleration in x and acceleration in y state transition control ---------------#

            self.all_agents[drone_idx].pos = np.array([self.all_agents[drone_idx].pos[0] + delta_x,
                                                       self.all_agents[drone_idx].pos[1] + delta_y])

            # cur_circle = Point(self.all_agents[drone_idx].pos[0],
            #                    self.all_agents[drone_idx].pos[1]).buffer(self.all_agents[drone_idx].protectiveBound,
            #                                                             cap_style='round')

            agentCoorKD_list_update.append(self.all_agents[drone_idx].pos)
            agentRefer_dict[(self.all_agents[drone_idx].pos[0],
                             self.all_agents[drone_idx].pos[1])] = self.all_agents[drone_idx].agent_name
            count = count + 1
        # self.cur_allAgentCoor_KD = KDTree(agentCoorKD_list_update)  # update all agent coordinate KDtree

        # print("for loop run {} times".format(count))

        # next_state, next_state_norm = self.cur_state_norm_state_fully_observable(agentRefer_dict)
        # start_acceleration_time = time.time()
        next_state, next_state_norm, polygons_list, all_agent_st_points, all_agent_ed_points, all_agent_intersection_point_list, all_agent_line_collection, all_agent_mini_intersection_list = self.cur_state_norm_state_v3(agentRefer_dict, full_observable_critic_flag)
        # print("obtain_current_state, time used {} milliseconds".format(
        #     (time.time() - start_acceleration_time) * 1000))

        # # update current agent's observable space state
        # agent.observableSpace = self.current_observable_space(agent)
        # #cur_ObsGrids.append(agent.observableSpace)
        #
        # # update the "surroundingNeighbor" attribute
        # agent.surroundingNeighbor = self.get_current_agent_nei(agent, agentRefer_dict)
        # #actor_obs.append(agent.surroundingNeighbor)
        #
        # # populate overall state
        # # next_combine_state.append(np.array([agent_own, agent.observableSpace, agent.surroundingNeighbor], dtype=object))
        # next_combine_state.append(np.concatenate((agent_own, np.array(other_pos).flatten())))


        # matplotlib.use('TkAgg')
        # plt.ion()
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.set_aspect('equal')
        #
        # for t in range(max_t):
        #     ax.set_xlim([self.bound[0], self.bound[1]])
        #     ax.set_ylim([self.bound[2], self.bound[3]])
        #     previous_time = deepcopy(self.global_time)
        #     cur_time = previous_time + self.time_step
        #     step_volume_collection = []
        #     agent_moving_vol = {}
        #     for agent_idx, agent in self.all_agents.items():
        #         previous_pos = deepcopy(agent.pos)
        #         dx = actions[agent_idx][0] * self.time_step
        #         dy = actions[agent_idx][1] * self.time_step
        #         agent.pos[0] = agent.pos[0] + dx
        #         agent.pos[1] = agent.pos[1] + dy
        #         cur_agent_passLine = LineString([(previous_pos[0], previous_pos[1]),
        #                                          (agent.pos[0], agent.pos[1])])
        #         cur_agent_passed_volume = cur_agent_passLine.buffer(agent.protectiveBound, cap_style='round')
        #         agent_moving_vol[agent_idx] = cur_agent_passed_volume
        #         step_volume_collection.append(cur_agent_passed_volume)
        #
        #         #plt.text(previous_pos[0], previous_pos[1], "{}, t={}".format(agent.agent_name, previous_time))
        #         matp_cur_volume = shapelypoly_to_matpoly(cur_agent_passed_volume, True, 'red', 'b')
        #         ax.add_patch(matp_cur_volume)
        #         plt.text(agent.pos[0], agent.pos[1], "{}".format(agent.agent_name))
        #
        #     step_volume_STR = STRtree(step_volume_collection)
        #
        #     # checking reach goal before the check collision. So that at the time step, when an agent reaches goal and
        #     # collide with other agent at the same time, it is consider as reaching destination instead of collision
        #
        #     collided_drone = []
        #     reached_drone = []
        #     for agentIdx_key, agent_passed_volume in agent_moving_vol.items():
        #         # check goal
        #         cur_drone_tar = Point(self.all_agents[agentIdx_key].goal[0][0],
        #                               self.all_agents[agentIdx_key].goal[0][1]).buffer(1, cap_style='round')
        #
        #         mat_cur_tar = shapelypoly_to_matpoly(cur_drone_tar, True, 'c', 'r')
        #         ax.add_patch(mat_cur_tar)
        #         plt.text(self.all_agents[agentIdx_key].goal[0][0],
        #                  self.all_agents[agentIdx_key].goal[0][1],
        #                  "{} goal".format(self.all_agents[agentIdx_key].agent_name))
        #
        #         if cur_drone_tar.intersects(agent_passed_volume):
        #             reached_drone.append(agentIdx_key)
        #             continue  # one drone reached its target no need to check any possible collision for this drone
        #
        #         # check collision
        #         possible_idx = step_volume_STR.query(agent_passed_volume)
        #         for other_agent_cir in step_volume_STR.geometries.take(possible_idx):
        #             if not other_agent_cir.equals(agent_passed_volume):
        #                 # record this volume only when not equals to itself.
        #                 collided_drone.append(agentIdx_key)
        #
        #     # if reached goal, remove the agent from the environment
        #     for i in reached_drone:
        #         del self.all_agents[i]
        #         print("agent_{} reached, it is removed from the environment".format(i))
        #     # Remove element in "collided_drone", such that these elements also present in "reached_drone"
        #     collided_drone = [x for x in collided_drone if x not in reached_drone]
        #     # remove possible duplicates in "collided_drone"
        #     collided_drone = list(set(collided_drone))
        #
        #     # if collide, remove any agents involved in the collision
        #     for i in collided_drone:
        #         del self.all_agents[i]
        #         print("removed agent_ {}, left {} agents".format(i, len(self.all_agents)))
        #     fig.canvas.draw()
        #     plt.show()
        #     time.sleep(2)
        #     fig.canvas.flush_events()
        #     ax.cla()

        return next_state, next_state_norm, polygons_list, all_agent_st_points, all_agent_ed_points, all_agent_intersection_point_list, all_agent_line_collection, all_agent_mini_intersection_list

    def fill_agents(self, max_agent_train, cur_state, norm_cur_state, remove_agent_keys):
        num_lack = int(max_agent_train-len(self.all_agents))
        # int_removed_name = [int(re.search(r'\d+(\.\d+)?', name).group()) for name in remove_agent_named]
        agent_filled = []
        if num_lack > 0:
            for i in range(num_lack):
                # when both drones / all drones reaches the goal at same ts. Will have error. self.all_agents.keys() will become zeros. Debug this problem.
                # ----------------original ------------------
                # selected_key = random.choice(list(self.all_agents.keys()))
                # selected_agent = self.all_agents[selected_key]
                # agent = deepcopy(selected_agent)
                # current_max = max(list(self.all_agents.keys()))
                # ----------------- end of original -----------
                agent = deepcopy(self.dummy_agent)
                if len(self.all_agents) == 0:
                    current_max = max(remove_agent_keys)  # all agents have reaches their destination at same time
                else:
                    try:
                        current_max = max(max(list(self.all_agents.keys())), max(remove_agent_keys))
                    except:
                        print("pause")

                agent.agent_name = 'agent_%s' % str(current_max+1)
                self.all_agents[current_max+1] = agent
                agent_filled.append(current_max+1)
            # overall_state, norm_overall_state = self.fill_agent_reset(agent_filled)
            updated_cur_state, updated_norm_cur_state = self.fill_agent_reset(agent_filled)
            return updated_cur_state, updated_norm_cur_state
        else:
            return cur_state, norm_cur_state

    def central_learning(self, ReplayBuffer, batch_size, maxIntruNum, intruFeature, UPDATE_EVERY):
        with torch.autograd.set_detect_anomaly(True):
            critic_losses, actor_losses = [], []

            cur_state, action, reward, next_state, done = ReplayBuffer.sample(batch_size, maxIntruNum, intruFeature, self.all_agents[0].max_grid_obs_dim)

            device = self.all_agents[0].actorNet.device

            # pre-process cur_state and next_state so that they can be used as input for every agent's critic network
            cur_state_pre_processed = preprocess_batch_for_critic_net_v2(cur_state, batch_size)
            next_state_pre_processed = preprocess_batch_for_critic_net_v2(next_state, batch_size)

            # load action, reward, done to tensor
            action = T.tensor(np.array(action), dtype=T.float).to(device)
            reward = T.tensor(np.array(reward), dtype=T.float).to(device)
            done = T.tensor(np.array(done)).to(device)

            # all these three different actions are needed to calculate the loss function
            all_agents_new_actions = []  # actions according to the target network for the new state
            all_agents_new_mu_actions = []  # actions according to the regular actor network for the current state
            old_agents_actions = []  # actions the agent actually took

            for agent_idx, agent in self.all_agents.items():
                # actions according to the target network for the new state
                # next_own = T.tensor(next_state[agent_idx][0], dtype=T.float).to(device)
                # next_grid = T.tensor(next_state[agent_idx][1], dtype=T.float).to(device)
                # next_nei = T.tensor(next_state[agent_idx][2], dtype=T.float).to(device)
                # agent_new_states = [next_own, next_grid, next_nei]

                next_own = T.tensor(next_state[agent_idx], dtype=T.float).to(device)
                new_pi = agent.target_actorNet.forward(next_own)  # individual agent's target network

                all_agents_new_actions.append(new_pi)  # record actions generated by each agent's target network
                # actions according to the regular actor network for the current state
                # cur_own = T.tensor(cur_state[agent_idx][0], dtype=T.float).to(device)
                # cur_grid = T.tensor(cur_state[agent_idx][1], dtype=T.float).to(device)
                # cur_nei = T.tensor(cur_state[agent_idx][2], dtype=T.float).to(device)

                # mu_states = [cur_own, cur_grid, cur_nei]
                cur_own = T.tensor(cur_state[agent_idx], dtype=T.float).to(device)

                # using agent's predict network to generate action based off current states
                pi = agent.actorNet.forward(cur_own)

                all_agents_new_mu_actions.append(pi)
                # record actions the agent actually took
                old_agents_actions.append(action[agent_idx])

            new_actions = T.cat([acts for acts in all_agents_new_actions], dim=1)  # batch_size X (agent_num X action dim)
            mu = T.cat([acts for acts in all_agents_new_mu_actions], dim=1)
            old_actions = T.cat([acts for acts in old_agents_actions], dim=1)

            # These two losses are used to record overall losses for entire system at each learning step
            critic_losses = []
            actor_losses = []
            # in_order_count = 0
            # handle cost function
            for agent_idx, agent in self.all_agents.items():
                # squeeze() will remove all dimensions with size 1
                # without squeeze() is 10x1x1, so is batch_number x 1 by 1 array.
                # using individual agent's critic prediction network
                # current Q estimate, shape is batch_size X 1
                critic_value = agent.criticNet.forward(cur_state_pre_processed, old_actions)
                # get target Q value for each agent
                with T.no_grad():
                    # next_state_pre_processed is arranged in a way that, always from 1st agent to last agent and ...
                    # agent's idx is ignored, that's why they were stored in a list.
                    # hence, "for agent_idx, agent in self.all_agents.items():" is just to loop through all agents in order
                    # DO NOT use "agent_idx" as index, or else it will produce error, when add/remove agents are added.
                    # First line for critic_value_prime did not involve any individual agent's attributes
                    # because we are using centralized critic, shape is batch_size X 1
                    critic_value_prime = agent.target_criticNet.forward(next_state_pre_processed, new_actions)
                    critic_value_prime[done[agent_idx]] = 0.0
                    target = reward[agent_idx] + agent.gamma * critic_value_prime


                # calculate critic loss for each agent
                critic_loss = F.mse_loss(critic_value, target)
                # optimization
                agent.criticNet.optimizer.zero_grad()
                critic_loss.backward()
                agent.criticNet.optimizer.step()

                # actor loss
                actor_loss = agent.criticNet.forward(cur_state_pre_processed, mu).squeeze()
                actor_loss = -T.mean(actor_loss)
                # actor optimization
                agent.actorNet.optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                # actor_loss.backward()
                agent.actorNet.optimizer.step()

                agent.update_count = agent.update_count + 1
                if agent.update_count == agent.target_update_step:
                    agent.update_network_parameters()  # soft-update is used here
                    print("{} network updated".format(agent.agent_name))
                    agent.update_count = 0  # reset update count

                critic_losses.append(critic_loss)
                actor_losses.append(actor_loss)

            return critic_losses, actor_losses

    def central_learning_v2(self, ReplayBuffer, batch_size, maxIntruNum, intruFeature):
        critic_losses, actor_losses = [], []

        cur_state, action, reward, next_state, done = ReplayBuffer.sample(batch_size, maxIntruNum, intruFeature,
                                                                          self.all_agents[0].max_grid_obs_dim)

        device = self.all_agents[0].actorNet.device

        # pre-process cur_state and next_state so that they can be used as input for every agent's critic network
        cur_state_pre_processed = preprocess_batch_for_critic_net_v2(cur_state, batch_size)
        next_state_pre_processed = preprocess_batch_for_critic_net_v2(next_state, batch_size)

        # load action, reward, done to tensor
        action = T.tensor(np.array(action), dtype=T.float).to(device)
        reward = T.tensor(np.array(reward), dtype=T.float).to(device)
        done = T.tensor(np.array(done)).to(device)

        # all these three different actions are needed to calculate the loss function
        all_agents_new_actions = []  # actions according to the target network for the new state
        all_agents_new_mu_actions = []  # actions according to the regular actor network for the current state
        old_agents_actions = []  # actions the agent actually took

        # this for loop to sort the action, reward, done, as they are in the form (agentNo X batch size X feature)
        for agent_idx, agent in self.all_agents.items():
            # actions according to the target network for the new state
            next_own = T.tensor(next_state[agent_idx][0], dtype=T.float).to(device)
            next_grid = T.tensor(next_state[agent_idx][1], dtype=T.float).to(device)
            next_nei = T.tensor(next_state[agent_idx][2], dtype=T.float).to(device)
            agent_new_states = [next_own, next_grid, next_nei]
            new_pi = agent.target_actorNet.forward(agent_new_states)  # individual agent's target network
            all_agents_new_actions.append(new_pi)  # record actions generated by each agent's target network

            # actions according to the regular actor network for the current state
            cur_own = T.tensor(cur_state[agent_idx][0], dtype=T.float).to(device)
            cur_grid = T.tensor(cur_state[agent_idx][1], dtype=T.float).to(device)
            cur_nei = T.tensor(cur_state[agent_idx][2], dtype=T.float).to(device)
            cur_states = [cur_own, cur_grid, cur_nei]

            # using agent's predict network to generate action based off current states
            pi = agent.actorNet.forward(cur_states)
            all_agents_new_mu_actions.append(pi)  # store the actions generated by actor's predict net based off cur_state

            # record actions the agent actually took
            old_agents_actions.append(action[agent_idx])

        new_actions = T.cat([acts for acts in all_agents_new_actions], dim=1)  # batch_size X agent_num X action dim
        old_actions = T.cat([acts for acts in old_agents_actions], dim=1)
        mu = T.cat([acts for acts in all_agents_new_mu_actions], dim=1)  # actions generate based off current agent's actor predict net

        for agent_idx, agent in self.all_agents.items():
            # zero grad()
            agent.criticNet.optimizer.zero_grad()
            agent.actorNet.optimizer.zero_grad()
            agent.criticNet.zero_grad()
            agent.actorNet.zero_grad()

            current_Q = agent.criticNet.forward(cur_state_pre_processed, old_actions)
            critic_value_prime = agent.target_criticNet.forward(next_state_pre_processed, new_actions)
            critic_value_prime[done[agent_idx]] = 0.0
            target_Q = reward[agent_idx] + agent.gamma * critic_value_prime

            loss_Q = nn.MSELoss()(current_Q, target_Q)
            loss_Q.backward()
            # torch.nn.utils.clip_grad_norm_(self.critics[agent].parameters(), 1)  # critic network grad clipping
            agent.criticNet.optimizer.step()

            # zero grad()
            agent.criticNet.optimizer.zero_grad()
            agent.actorNet.optimizer.zero_grad()
            agent.criticNet.zero_grad()
            agent.actorNet.zero_grad()

            # evaluate actor loss using current state and actor's critic
            actor_loss = -agent.criticNet.forward(cur_state_pre_processed, mu).squeeze().mean()
            actor_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.actors[agent].parameters(), 1)
            # torch.nn.utils.clip_grad_norm_(self.critics[agent].parameters(), 1)
            agent.actorNet.optimizer.step()

            critic_losses.append(loss_Q)
            actor_losses.append(actor_loss)

            agent.update_count = agent.update_count + 1
            if agent.update_count == agent.target_update_step:
                agent.update_network_parameters()  # soft-update is used here
                print("{} network updated".format(agent.agent_name))
                agent.update_count = 0  # reset update count

        return critic_losses, actor_losses

    def central_update(self, ReplayBuffer, batch_size, maxIntruNum, intruFeature, ts):
        critic_losses, actor_losses = [], []
        cur_state, action, reward, next_state, done = ReplayBuffer.sample(batch_size, maxIntruNum, intruFeature,
                                                                          self.all_agents[0].max_grid_obs_dim)
        FloatTensor = torch.FloatTensor
        device = self.all_agents[0].actorNet.device
        # load action, reward, done to tensor
        actionQ = T.tensor(np.array(action).transpose(1, 0, 2), dtype=T.float).contiguous().view(batch_size, -1).to(device)
        #reward = T.tensor(np.array(reward).transpose(1, 0, 2), dtype=T.float).contiguous().view(batch_size, -1).to(device)
        #done = T.tensor(np.array(done).transpose(1, 0, 2)).contiguous().view(batch_size, -1).to(device)
        next_ = T.tensor(np.array(next_state), dtype=T.float).to(device)

        # cur_ = T.tensor(np.array(cur_state), dtype=T.float).to(device)

        # pre-process cur_state and next_state so that they can be used as input for every agent's critic network
        cur_state_pre_processed = preprocess_batch_for_critic_net_v2(cur_state, batch_size)
        next_state_pre_processed = preprocess_batch_for_critic_net_v2(next_state, batch_size)

        all_agents_next_actions = []
        all_agents_new_mu_actions = []
        old_agents_actions = []  # actions the agent actually took

        # for agent_idx, agent in self.all_agents.items():  # for generate next actions
        #     # for next action, from next state go into actor's target net
        #     next_own = T.tensor(next_state[agent_idx], dtype=T.float).to(device)
        #     new_pi = agent.target_actorNet.forward(next_own)  # individual agent's target network
        #     all_agents_next_actions.append(new_pi)
        #     # for current action, from current state go into actor's prediction net
        #     cur_own = T.tensor(cur_state[agent_idx], dtype=T.float).to(device)
        #     pi = agent.actorNet(cur_own)
        #
        #     # pi = pi.clone()
        #
        #     all_agents_new_mu_actions.append(pi)
        #
        #     # record actions the agent actually took
        #     old_agents_actions.append(action[agent_idx])
        #
        # next_actions = T.cat([acts for acts in all_agents_next_actions], dim=1)
        # # next_actions = next_actions.clone()
        # mu = T.cat([acts for acts in all_agents_new_mu_actions], dim=1)
        # # might be necessary
        # # mu_one = mu.clone()
        # cur_action = T.cat([acts for acts in old_agents_actions], dim=1)

        for agent_idx, agent in self.all_agents.items():

            # current Q estimate, shape is batch_size X 1
            critic_value = agent.criticNet(cur_state_pre_processed, actionQ)

            critic_value_prime = torch.zeros(batch_size, 1).type(FloatTensor)
            next_actions = [self.all_agents[i].target_actorNet(next_[i, :, :]) for i in range(len(self.all_agents))]
            # critic_value_prime = agent.target_criticNet(next_state_pre_processed, next_actions).detach()
            next_action_stack = torch.stack(next_actions).permute(1, 0, 2).contiguous().view(batch_size, -1)

            mask = T.tensor(done[agent_idx]).int()
            flipped_mask = 1 - mask
            critic_value_prime = flipped_mask*agent.target_criticNet(next_state_pre_processed, next_action_stack)

            target = T.tensor(reward[agent_idx]) + agent.gamma * critic_value_prime

            critic_loss = F.mse_loss(critic_value, target.detach())
            agent.criticNet.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            # torch.nn.utils.clip_grad_norm_(agent.criticNet.parameters(), 1)
            agent.criticNet.optimizer.step()

            action_i = agent.actorNet(T.tensor(cur_state[agent_idx]))
            pi = T.tensor(action).clone()
            pi[agent_idx] = action_i  # only change the action batch with one update action
            mu = pi.view(batch_size, -1)

            # next_actions_cur = [self.all_agents[i].target_actorNet(cur_[i, :, :]) for i in range(len(self.all_agents))]
            # mu = torch.stack(next_actions_cur).permute(1, 0, 2).contiguous().view(batch_size, -1)

            actor_loss = -agent.criticNet(cur_state_pre_processed, mu).mean()
            agent.actorNet.optimizer.zero_grad()
            actor_loss.backward()
            # torch.nn.utils.clip_grad_norm_(agent.actorNet.parameters(), 1)
            # torch.nn.utils.clip_grad_norm_(agent.criticNet.parameters(), 1)
            agent.actorNet.optimizer.step()

            critic_losses.append(critic_loss)
            actor_losses.append(actor_loss)

            agent.update_count = agent.update_count + 1
            if agent.update_count == agent.target_update_step:
                agent.update_network_parameters()  # soft-update is used here
                print("{} network updated at episode equals to {}".format(agent.agent_name, ts))
                agent.update_count = 0  # reset update count after one update

        return critic_losses, actor_losses

    def central_update_v2(self, ReplayBuffer, batch_size, maxIntruNum, intruFeature):
        critic_losses, actor_losses = [], []
        cur_state, action, reward, next_state, done = ReplayBuffer.sample(batch_size, maxIntruNum, intruFeature,
                                                                          self.all_agents[0].max_grid_obs_dim)

        device = self.all_agents[0].actorNet.device
        # load action, reward, done to tensor
        action = T.tensor(np.array(action), dtype=T.float).to(device)
        reward = T.tensor(np.array(reward), dtype=T.float).to(device)
        done = T.tensor(np.array(done)).to(device)

        # pre-process cur_state and next_state so that they can be used as input for every agent's critic network
        cur_state_pre_processed = preprocess_batch_for_critic_net_v2(cur_state, batch_size)
        next_state_pre_processed = preprocess_batch_for_critic_net_v2(next_state, batch_size)

        all_agents_next_actions = []
        all_agents_new_mu_actions = []
        old_agents_actions = []  # actions the agent actually took

        for agent_idx, agent in self.all_agents.items():  # for generate next actions
            # for next action, from next state go into actor's target net
            next_own = T.tensor(next_state[agent_idx], dtype=T.float).to(device)
            new_pi = agent.target_actorNet.forward(next_own)  # individual agent's target network
            all_agents_next_actions.append(new_pi)
            # for current action, from current state go into actor's prediction net
            cur_own = T.tensor(cur_state[agent_idx], dtype=T.float).to(device)
            pi = agent.actorNet.forward(cur_own)
            all_agents_new_mu_actions.append(pi)

            # record actions the agent actually took
            old_agents_actions.append(action[agent_idx])

        next_actions = T.cat([acts for acts in all_agents_next_actions], dim=1)
        mu = T.cat([acts for acts in all_agents_new_mu_actions], dim=1)
        cur_action = T.cat([acts for acts in old_agents_actions], dim=1)

        for agent_idx, agent in self.all_agents.items():

            # current Q estimate, shape is batch_size X 1
            critic_value = agent.criticNet.forward(cur_state_pre_processed, cur_action)
            critic_value_prime = agent.target_criticNet.forward(next_state_pre_processed, next_actions)
            critic_value_prime[done[agent_idx]] = 0.0
            target = reward[agent_idx] + agent.gamma * critic_value_prime

            critic_loss = F.mse_loss(critic_value, target.detach())

            agent.criticNet.optimizer.zero_grad()
            critic_loss.backward()
            # torch.nn.utils.clip_grad_norm_(agent.criticNet.parameters(), 1)
            agent.criticNet.optimizer.step()

            actor_loss = -agent.criticNet(cur_state_pre_processed, mu).mean()
            agent.actorNet.optimizer.zero_grad()
            actor_loss.backward()
            # torch.nn.utils.clip_grad_norm_(agent.actorNet.parameters(), 1)
            # torch.nn.utils.clip_grad_norm_(agent.criticNet.parameters(), 1)
            agent.actorNet.optimizer.step()

            critic_losses.append(critic_loss)
            actor_losses.append(actor_loss)

            agent.update_count = agent.update_count + 1
            if agent.update_count == agent.target_update_step:
                agent.update_network_parameters()  # soft-update is used here
                print("{} network updated".format(agent.agent_name))
                agent.update_count = 0  # reset update count

        return critic_losses, actor_losses

    def cross_track_error(self, point, line):
        # Find the nearest point on the line to the given point
        nearest_pt = nearest_points(point, line)[1]

        # Calculate the cross-track distance
        distance = point.distance(nearest_pt)

        # Calculate the x and y components of the cross-track error
        x_error = abs(point.x - nearest_pt.x)
        y_error = abs(point.y - nearest_pt.y)

        return distance, x_error, y_error, nearest_pt

    def save_model_actor_net(self, file_path):
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        # only actor net is required to be saved, because when evaluation, only actor network is required
        for agent_idx, agent_obj in self.all_agents.items():
            torch.save(agent_obj.actorNet.state_dict(), file_path + '/' +agent_obj.agent_name + 'actor_net')
            # torch.save(agent_obj.target_actorNet.state_dict(), file_path + '/' +agent_obj.agent_name + 'target_actor_net')
            # torch.save(agent_obj.criticNet.state_dict(), file_path + '/' +agent_obj.agent_name + 'critic_net')
            # torch.save(agent_obj.target_criticNet.state_dict(), file_path + '/' +agent_obj.agent_name + 'target_critic_net')






































