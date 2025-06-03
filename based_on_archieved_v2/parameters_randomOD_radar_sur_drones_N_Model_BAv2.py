# -*- coding: utf-8 -*-
"""
@Time    : 3/1/2023 7:58 PM
@Author  : Mingcheng
@FileName: 
@Description: 
@Package dependency:
"""
import numpy as np
from pc_specfic_config import PCconfig
from grid_env_generation_newframe_randomOD_radar_sur_drones_N_Model_BAv2 import env_generation
from env_simulator_randomOD_radar_sur_drones_N_Model_BAv2 import env_simulator


def initialize_parameters():
    eps_start = 1.0
    eps_end = 0.05  # The minimum number of epsilon
    #eps_decay = 0.99992
    eps_period = 8500  # The number of steps needed for the epsilon to drop until the minimum number of the "eps_end"
    eps = eps_start  # initialize epsilon
    agent_grid_obs = np.zeros((7, 7))
    agent_obs_dim = 6  # Vx, Vy, delta_Gx, delta_Gy, Acc_x, Acc_y
    BUFFER_SIZE = int(1e6)  # replay buffer size
    # BUFFER_SIZE = int(1e5)  # replay buffer size
    # BUFFER_SIZE = int(5e4)  # replay buffer size
    BATCH_SIZE = 256  # minibatch size
    # GAMMA = 0.90  # discount factor
    GAMMA = 0.95  # discount factor
    TAU = 0.01  # for soft update of target parameters, 0.001, so 99.9% of the weights in the target network is
    UPDATE_EVERY = 1  # how often to update the network

    # set boundary
    xlow = 455
    xhigh = 680
    ylow = 255
    yhigh = 385
    bound = [xlow, xhigh, ylow, yhigh]

    # generate static env from shape file
    # shapePath = 'D:\deep_Q_learning\DQN_new_framework\lakesideMap\lakeSide.shp'
    shapePath = PCconfig().shape_path
    # shapePath = 'F:\githubClone\deep_Q_learning\DQN_new_framework\lakesideMap\lakeSide.shp'
    # shapePath = 'D:\github_clone\Multi_agent_AAC\MA_ver1\lakesideMap\lakeSide.shp'
    staticEnv = env_generation(shapePath, bound)  # it is a tuple of 4 element, 1st is the 2D binary array of the filled map, 2nd is the list of all buildings expressed as polygons, 3rd is the gird length, 4th element is a list of length 1, inside has 2 element. 0th element is the occupied polygon, 1st element is the unoccupied polygon
    seed = 3407  # this seed is only used for torch manuel.seed
    # set boundary
    xlow = 455
    xhigh = 680
    ylow = 255
    yhigh = 385
    bound = [xlow, xhigh, ylow, yhigh]
    # agent config address
    # read the Excel file into a pandas dataframe
    # agentConfig = (r'F:\githubClone\Multi_agent_AAC\MA_ver1\fixedDrone.xlsx')
    # agentConfig = (r'F:\githubClone\Multi_agent_AAC\MA_ver1\fixedDrone_5_adj.xlsx')
    agentConfig = (r'F:\githubClone\Multi_agent_AAC\MA_ver1\fixedDrone_3drones.xlsx')
    # agentConfig = (r'F:\githubClone\Multi_agent_AAC\MA_ver1\fixedDrone_2_drone.xlsx')
    # agentConfig = (r'F:\githubClone\Multi_agent_AAC\MA_ver1\reward_test.xlsx')  # for perform reward testing
    # agentConfig = (r'D:\Multi_agent_AAC\MA_ver1\fixedDrone.xlsx')
    # agentConfig = (r'D:\Multi_agent_AAC\MA_ver1\fixedDrone_3drones.xlsx')
    # agentConfig = (r'D:\Multi_agent_AAC\MA_ver1\fixedDrone_2_drone.xlsx')
    # agentConfig = (r'D:\github_clone\Multi_agent_AAC\MA_ver1\fixedDrone_5_adj.xlsx')
    env = env_simulator(staticEnv[0], staticEnv[1], staticEnv[2], bound, staticEnv[3], agentConfig)
    return eps_start, eps_end, eps_period, eps, env, agent_grid_obs, BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, UPDATE_EVERY, seed, staticEnv[-1]
