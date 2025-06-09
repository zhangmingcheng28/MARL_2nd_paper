# -*- coding: utf-8 -*-
"""
@Time    : 27/5/2025 2:51 pm
@Author  : Mingcheng
@FileName: 
@Description: 
@Package dependency:
"""
# from Nnetworks_MADDPGv3 import CriticNetwork_0724, ActorNetwork
from Nnetworks_randomOD_radar_sur_drones_N_Model_BAv3 import *
from Utilities_own_randomOD_radar_sur_drones_N_Model_BAv3 import *
import torch
from copy import deepcopy
from torch.optim import Adam
from memory_randomOD_radar_sur_drones_N_Model_BAv3 import ReplayMemory, Experience
# from random_process_MADDPGv3_randomOD import OrnsteinUhlenbeckProcess
from torch.autograd import Variable
import os
import pandas as pd
import torch.nn as nn
import time
import numpy as np
import torch as T
# from utils_randomOD_radar_sur_drones_oneModel_use_tdCPA import device
import csv


class MADDPG_original:
    def __init__(self, actor_dim, critic_dim, dim_act, actor_hidden_state_size, gru_history_length, n_agents,
                 args, cr_lr, ac_lr, gamma, tau, full_observable_critic_flag, use_GRU_flag, use_single_portion_selfATT,
                 use_selfATT_with_radar, use_allNeigh_wRadar, own_obs_only, normalizer, device):
        self.args = args
        self.mode = args.mode
        self.actors = []
        self.critics = []
        self.device = device
        # original
        # self.actors = [Actor(dim_obs, dim_act) for _ in range(n_agents)]
        # self.critics = [Critic(n_agents, dim_obs, dim_act) for _ in range(n_agents)]

        # self.actors = [Stocha_actor(actor_dim, dim_act) for _ in range(n_agents)]  # use stochastic policy
        # self.actors = [ActorNetwork_TwoPortion(actor_dim, dim_act) for _ in range(n_agents)]  # use deterministic policy
        # only construct one-model
        if full_observable_critic_flag:
            # self.actors = ActorNetwork_allnei_wRadar(actor_dim, dim_act)
            self.actors = [ActorNetwork_allnei_wRadar(actor_dim, dim_act) for _ in range(n_agents)]
        else:
            if use_GRU_flag:
                self.actors = ActorNetwork_GRU_TwoPortion(actor_dim, dim_act, actor_hidden_state_size)
            elif use_single_portion_selfATT:
                self.actors = ActorNetwork_ATT(actor_dim, dim_act)
            elif use_allNeigh_wRadar:
                if own_obs_only:
                    self.actors = ActorNetwork_obs_only(actor_dim, dim_act)
                else:
                    self.actors = ActorNetwork_allnei_wRadar(actor_dim, dim_act)
            elif use_selfATT_with_radar:
                self.actors = ActorNetwork_ATT_wRadar(actor_dim, dim_act)
            else:
                self.actors = ActorNetwork_TwoPortion(actor_dim, dim_act)
        # end of only construct one-model
        # self.actors = [ActorNetwork_OnePortion(actor_dim, dim_act) for _ in range(n_agents)]  # use deterministic policy
        # self.actors = [GRUCELL_actor(actor_dim, dim_act, actor_hidden_state_size) for _ in range(n_agents)]  # use deterministic with GRU module policy
        # self.critics = [CriticNetwork_0724(critic_dim, n_agents, dim_act) for _ in range(n_agents)]
        # self.critics = [CriticNetwork(critic_dim, n_agents, dim_act) for _ in range(n_agents)]
        # self.critics = [CriticNetwork_wGru(critic_dim, n_agents, dim_act, gru_history_length) for _ in range(n_agents)]
        if full_observable_critic_flag:
            # self.critics = [critic_combine_TwoPortion(critic_dim, n_agents, dim_act, gru_history_length,
            #                                           actor_hidden_state_size) for _ in range(n_agents)]

            # one model centralized critic
            self.critics = critic_combine_TwoPortion_fullneiWradar(critic_dim, n_agents, dim_act, gru_history_length, actor_hidden_state_size, normalizer, device)
        else:
            # self.critics = [critic_single_TwoPortion(critic_dim, n_agents, dim_act, gru_history_length, actor_hidden_state_size) for _ in range(n_agents)]
            # only construct one-model
            if use_GRU_flag:
                self.critics = critic_single_GRU_TwoPortion(critic_dim, n_agents, dim_act, gru_history_length,
                                                        actor_hidden_state_size)
            elif use_single_portion_selfATT:
                self.critics = critic_single_TwoPortion(critic_dim, n_agents, dim_act, gru_history_length,
                                                        actor_hidden_state_size)
            elif use_allNeigh_wRadar:
                if own_obs_only:
                    self.critics = critic_single_obs_only(critic_dim, n_agents, dim_act, gru_history_length,
                                                        actor_hidden_state_size)
                else:
                    self.critics = critic_single_TwoPortion_wRadar(critic_dim, n_agents, dim_act, gru_history_length,
                                                        actor_hidden_state_size)
            elif use_selfATT_with_radar:
                self.critics = critic_single_TwoPortion_wRadar(critic_dim, n_agents, dim_act, gru_history_length,
                                                        actor_hidden_state_size)
            else:
                self.critics = critic_single_TwoPortion(critic_dim, n_agents, dim_act, gru_history_length, actor_hidden_state_size)
            # end of only construct one-model
        # self.critics = [critic_single_OnePortion(critic_dim, n_agents, dim_act, gru_history_length, actor_hidden_state_size) for _ in range(n_agents)]

        self.n_agents = n_agents
        self.n_actor_dim = actor_dim
        self.n_critic_dim = critic_dim
        self.n_actions = dim_act

        # self.actors_target = deepcopy(self.actors)
        self.actors_target = deepcopy(self.actors)
        self.critics_target = deepcopy(self.critics)

        self.memory = ReplayMemory(args.memory_length)
        self.batch_size = args.batch_size
        self.use_cuda = torch.cuda.is_available()
        # self.episodes_before_train = args.episode_before_train

        # self.GAMMA = 0.95  # original
        # self.tau = 0.01  # original

        self.GAMMA = gamma
        self.tau = tau

        self.var = [1.0 for i in range(n_agents)]
        # self.var = [0.5 for i in range(n_agents)]

        # original, critic learning rate is 10 times larger compared to actor
        # self.critic_optimizer = [Adam(x.parameters(), lr=0.001) for x in self.critics]
        # self.actor_optimizer = [Adam(x.parameters(), lr=0.0001) for x in self.actors]

        # self.critic_optimizer = [Adam(x.parameters(), lr=cr_lr) for x in self.critics]
        # self.actor_optimizer = [Adam(x.parameters(), lr=ac_lr) for x in self.actors]

        # only construct one-model
        # self.critic_optimizer = Adam(self.critics.parameters(), lr=cr_lr)
        self.critic_optimizer = [Adam(x.parameters(), lr=cr_lr) for x in self.critics] if isinstance(self.critics, list) else Adam(self.critics.parameters(), lr=cr_lr)
        # self.actor_optimizer = Adam(self.actors.parameters(), lr=ac_lr)
        self.actor_optimizer = [Adam(x.parameters(), lr=ac_lr) for x in self.actors] if isinstance(self.actors, list) else Adam(self.actors.parameters(), lr=ac_lr)
        # end of only construct one-model

        if self.device.type == 'cuda':
            if isinstance(self.actors, list):
                for act_model_idx, act_model in enumerate(self.actors):
                    self.actors[act_model_idx].cuda().to(dtype=torch.float64)
                    self.actors_target[act_model_idx].cuda().to(dtype=torch.float64)
            else:
                self.actors.cuda().to(dtype=torch.float64)
                self.actors_target.cuda().to(dtype=torch.float64)

            if isinstance(self.critics, list):
                for critic_model_idx, critic_model in enumerate(self.critics):
                    self.critics[critic_model_idx].cuda().to(dtype=torch.float64)
                    self.critics_target[critic_model_idx].cuda().to(dtype=torch.float64)
            else:
                self.critics.cuda().to(dtype=torch.float64)
                self.critics_target.cuda().to(dtype=torch.float64)
        else:
            if isinstance(self.actors, list):
                for act_model_idx, act_model in enumerate(self.actors):
                    self.actors[act_model_idx].to(dtype=torch.float64)
                    self.actors_target[act_model_idx].to(dtype=torch.float64)
            else:
                self.actors.to(dtype=torch.float64)
                self.actors_target.to(dtype=torch.float64)

            if isinstance(self.critics, list):
                for critic_model_idx, critic_model in enumerate(self.critics):
                    self.critics[critic_model_idx].to(dtype=torch.float64)
                    self.critics_target[critic_model_idx].to(dtype=torch.float64)
            else:
                self.critics.to(dtype=torch.float64)
                self.critics_target.to(dtype=torch.float64)

        self.steps_done = 0
        self.episode_done = 0