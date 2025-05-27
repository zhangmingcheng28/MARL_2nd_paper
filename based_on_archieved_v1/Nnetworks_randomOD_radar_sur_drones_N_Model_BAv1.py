# -*- coding: utf-8 -*-
"""
@Time    : 3/3/2023 10:34 AM
@Author  : Mingcheng
@FileName: 
@Description: 
@Package dependency:
"""
import torch as T
import torch.nn as nn
from Utilities_own_randomOD_radar_sur_drones_N_Model_BAv1 import *
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np
import math
from torch.distributions import Normal
# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda" if use_cuda else "cpu")

class SelfAttention(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.K = nn.Linear(input_dim, input_dim)
        self.Q = nn.Linear(input_dim, input_dim)
        self.V = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        batch_size, feature_dim = x.shape

        # Apply linear transformations
        K = self.K(x)
        Q = self.Q(x)
        V = self.V(x)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(feature_dim, dtype=torch.float32))
        attention_weights = nn.functional.softmax(scores, dim=-1)

        # Apply attention to values
        values = torch.matmul(attention_weights, V)

        return values


class ActorNetwork(nn.Module):
    def __init__(self, actor_dim, n_actions):  # actor_obs consists of three parts 0 = own, 1 = own grid, 2 = surrounding drones
        super(ActorNetwork, self).__init__()

        # self.n_heads = 3
        # self.single_head_dim = int((64+64+64) / self.n_heads)

        # self.own_fc = nn.Sequential(nn.Linear(actor_dim[0], 128), nn.ReLU())
        self.own_fc = nn.Sequential(nn.Linear(actor_dim[0], 64), nn.ReLU())
        # self.own_fc_lay2 = nn.Sequential(nn.Linear(256, 256), nn.ReLU())
        # self.own_fc_lay2 = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())

        # gru layer
        # self.gru = nn.GRU(256, 256, 1, batch_first=True)

        self.own_fc_lay2 = nn.Sequential(nn.Linear(64, 64), nn.ReLU(),
                                         nn.Linear(64, 64), nn.ReLU())
        # self.own_fc_lay2 = nn.Sequential(nn.Linear(256, 256), nn.ReLU())

        self.own_fc_outlay = nn.Sequential(nn.Linear(64, n_actions), nn.Tanh())
        # self.intrude_fc = nn.Sequential(nn.Linear(6, 256), nn.ReLU())
        # self.own_grid_fc = nn.Sequential(nn.Linear(actor_dim[1], 128), nn.ReLU())

        # perform a self-attention for own obs_grids, actor_obs[1], assume actor_obs = [6, 6, 6]

        # self.grid_K = nn.Linear(actor_obs[1], 64)
        # self.grid_Q = nn.Linear(actor_obs[1], 64)
        # self.grid_V = nn.Linear(actor_obs[1], 64)

        # self.own_grid = nn.Sequential(nn.Linear(actor_obs[1], 64), nn.ReLU())

        # self.surr_drone = nn.Sequential(nn.Linear(max_nei_num * actor_obs[2], 64), nn.ReLU(),
        #                                 nn.Linear(64, 64), nn.ReLU())
        # self-attention for 2D grids that has arbitrary length after flattened, with 2 head.

        # # use attention here
        # self.combine_att_xe = nn.Sequential(nn.Linear(64+64+64, 128), nn.ReLU())
        # # NOTE: For the com_k,q,v here, they are used for "single head" attention calculation, so we only use
        # # dimension of the single_head_dim, in the linear transformation.
        #
        # self.com_k = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        # self.com_q = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        # self.com_v = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        #
        # self.multi_att_out = nn.Linear(self.n_heads * self.single_head_dim, 128)
        #
        # self.action_out = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, n_actions), nn.Tanh())
        #
        # self.action_out_V3 = nn.Sequential(nn.Linear(64+64+64, 64), nn.ReLU(), nn.Linear(64, n_actions), nn.Tanh())
        # self.action_out_V4 = nn.Sequential(nn.Linear(64+64, 64), nn.ReLU(), nn.Linear(64, n_actions), nn.Tanh())
        # self.action_out_V5 = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, n_actions), nn.Tanh())
        # self.action_out_V5_1 = nn.Sequential(nn.Linear(128+128+128, 128), nn.ReLU(), nn.Linear(128, n_actions), nn.Tanh())
        # self.action_out_V5_1 = nn.Sequential(nn.Linear(256+256, 256), nn.ReLU(), nn.Linear(256, n_actions), nn.Tanh())
        # self.action_out_V5_2 = nn.Sequential(nn.Linear(128+128+128, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(),
        #                                      nn.Linear(512, n_actions), nn.Tanh())

        # attention for NN
        # self.k = nn.Linear(256, 256, bias=False)
        # self.q = nn.Linear(256, 256, bias=False)
        # self.v = nn.Linear(256, 256, bias=False)

        # self.name = name

        # self.optimizer = optim.Adam(self.parameters(), lr=actor_lr)
        # self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        #
        # self.to(self.device)

    def forward(self, state):
        own_e = self.own_fc(state[0])

        # gru_output, hn = self.gru(own_e)  # h0 is not necessary, as it auto set to 0.

        own_hidden = self.own_fc_lay2(own_e)
        # own_hidden = self.own_fc_lay2(hn)

        action_out = self.own_fc_outlay(own_hidden)
        # x_e = self.intrude_fc(state[1])
        # #
        # # # mask attention embedding
        # q = self.q(own_e)
        # k = self.k(x_e)
        # v = self.v(x_e)
        # mask = state[1].mean(axis=2, keepdim=True).bool()
        # score = torch.bmm(k, q.unsqueeze(axis=2))
        # score_mask = score.clone()  # clone操作很必要
        # score_mask[~mask] = float('-inf')  # 不然赋值操作后会无法计算梯度
        # #
        # alpha = F.softmax(score_mask / np.sqrt(k.size(-1)), dim=1)  # we use dim=1 here because we need to get attention of each sequence in K towards all hidden vector of q in each batch.
        # alpha_mask = alpha.clone()
        # alpha_mask[~mask] = 0
        # v_att = torch.sum(v * alpha_mask, axis=1)
        # #
        # concat = torch.cat((own_hidden, v_att), dim=1)
        # action_out = self.action_out_V5_1(concat)
        return action_out


class ActorNetwork_TwoPortion(nn.Module):
    def __init__(self, actor_dim, n_actions):  # actor_obs consists of three parts 0 = own, 1 = own grid, 2 = surrounding drones
        super(ActorNetwork_TwoPortion, self).__init__()

        self.own_fc = nn.Sequential(nn.Linear(actor_dim[0], 64), nn.ReLU())
        # self.own_fc = nn.Sequential(nn.Linear(actor_dim[0], 128), nn.ReLU())
        self.own_grid = nn.Sequential(nn.Linear(actor_dim[1], 64), nn.ReLU())
        # self.own_grid = nn.Sequential(nn.Linear(actor_dim[1], 128), nn.ReLU())
        self.merge_feature = nn.Sequential(nn.Linear(64+64, 128), nn.ReLU())

        # self.merge_feature_l2 = nn.Sequential(nn.Linear(64 + 64, 128), nn.ReLU())
        # self.merge_feature = nn.Sequential(nn.Linear(128+128, 256), nn.ReLU())
        self.act_out = nn.Sequential(nn.Linear(128, n_actions), nn.Tanh())
        # self.act_out = nn.Sequential(nn.Linear(128, n_actions))
        # self.act_out = nn.Sequential(nn.Linear(256, n_actions), nn.Tanh())

        # ignore radar
        # self.own_fc = nn.Sequential(nn.Linear(actor_dim[0], 128), nn.ReLU())
        # self.hidden = nn.Sequential(nn.Linear(128, 128), nn.ReLU())
        # # self.hidden = nn.Sequential(nn.Linear(128, 128), nn.ReLU(),
        # #                             nn.Linear(128, 128), nn.ReLU(),
        # #                             nn.Linear(128,128), nn.ReLU())
        # self.act_out = nn.Sequential(nn.Linear(128, n_actions), nn.Tanh())
        # # self.act_out = nn.Sequential(nn.Linear(128, n_actions))

        # # ignore radar but single out relative distance, actually same NN as with radar.
        # self.own_fc = nn.Sequential(nn.Linear(actor_dim[0], 64), nn.ReLU())
        # # self.own_fc = nn.Sequential(nn.Linear(actor_dim[0], 128), nn.ReLU())
        # self.own_grid = nn.Sequential(nn.Linear(actor_dim[1], 64), nn.ReLU())
        # # self.own_grid = nn.Sequential(nn.Linear(actor_dim[1], 128), nn.ReLU())
        # self.merge_feature = nn.Sequential(nn.Linear(64+64, 128), nn.ReLU())
        # # self.merge_feature_l2 = nn.Sequential(nn.Linear(64 + 64, 128), nn.ReLU())
        # # self.merge_feature = nn.Sequential(nn.Linear(128+128, 256), nn.ReLU())
        # self.act_out = nn.Sequential(nn.Linear(128, n_actions), nn.Tanh())
        # # self.act_out = nn.Sequential(nn.Linear(256, n_actions), nn.Tanh())

    def forward(self, cur_state):
        own_obs = self.own_fc(cur_state[0])
        own_grid = self.own_grid(cur_state[1])
        merge_obs_grid = torch.cat((own_obs, own_grid), dim=1)
        merge_feature = self.merge_feature(merge_obs_grid)
        out_action = self.act_out(merge_feature)

        # ignore radar
        # own_obs = self.own_fc(cur_state[0])
        # hidden = self.hidden(own_obs)
        # out_action = self.act_out(hidden)

        # # ignore radar but single out relative distance, actually same NN as with radar.
        # own_obs = self.own_fc(cur_state[0])
        # own_grid = self.own_grid(cur_state[1])
        # merge_obs_grid = torch.cat((own_obs, own_grid), dim=1)
        # merge_feature = self.merge_feature(merge_obs_grid)
        # out_action = self.act_out(merge_feature)

        return out_action


class ActorNetwork_ATT(nn.Module):
    def __init__(self, actor_dim, n_actions):
        super(ActorNetwork_ATT, self).__init__()

        self.own_fc = nn.Sequential(nn.Linear(actor_dim[0], 64), nn.ReLU())
        self.own_grid = nn.Sequential(nn.Linear(actor_dim[1], 64), nn.ReLU())
        self.merge_feature = nn.Sequential(nn.Linear(64+64, 128), nn.ReLU())
        self.act_out = nn.Sequential(nn.Linear(128, n_actions), nn.Tanh())
        # attention
        self.k = nn.Linear(64, 64, bias=False)
        self.q = nn.Linear(64, 64, bias=False)
        self.v = nn.Linear(64, 64, bias=False)

    def forward(self, cur_state):
        if len(cur_state[1].shape) == 2:
            cur_state[1] = cur_state[1].unsqueeze(1)  # insert an one dimension at middle, because we want batch size always at 0th dimension
        own_obs = self.own_fc(cur_state[0])
        own_nei = self.own_grid(cur_state[1])
        # mask attention embedding
        q = self.q(own_obs)
        k = self.k(own_nei)
        v = self.v(own_nei)
        score = torch.bmm(k, q.unsqueeze(axis=2))
        alpha = F.softmax(score / np.sqrt(k.size(-1)), dim=1)
        v_att = torch.sum(v * alpha, axis=1)
        merge_obs_grid = torch.cat((own_obs, v_att), dim=1)
        merge_feature = self.merge_feature(merge_obs_grid)
        out_action = self.act_out(merge_feature)
        return out_action


class ActorNetwork_ATT_wRadar(nn.Module):
    def __init__(self, actor_dim, n_actions):
        super(ActorNetwork_ATT_wRadar, self).__init__()
        self.own_fc = nn.Sequential(nn.Linear(actor_dim[0], 64), nn.ReLU())
        self.own_full_nei = nn.Sequential(nn.Linear(actor_dim[1], 64), nn.ReLU())
        self.own_grid = nn.Sequential(nn.Linear(actor_dim[2], 64), nn.ReLU())
        self.merge_feature = nn.Sequential(nn.Linear(64+64+64, 256), nn.ReLU())
        self.act_out = nn.Sequential(nn.Linear(256, n_actions), nn.Tanh())
        # attention
        self.k = nn.Linear(64, 64, bias=False)
        self.q = nn.Linear(64, 64, bias=False)
        self.v = nn.Linear(64, 64, bias=False)

    def forward(self, cur_state):
        if len(cur_state[1].shape) == 2:
            cur_state[1] = cur_state[1].unsqueeze(1)  # insert an one dimension at middle, because we want batch size always at 0th dimension
        own_obs = self.own_fc(cur_state[0])
        own_nei = self.own_full_nei(cur_state[1])
        own_radar = self.own_grid(cur_state[2])
        # mask attention embedding
        q = self.q(own_obs)
        k = self.k(own_nei)
        v = self.v(own_nei)
        mask = cur_state[1].mean(axis=2, keepdim=True).bool()
        score = torch.bmm(k, q.unsqueeze(axis=2))
        score_mask = score.clone()  # clone操作很必要
        score_mask[~mask] = float('-inf')  # 不然赋值操作后会无法计算梯度

        alpha = F.softmax(score_mask / np.sqrt(k.size(-1)), dim=1)  # we use dim=1 here because we need to get attention of each sequence in K towards all hidden vector of q in each batch.
        alpha_mask = alpha.clone()
        alpha_mask[~mask] = 0
        v_att = torch.sum(v * alpha_mask, axis=1)

        merge_obs_grid = torch.cat((own_obs, v_att, own_radar), dim=1)
        merge_feature = self.merge_feature(merge_obs_grid)
        out_action = self.act_out(merge_feature)
        return out_action


class ActorNetwork_allnei_wRadar(nn.Module):
    def __init__(self, actor_dim, n_actions):
        super(ActorNetwork_allnei_wRadar, self).__init__()
        self.own_fc = nn.Sequential(nn.Linear(actor_dim[0], 64), nn.ReLU())
        self.own_full_nei = nn.Sequential(nn.Linear(actor_dim[1], 64), nn.ReLU())
        self.own_grid = nn.Sequential(nn.Linear(actor_dim[2], 64), nn.ReLU())
        self.merge_feature = nn.Sequential(nn.Linear(64+64+64, 256), nn.ReLU())
        self.act_out = nn.Sequential(nn.Linear(256, 256), nn.ReLU(),
                                     nn.Linear(256, n_actions), nn.Tanh())
        # self.own_fc = nn.Sequential(nn.Linear(actor_dim[0], 64), nn.LeakyReLU(0.01))
        # self.own_full_nei = nn.Sequential(nn.Linear(actor_dim[1], 64), nn.LeakyReLU(0.01))
        # self.own_grid = nn.Sequential(nn.Linear(actor_dim[2], 64), nn.LeakyReLU(0.01))
        # self.merge_feature = nn.Sequential(nn.Linear(64+64+64, 256), nn.LeakyReLU(0.01))
        # self.act_out = nn.Sequential(nn.Linear(256, 256), nn.LeakyReLU(0.01),
        #                              nn.Linear(256, n_actions), nn.Tanh())

        # self.own_fc = nn.Sequential(nn.Linear(actor_dim[0], 64), nn.ReLU())
        # self.own_full_nei = nn.Sequential(nn.Linear(actor_dim[1], 256), nn.ReLU())
        # self.own_grid = nn.Sequential(nn.Linear(actor_dim[2], 128), nn.ReLU())
        # self.merge_feature = nn.Sequential(nn.Linear(64+256+128, 1024), nn.ReLU())
        # self.act_out = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(),
        #                              nn.Linear(512, n_actions), nn.Tanh())

        # self.own_fc = nn.Sequential(nn.Linear(actor_dim[0], 64), nn.ReLU())
        # self.own_full_nei = nn.Sequential(nn.Linear(actor_dim[1], 64), nn.ReLU())
        # self.own_grid = nn.Sequential(nn.Linear(actor_dim[2], 64), nn.ReLU())
        # self.merge_feature = nn.Sequential(nn.Linear(64+64+64, 512), nn.ReLU(),
        #                                    nn.Linear(512, 512), nn.ReLU(),
        #                                    nn.Linear(512, 512), nn.ReLU())
        # self.act_out = nn.Sequential(nn.Linear(512, 512), nn.ReLU(),
        #                              nn.Linear(512, n_actions), nn.Tanh())

        # self.own_fc = nn.Sequential(nn.Linear(actor_dim[0], 256), nn.ReLU())
        # self.own_full_nei = nn.Sequential(nn.Linear(actor_dim[1], 256), nn.ReLU())
        # self.own_grid = nn.Sequential(nn.Linear(actor_dim[2], 256), nn.ReLU())
        # self.merge_feature = nn.Sequential(nn.Linear(256+256+256, 1024), nn.ReLU())
        #                                    # nn.Linear(512, 512), nn.ReLU(),
        #                                    # nn.Linear(512, 512), nn.ReLU())
        # self.act_out = nn.Sequential(nn.Linear(1024, 1024), nn.ReLU(),
        #                              nn.Linear(512, n_actions), nn.Tanh())

        # --to compare
        # self.own_fc = nn.Sequential(nn.Linear(actor_dim[0], 64), nn.ReLU())
        # # self.own_grid = nn.Sequential(nn.Linear(actor_dim[2], 64), nn.ReLU())
        # # self.merge_feature = nn.Sequential(nn.Linear(64+64, 128), nn.ReLU())
        # self.merge_feature = nn.Sequential(nn.Linear(64, 128), nn.ReLU())
        # self.act_out = nn.Sequential(nn.Linear(128, n_actions), nn.Tanh())
        #-- end of to compare

    def forward(self, cur_state):
        own_obs = self.own_fc(cur_state[0])
        own_nei = self.own_full_nei(cur_state[1])
        own_radar = self.own_grid(cur_state[2])
        merge_obs_grid = torch.cat((own_obs, own_nei, own_radar), dim=1)
        # merge_obs_grid = torch.cat((own_obs, own_radar), dim=1)
        merge_feature = self.merge_feature(merge_obs_grid)
        out_action = self.act_out(merge_feature)

        # to compare
        # own_obs = self.own_fc(cur_state[0])
        # # own_grid = self.own_grid(cur_state[2])
        # # merge_obs_grid = torch.cat((own_obs, own_grid), dim=1)
        # merge_feature = self.merge_feature(own_obs)
        # out_action = self.act_out(merge_feature)
        # end of to compare
        return out_action


class ActorNetwork_obs_only(nn.Module):
    def __init__(self, actor_dim, n_actions):
        super(ActorNetwork_obs_only, self).__init__()
        self.own_fc = nn.Sequential(nn.Linear(actor_dim[0], 64), nn.ReLU())
        self.act_out = nn.Sequential(nn.Linear(64, 64), nn.ReLU(),
                                     nn.Linear(64, n_actions), nn.Tanh())

    def forward(self, cur_state):
        own_obs = self.own_fc(cur_state[0])
        out_action = self.act_out(own_obs)
        return out_action



class ActorNetwork_GRU_TwoPortion(nn.Module):
    def __init__(self, actor_dim, n_actions, actor_hidden_state_size):  # actor_obs consists of three parts 0 = own, 1 = own grid, 2 = surrounding drones
        super(ActorNetwork_GRU_TwoPortion, self).__init__()

        self.own_fc = nn.Sequential(nn.Linear(actor_dim[0], 64), nn.ReLU())
        # self.own_fc = nn.Sequential(nn.Linear(actor_dim[0], 128), nn.ReLU())
        self.own_grid = nn.Sequential(nn.Linear(actor_dim[1], 64), nn.ReLU())
        # self.own_grid = nn.Sequential(nn.Linear(actor_dim[1], 128), nn.ReLU())
        self.merge_feature = nn.Sequential(nn.Linear(64+64, 128), nn.ReLU())
        self.rnn_hidden_dim = actor_hidden_state_size
        self.gru_cell = nn.GRUCell(64, actor_hidden_state_size)
        # self.merge_feature = nn.Sequential(nn.Linear(128+128, 256), nn.ReLU())
        self.act_out = nn.Sequential(nn.Linear(128, n_actions), nn.Tanh())
        # self.act_out = nn.Sequential(nn.Linear(256, n_actions), nn.Tanh())

    def forward(self, cur_state, history_hidden_state):
        own_obs = self.own_fc(cur_state[0])
        own_grid = self.own_grid(cur_state[1])
        h_in = history_hidden_state.reshape(-1, self.rnn_hidden_dim)
        h_out = self.gru_cell(own_grid, h_in)  # gru apply to 2nd portion, neigh
        # h_out = self.gru_cell(own_obs, h_in)  # gru apply to 1st portion, own_obs
        merge_obs_grid = torch.cat((own_obs, h_out), dim=1)  # gru apply to 2nd portion, neigh
        # merge_obs_grid = torch.cat((own_grid, h_out), dim=1)  # gru apply to 1st portion, own_obs
        merge_feature = self.merge_feature(merge_obs_grid)
        out_action = self.act_out(merge_feature)
        return out_action, h_out


class ActorNetwork_OnePortion(nn.Module):
    def __init__(self, actor_dim, n_actions):  # actor_obs consists of three parts 0 = own, 1 = own grid, 2 = surrounding drones
        super(ActorNetwork_OnePortion, self).__init__()

        self.own_fcWgrid = nn.Sequential(nn.Linear(actor_dim[0]+actor_dim[1], 64), nn.ReLU())
        self.merge_feature = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        self.act_out = nn.Sequential(nn.Linear(64, n_actions), nn.Tanh())

    def forward(self, cur_state):
        own_obsWgrid = torch.cat((cur_state[0], cur_state[1]), dim=1)
        obsWgrid_feat = self.own_fcWgrid(own_obsWgrid)
        merge_feature = self.merge_feature(obsWgrid_feat)
        out_action = self.act_out(merge_feature)
        return out_action


class GRU_actor(nn.Module):
    def __init__(self, actor_dim, n_actions, actor_hidden_state_size):
        super(GRU_actor, self).__init__()
        self.own_fc = nn.Sequential(nn.Linear(actor_dim[0], 64), nn.ReLU())
        # gru layer
        self.gru = nn.GRU(actor_dim[0], actor_hidden_state_size, 1, batch_first=True)

        self.own_fc_outlay = nn.Sequential(nn.Linear(64+64, n_actions), nn.Tanh())

    def forward(self, cur_state, history_info):
        own_e = self.own_fc(cur_state)
        gru_out, hn = self.gru(history_info)  # hn = last column (or the most recent one) of the output hidden state
        hn_owne = torch.cat((own_e, hn.squeeze(0)),dim=1)
        action_out = self.own_fc_outlay(hn_owne)
        return action_out, hn


class GRUCELL_actor(nn.Module):
    def __init__(self, actor_dim, n_actions, actor_hidden_state_size):
        super(GRUCELL_actor, self).__init__()
        self.rnn_hidden_dim = actor_hidden_state_size
        self.own_fc = nn.Sequential(nn.Linear(actor_dim[0], 64), nn.ReLU())
        self.gru_cell = nn.GRUCell(64, actor_hidden_state_size)
        self.own_fc_outlay = nn.Sequential(nn.Linear(64, n_actions), nn.Tanh())

    def forward(self, cur_state, history_hidden_state):
        own_e = self.own_fc(cur_state)
        h_in = history_hidden_state.reshape(-1, self.rnn_hidden_dim)
        h = self.gru_cell(own_e, h_in)
        action_out = self.own_fc_outlay(h)
        return action_out, h


class Stocha_actor(nn.Module):
    def __init__(self, actor_dim, n_actions):  # actor_obs consists of three parts 0 = own, 1 = own grid, 2 = surrounding drones
        super(Stocha_actor, self).__init__()
        init_w = 3e-3
        self.own_fc = nn.Sequential(nn.Linear(actor_dim[0], 256), nn.ReLU())
        self.own_fc_lay2 = nn.Sequential(nn.Linear(256, 256), nn.ReLU(),
                                         nn.Linear(256, 256), nn.ReLU(),
                                         nn.Linear(256, 256), nn.ReLU())
        self.log_std_min = -20
        self.log_std_max = 2
        self.mean_linear = nn.Linear(256, n_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)

        self.log_std_linear = nn.Linear(256, n_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        own_e = self.own_fc(state[0])
        own_hidden = self.own_fc_lay2(own_e)
        mean = self.mean_linear(own_hidden)
        log_std = self.log_std_linear(own_hidden)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()
        normal = Normal(0, 1)  # this is the re-parameterization trick
        z = normal.sample(mean.shape)
        action = torch.tanh(mean + std * z.to(device))
        return action


class CriticNetwork(nn.Module):
    def __init__(self, critic_obs, n_agents, n_actions):
        super(CriticNetwork, self).__init__()

        # in critic network we should use multi-head attention mechanism to help to capture more complex relationship
        # between different inputs, in the context of this paper, the input consists of many drone's states as well as
        # their actions. This is two group of inputs, therefore my hypothesis is that using multi-head attention is
        # better here.

        # critic_obs[0] is sum of all agent's own observed states
        # critic_obs[1] is sum of all agent's observed grid maps
        # critic_obs[3] is sum of all agent's action taken

        # self.sum_own_fc = nn.Sequential(nn.Linear(critic_obs*n_agents, 1024), nn.ReLU())  # may be here can be replaced with another attention mechanism
        # self.sum_own_fc = nn.Sequential(nn.Linear(critic_obs[0]*n_agents, 256), nn.ReLU())  # may be here can be replaced with another attention mechanism
        # self.sum_grid_fc = nn.Sequential(nn.Linear(critic_obs[1]*n_agents, 128), nn.ReLU())
        # self.sum_combine_fc = nn.Sequential(nn.Linear((critic_obs[0]+critic_obs[1])*n_agents, 128), nn.ReLU())
        self.sum_combine_fc = nn.Sequential(nn.Linear((critic_obs[0])*n_agents, 256), nn.ReLU())
        # self.sum_combine_fc = nn.Sequential(nn.Linear((critic_obs[0])*n_agents, 2048), nn.ReLU(),
        #                                     nn.Linear(2048, 1024), nn.ReLU())
        # self.sum_combine_fc = nn.Sequential(nn.Linear((critic_obs[0])*n_agents, 2048), nn.ReLU(),
        #                                     nn.Linear(2048, 2048), nn.ReLU())
        # self.sum_combine_fc = nn.Sequential(nn.Linear((critic_obs[0]+critic_obs[1])*n_agents, 256), nn.ReLU())
        # self.sum_combine_actionFC = nn.Sequential(nn.Linear(2*n_agents, 64), nn.ReLU())
        #
        # self.single_own_fc = nn.Sequential(nn.Linear(critic_obs[0], 128), nn.ReLU())  # may be here can be replaced with another attention mechanism
        # self.single_grid_fc = nn.Sequential(nn.Linear(critic_obs[1], 128), nn.ReLU())
        # self.single_surr = nn.Sequential(nn.Linear(critic_obs[2], 128), nn.ReLU())

        # for surrounding agents' encoding, for each agent, we there are n-neighbours, each neighbour is represented by
        # a vector of length = 6. Before we put into an experience replay, we pad it up to max_num_neigh * 6 array.
        # so, one agent will have an array of max_num_neigh * 6, after flatten, then for one batch, there are a total of
        # n_agents exist in the airspace, therefore, the final dimension will be max_num_neigh * 6 * max_num_neigh.
        # self.sum_sur_fc = nn.Sequential(nn.Linear(critic_obs[2]*n_agents*n_agents, 256), nn.ReLU())

        # critic attention for overall sur_neighbours with overall own_state
        # self.single_k = nn.Linear(128, 128, bias=False)
        # self.single_q = nn.Linear(128, 128, bias=False)
        # self.single_v = nn.Linear(128, 128, bias=False)
        #
        # self.n_heads = 3
        # self.single_head_dim = int((256+256+256) / self.n_heads)
        # self.com_k = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        # self.com_q = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        # self.com_v = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        # self.multi_att_out = nn.Sequential(nn.Linear(self.n_heads * self.single_head_dim + n_agents * n_actions, 128),
        #                                    nn.ReLU())
        #
        # self.combine_env_fc = nn.Sequential(nn.Linear(256+256+256, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(),
        #                                     nn.Linear(128, 64), nn.ReLU())
        # self.combine_env_fc = nn.Sequential(nn.Linear(256+128, 128), nn.ReLU())
        # self.combine_env_fc = nn.Sequential(nn.Linear(128+(2*n_agents), 256), nn.ReLU())
        # self.combine_env_fc = nn.Sequential(nn.Linear(1024+(2*n_agents), 512), nn.ReLU())
        self.combine_env_fc = nn.Sequential(nn.Linear(256+(2*n_agents), 256), nn.ReLU(),
                                            nn.Linear(256, 256), nn.ReLU())
        # self.combine_env_fc = nn.Sequential(nn.Linear(1024+(2*n_agents), 512), nn.ReLU(),
        #                                     nn.Linear(512, 512), nn.ReLU(),
        #                                     nn.Linear(512, 512), nn.ReLU())
        # self.combine_env_fc = nn.Sequential(nn.Linear(2048+(2*n_agents), 2048), nn.ReLU(),
        #                                     nn.Linear(2048, 2048), nn.ReLU(),
        #                                     nn.Linear(2048, 2048), nn.ReLU(),
        #                                     nn.Linear(2048, 512), nn.ReLU())
        # self.combine_env_fc = nn.Sequential(nn.Linear(128+(2*n_agents), 512), nn.ReLU())
        # self.combine_env_fc = nn.Sequential(nn.Linear(256+(2*n_agents), 512), nn.ReLU())
        # self.combine_env_fc = nn.Sequential(nn.Linear(128+64, 256), nn.ReLU())
        # self.combine_env_fc = nn.Sequential(nn.Linear(30+4, 256), nn.ReLU())
        # self.combine_env_fc = nn.Sequential(nn.Linear((n_agents*128)+(n_agents*128), 64), nn.ReLU())

        # self.combine_all = nn.Sequential(nn.Linear(128+n_agents * n_actions, 64), nn.ReLU(), nn.Linear(64, 1))
        # self.combine_all = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 1))
        self.combine_all = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))
        # self.combine_all = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))
        # self.combine_all = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 1))

        # self.sum_agents_action_fc = nn.Sequential(nn.Linear(critic_obs[2]*n_agents, 256), nn.ReLU())
        #
        # self.multi_attention = nn.MultiheadAttention(embed_dim=256+256, num_heads=2)  # 1st input is the sum of neurons from actions and combined states encoding
        #
        # # the input of this judgement layer is 256+256 because this is right after the multi-head attention layer
        # # the output dimension of the multi-head attention is default to be the dimension of the "embed_dim"
        # self.judgement_fc = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

        # self.name = name

        # self.optimizer = optim.Adam(self.parameters(), lr=critic_lr)
        # self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        #
        # self.to(self.device)

    def forward(self, state, actor_actions):  # state[0] is sum of all agent's own observed states, state[1] is is sum of all agent's observed grid maps
        # pre-process, compute attention for every agent based on their surrounding agents
        # attention_all_agent = []
        # grid_all_agent = []
        # own_all_agent = []
        # for one_agent_batch_own, one_agent_batch_grid, one_agent_batch_surr in zip(*state):  # automatically loop over 5 times
        #     single_grid_out = self.single_grid_fc(one_agent_batch_grid)
        #     single_own_out = self.single_own_fc(one_agent_batch_own)
        #
        #     # single_surr_out = self.single_surr(one_agent_batch_surr)
        #     # single_q = self.single_q(single_own_out)
        #     # single_k = self.single_k(single_surr_out)
        #     # single_v = self.single_v(single_surr_out)
        #     # mask = one_agent_batch_surr.mean(axis=2, keepdim=True).bool()
        #     # score = torch.bmm(single_k, single_q.unsqueeze(axis=2))
        #     # score_mask = score.clone()  # clone操作很必要
        #     # score_mask[~mask] = float('-inf')  # 不然赋值操作后会无法计算梯度
        #     # alpha = F.softmax(score_mask / np.sqrt(single_k.size(-1)), dim=1)  # we use dim=1 here because we need to get attention of each sequence in K towards all hidden vector of q in each batch.
        #     # alpha_mask = alpha.clone()
        #     # alpha_mask[~mask] = 0
        #     # v_att = torch.sum(single_v * alpha_mask, axis=1)
        #     # attention_all_agent.append(v_att)
        #
        #     grid_all_agent.append(single_grid_out)
        #     own_all_agent.append(single_own_out)


        # sum_att = torch.stack(attention_all_agent).transpose(0, 1)
        # sum_att = sum_att.reshape(sum_att.shape[0], -1)

        # sum_grid = torch.stack(grid_all_agent).transpose(0, 1)
        # sum_grid = sum_grid.reshape(sum_grid.shape[0], -1)
        #
        # sum_own = torch.stack(grid_all_agent).transpose(0, 1)
        # sum_own = sum_own.reshape(sum_own.shape[0], -1)

        # sum_own = self.sum_own_fc(state[0])
        # sum_grid = self.sum_grid_fc(state[1])
        #
        # # env_concat = torch.cat((sum_att, sum_grid), dim=1)
        # env_concat = torch.cat((sum_own, sum_grid), dim=1)
        combine_state = self.sum_combine_fc(state)
        # combine_action = self.sum_combine_actionFC(actor_actions)

        # combine_raw_SA = torch.cat((state, actor_actions), dim=1)
        combine_SA = torch.cat((combine_state, actor_actions), dim=1)
        # combine_SA = torch.cat((combine_state, combine_action), dim=1)

        # env_encode = self.combine_env_fc(env_concat)
        env_encode = self.combine_env_fc(combine_SA)
        # entire_comb = torch.cat((env_encode, actor_actions), dim=1)
        # entire_comb = torch.cat((sum_own_e, actor_actions), dim=1)
        q = self.combine_all(env_encode)
        return q


class CriticNetwork_woGru(nn.Module):
    def __init__(self, critic_obs, n_agents, n_actions, history_horizon_step):
        super(CriticNetwork_woGru, self).__init__()
        self.combine_state_fc = nn.Sequential(nn.Linear((critic_obs[0]) * n_agents, 256), nn.ReLU()) # extract combine state information
        self.combine_hn_fc = nn.Sequential(nn.Linear(history_horizon_step * (critic_obs[0]) * n_agents, 512), nn.ReLU()) # extract history state information
        self.sum_inputs = nn.Sequential(nn.Linear(256+512+(n_actions * n_agents), 512), nn.ReLU(),
                                        nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 1))  # obtain Q value

    def forward(self, state, actor_actions, history_info):
        combine_state = self.combine_state_fc(state)
        combine_hn = self.combine_hn_fc(history_info)
        combine_S_A_hn = torch.cat((combine_state, combine_hn, actor_actions), dim=1)
        q = self.sum_inputs(combine_S_A_hn)
        return q


class CriticNetwork_wGru(nn.Module):  #
    def __init__(self, critic_obs, n_agents, n_actions, combine_history):
        super(CriticNetwork_wGru, self).__init__()
        self.combine_state_fc = nn.Sequential(nn.Linear((critic_obs[0]) * n_agents, 256), nn.ReLU()) # extract combine state information
        # gru layer
        # self.gru = nn.GRU((critic_obs[0]) * n_agents, 256, 1, batch_first=True)
        self.gru = nn.GRU(critic_obs[0], 256, 1, batch_first=True)

        self.sum_inputs = nn.Sequential(nn.Linear(256+256+(n_actions * n_agents), 512), nn.ReLU(),
                                        nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 1))  # obtain Q value

    def forward(self, state, actor_actions, history_info):
        combine_state = self.combine_state_fc(state)
        combine_out, combine_hn = self.gru(history_info)
        combine_S_A_hn = torch.cat((combine_state, combine_hn.squeeze(0), actor_actions), dim=1)
        q = self.sum_inputs(combine_S_A_hn)
        return q


class critic_single_obs_wGRU(nn.Module):
    def __init__(self, critic_obs, n_agents, n_actions, single_history, hidden_state_size):
        super(critic_single_obs_wGRU, self).__init__()
        self.rnn_hidden_dim = hidden_state_size
        self.SA_fc = nn.Sequential(nn.Linear(critic_obs[0]+n_actions, 64), nn.ReLU())
        self.gru_cell = nn.GRUCell(64, hidden_state_size)
        self.own_fc_outlay = nn.Linear(64, 1)

    def forward(self, single_state, single_action, history_hidden_state):
        SA_combine = torch.cat((single_state, single_action), dim=1)
        SA_feature = self.SA_fc(SA_combine)
        h_in = history_hidden_state.reshape(-1, self.rnn_hidden_dim)
        h = self.gru_cell(SA_feature, h_in)
        q = self.own_fc_outlay(h)
        return q, h


class critic_single_TwoPortion(nn.Module):
    def __init__(self, critic_obs, n_agents, n_actions, single_history, hidden_state_size):
        super(critic_single_TwoPortion, self).__init__()
        # --- original, used in 16 Feb MADDPG_test_160224_13_17_59 ---
        self.SA_fc = nn.Sequential(nn.Linear(critic_obs[0]+n_actions, 64), nn.ReLU())
        # self.SA_fc = nn.Sequential(nn.Linear(critic_obs[0]+n_actions, 512), nn.ReLU())
        self.SA_grid = nn.Sequential(nn.Linear(critic_obs[1], 64), nn.ReLU())
        # self.SA_grid = nn.Sequential(nn.Linear(critic_obs[1], 512), nn.ReLU())
        self.merge_fc_grid = nn.Sequential(nn.Linear(64+64, 256), nn.ReLU())
        # self.merge_fc_grid = nn.Sequential(nn.Linear(512+512, 1024), nn.ReLU())
        self.out_feature_q = nn.Sequential(nn.Linear(256, 1))
        # self.out_feature_q = nn.Sequential(nn.Linear(1024, 1))
        # --- end of original --------------------------------------

        # ---------- ignore radar critic ---------------
        # self.owb_obs_fc = nn.Sequential(nn.Linear(critic_obs[0], 128), nn.ReLU())
        # self.owb_obs_hidden = nn.Sequential(nn.Linear(128+2, 256), nn.ReLU(),
        #                                     nn.Linear(256, 256), nn.ReLU())
        # self.out_feature_q = nn.Sequential(nn.Linear(256, 1))

        # self.owb_obs_fc = nn.Sequential(nn.Linear(critic_obs[0]+n_actions, 128), nn.ReLU())  #ignore radar v2
        # self.owb_obs_hidden = nn.Sequential(nn.Linear(128, 256), nn.ReLU(),
        #                                     nn.Linear(256, 256), nn.ReLU())
        # self.out_feature_q = nn.Sequential(nn.Linear(256, 1))
        # ----------- end of ignore radar critic -------------

        # ----- v1 -------
        # self.sa_fc = nn.Sequential(nn.Linear(critic_obs[0], 64), nn.ReLU())
        # self.sa_grid = nn.Sequential(nn.Linear(critic_obs[1], 64), nn.ReLU())
        # self.merge_fc_gridWact = nn.Sequential(nn.Linear(64+64+n_actions, 256), nn.ReLU())
        # self.out_feature_q = nn.Sequential(nn.Linear(256, 1))
        # ------ end of v1 ----

        # ----- v1_1 -------
        # self.sa_fc = nn.Sequential(nn.Linear(critic_obs[0], 64), nn.ReLU())
        # self.sa_grid = nn.Sequential(nn.Linear(critic_obs[1], 64), nn.ReLU())
        # self.act_feature = nn.Sequential(nn.Linear(n_actions, 64), nn.ReLU())
        # self.merge_fc_gridWact = nn.Sequential(nn.Linear(64+64+64, 256), nn.ReLU())
        # self.out_feature_q = nn.Sequential(nn.Linear(256, 1))
        # ------ end of v1_1 ----

    def forward(self, single_state, single_action):
        # --- original, used in 16 Feb MADDPG_test_160224_13_17_59 ---
        obsWaction = torch.cat((single_state[0], single_action), dim=1)
        own_obsWaction = self.SA_fc(obsWaction)
        own_grid = self.SA_grid(single_state[1])
        merge_obs_grid = torch.cat((own_obsWaction, own_grid), dim=1)
        merge_feature = self.merge_fc_grid(merge_obs_grid)
        q = self.out_feature_q(merge_feature)
        # --- end of original --------------------------------------

        # ---------- ignore radar critic ---------------
        # obs_fea = self.owb_obs_fc(single_state[0])
        # obsWaction = torch.cat((obs_fea, single_action), dim=1)
        # obsWaction_fea = self.owb_obs_hidden(obsWaction)
        # q = self.out_feature_q(obsWaction_fea)

        # obsWaction = torch.cat((single_state[0], single_action), dim=1)  # ignore radar v2
        # obs_fea = self.owb_obs_fc(obsWaction)
        # obsWaction_fea = self.owb_obs_hidden(obs_fea)
        # q = self.out_feature_q(obsWaction_fea)
        # ----------- end of ignore radar critic -------------

        # ----- v1 -------
        # own_obs = self.sa_fc(single_state[0])
        # own_grid = self.sa_grid(single_state[1])
        # merge_obs_gridWact = torch.cat((own_obs, own_grid, single_action), dim=1)
        # merge_feature = self.merge_fc_gridWact(merge_obs_gridWact)
        # q = self.out_feature_q(merge_feature)
        # ------ end of v1 ----

        # ----- v1_1 -------
        # own_obs = self.sa_fc(single_state[0])
        # own_grid = self.sa_grid(single_state[1])
        # own_act = self.act_feature(single_action)
        # merge_obs_gridWact = torch.cat((own_obs, own_grid, own_act), dim=1)
        # merge_feature = self.merge_fc_gridWact(merge_obs_gridWact)
        # q = self.out_feature_q(merge_feature)
        # ------ end of v1_1 ----
        return q


class critic_single_TwoPortion_wRadar(nn.Module):
    def __init__(self, critic_obs, n_agents, n_actions, single_history, hidden_state_size):
        super(critic_single_TwoPortion_wRadar, self).__init__()
        self.SA_fc = nn.Sequential(nn.Linear(critic_obs[0]+n_actions, 64), nn.ReLU())
        self.S_all_nei = nn.Sequential(nn.Linear(critic_obs[1], 128), nn.ReLU())
        self.S_radar = nn.Sequential(nn.Linear(critic_obs[2], 128), nn.ReLU())
        self.merge_fc_grid = nn.Sequential(nn.Linear(64+128+128, 512), nn.ReLU())
        # self.merge_fc_grid = nn.Sequential(nn.Linear(64+128, 512), nn.ReLU())
        self.out_feature_q = nn.Sequential(nn.Linear(512, 256), nn.ReLU(),
                                           nn.Linear(256, 1))
        # self.SA_fc = nn.Sequential(nn.Linear(critic_obs[0]+n_actions, 64), nn.LeakyReLU(0.01))
        # self.S_all_nei = nn.Sequential(nn.Linear(critic_obs[1], 128), nn.LeakyReLU(0.01))
        # self.S_radar = nn.Sequential(nn.Linear(critic_obs[2], 128), nn.LeakyReLU(0.01))
        # self.merge_fc_grid = nn.Sequential(nn.Linear(64+128+128, 512), nn.LeakyReLU(0.01))
        # # self.merge_fc_grid = nn.Sequential(nn.Linear(64+128, 512), nn.ReLU())
        # self.out_feature_q = nn.Sequential(nn.Linear(512, 256), nn.LeakyReLU(0.01),
        #                                    nn.Linear(256, 1))

        # ----- to compare
        # self.SA_fc = nn.Sequential(nn.Linear(critic_obs[0]+n_actions, 64), nn.ReLU())
        # self.SA_grid = nn.Sequential(nn.Linear(critic_obs[2], 64), nn.ReLU())
        # # self.merge_fc_grid = nn.Sequential(nn.Linear(64+64, 256), nn.ReLU())
        # self.merge_fc_grid = nn.Sequential(nn.Linear(64, 256), nn.ReLU())
        # self.out_feature_q = nn.Sequential(nn.Linear(256, 1))
        # ---- end of to compare

    def forward(self, single_state, single_action):
        obsWaction = torch.cat((single_state[0], single_action), dim=1)
        own_obsWaction = self.SA_fc(obsWaction)
        own_full_neigh = self.S_all_nei(single_state[1])
        own_radar = self.S_radar(single_state[2])
        merge_obs_grid = torch.cat((own_obsWaction, own_full_neigh, own_radar), dim=1)
        # merge_obs_grid = torch.cat((own_obsWaction, own_radar), dim=1)
        merge_feature = self.merge_fc_grid(merge_obs_grid)
        q = self.out_feature_q(merge_feature)

        # --- to compare
        # obsWaction = torch.cat((single_state[0], single_action), dim=1)
        # own_obsWaction = self.SA_fc(obsWaction)
        # # own_grid = self.SA_grid(single_state[2])
        # # merge_obs_grid = torch.cat((own_obsWaction, own_grid), dim=1)
        # merge_feature = self.merge_fc_grid(own_obsWaction)
        # q = self.out_feature_q(merge_feature)
        # --- end of to compare
        return q


class critic_single_obs_only(nn.Module):
    def __init__(self, critic_obs, n_agents, n_actions, single_history, hidden_state_size):
        super(critic_single_obs_only, self).__init__()
        self.SA_fc = nn.Sequential(nn.Linear(critic_obs[0]+n_actions, 64), nn.ReLU())
        self.out_feature_q = nn.Sequential(nn.Linear(64, 64), nn.ReLU(),
                                           nn.Linear(64, 1))

    def forward(self, single_state, single_action):
        obsWaction = torch.cat((single_state[0], single_action), dim=1)
        obsWaction_fea = self.SA_fc(obsWaction)
        q = self.out_feature_q(obsWaction_fea)
        return q


class critic_single_GRU_TwoPortion(nn.Module):
    def __init__(self, critic_obs, n_agents, n_actions, single_history, hidden_state_size):
        super(critic_single_GRU_TwoPortion, self).__init__()
        self.SA_fc = nn.Sequential(nn.Linear(critic_obs[0]+n_actions, 64), nn.ReLU())
        self.SA_grid = nn.Sequential(nn.Linear(critic_obs[1], 64), nn.ReLU())
        self.rnn_hidden_dim = hidden_state_size
        self.gru_cell = nn.GRUCell(64, hidden_state_size)
        self.own_fc_outlay = nn.Sequential(nn.Linear(64+64, 128), nn.ReLU(),
                                           nn.Linear(128, 1))

    def forward(self, single_state, single_action, history_hidden_state):
        obsWaction = torch.cat((single_state[0], single_action), dim=1)
        own_obsWaction = self.SA_fc(obsWaction)
        own_grid = self.SA_grid(single_state[1])
        h_in = history_hidden_state.reshape(-1, self.rnn_hidden_dim)
        # gru apply to 2nd portion
        # h_out = self.gru_cell(own_grid, h_in)
        # merge_obs_Hgrid = torch.cat((own_obsWaction, h_out), dim=1)
        # gru apply to 1st portion
        h_out = self.gru_cell(own_obsWaction, h_in)
        merge_obs_Hgrid = torch.cat((own_grid, h_out), dim=1)
        q = self.own_fc_outlay(merge_obs_Hgrid)

        return q, h_out


class critic_single_OnePortion(nn.Module):
    def __init__(self, critic_obs, n_agents, n_actions, single_history, hidden_state_size):
        super(critic_single_OnePortion, self).__init__()
        self.SA_fcWgrid = nn.Sequential(nn.Linear(critic_obs[0]+n_actions+critic_obs[1], 64), nn.ReLU())
        self.merge_fc_grid = nn.Sequential(nn.Linear(64, 256), nn.ReLU())
        self.out_feature_q = nn.Sequential(nn.Linear(256, 1))

    def forward(self, single_state, single_action):
        obsWactionWgrid = torch.cat((single_state[0], single_action, single_state[1]), dim=1)
        own_obsWactionWgrid = self.SA_fcWgrid(obsWactionWgrid)
        merge_feature = self.merge_fc_grid(own_obsWactionWgrid)
        q = self.out_feature_q(merge_feature)
        return q


class critic_combine_TwoPortion(nn.Module):
    def __init__(self, critic_obs, n_agents, n_actions, single_history, hidden_state_size):
        super(critic_combine_TwoPortion, self).__init__()
        # v1 #
        # # self.SA_fc = nn.Sequential(nn.Linear(critic_obs[0]+(n_actions*n_agents), 128), nn.ReLU())
        # self.SA_fc = nn.Sequential(nn.Linear(critic_obs[0]+(n_actions*n_agents), 256), nn.ReLU())
        # # self.SA_grid = nn.Sequential(nn.Linear(critic_obs[1], 128), nn.ReLU())
        # self.SA_grid = nn.Sequential(nn.Linear(critic_obs[1], 256), nn.ReLU())
        # # self.merge_fc_grid = nn.Sequential(nn.Linear(128+128, 256), nn.ReLU())
        # self.merge_fc_grid = nn.Sequential(nn.Linear(256+256, 256), nn.ReLU())
        # self.out_feature_q = nn.Sequential(nn.Linear(256, 1))
        # end of v1 #

        #  # v2 #
        # self.S_fc = nn.Sequential(nn.Linear(critic_obs[0], 128), nn.ReLU())
        # self.grid_fc = nn.Sequential(nn.Linear(critic_obs[1], 128), nn.ReLU())
        # self.combine_inputWact = nn.Sequential(nn.Linear(128+128+(n_actions*n_agents), 256), nn.ReLU())
        # self.out_feature_q = nn.Sequential(nn.Linear(256, 1))
        # # end of v2 #

        # v1_yc #
        self.o1a1 = nn.Sequential(nn.Linear(critic_obs[0]+critic_obs[1]+n_actions, 128), nn.ReLU())
        self.o2a2 = nn.Sequential(nn.Linear(critic_obs[0]+critic_obs[1]+n_actions, 128), nn.ReLU())
        self.o3a3 = nn.Sequential(nn.Linear(critic_obs[0]+critic_obs[1]+n_actions, 128), nn.ReLU())
        self.combine_agents_fea = nn.Sequential(nn.Linear(128+128+128, 256), nn.ReLU())
        self.out_feature_q = nn.Sequential(nn.Linear(256, 1))
        # end of v1_yc #

    def forward(self, combine_state, combine_action):
        # ---- v1 -----
        # obsWaction = torch.cat((combine_state[0], combine_action), dim=1)  # obs + action
        # own_obsWaction = self.SA_fc(obsWaction)
        # own_grid = self.SA_grid(combine_state[1])  # grid
        # merge_obs_grid = torch.cat((own_obsWaction, own_grid), dim=1)
        # merge_feature = self.merge_fc_grid(merge_obs_grid)
        # q = self.out_feature_q(merge_feature)
        # --- end of v1 ---

        # --- v2 ---
        # own_obs = self.S_fc(combine_state[0])
        # own_grid = self.grid_fc(combine_state[1])
        # combine_obs = torch.cat((own_obs, own_grid, combine_action), dim=1)
        # combine_obs_feature = self.combine_inputWact(combine_obs)
        # q = self.out_feature_q(combine_obs_feature)
        # --- end of v2 ---

        # ---- yc_v1 -----
        for agent_idx in range(3):
            agent_obs = torch.cat((combine_state[0][:, agent_idx,:], combine_state[1][:, agent_idx,:]), dim=1)
            if isinstance(combine_action, list):
                agent_act = combine_action[agent_idx]
            else:
                agent_act = combine_action[:, agent_idx, :]
            obsWact = torch.cat((agent_obs, agent_act), dim=1)  # obs + action
            if agent_idx == 0:
                o1a1_fea = self.o1a1(obsWact)
            elif agent_idx == 1:
                o2a2_fea = self.o2a2(obsWact)
            elif agent_idx == 2:
                o3a3_fea = self.o3a3(obsWact)

        merge_all_agent = torch.cat((o1a1_fea, o2a2_fea, o3a3_fea), dim=1)
        merge_feature = self.combine_agents_fea(merge_all_agent)
        q = self.out_feature_q(merge_feature)
        # --- end of yc_v1 ---
        return q


class critic_combine_TwoPortion_fullneiWradar(nn.Module):
    def __init__(self, critic_obs, n_agents, n_actions, single_history, hidden_state_size, normalizer, my_device):
        super(critic_combine_TwoPortion_fullneiWradar, self).__init__()
        self.normlize_tool = normalizer
        self.device = my_device
        # Create a list to hold individual agent networks
        self.agents_obs = nn.ModuleList()
        self.agents_all_neigh = nn.ModuleList()
        self.agents_grids = nn.ModuleList()
        self.combined_agent_feature = nn.ModuleList()
        # Define similar network structures for each agent
        for _ in range(n_agents):
            agent_obs = nn.Sequential(
                nn.Linear(critic_obs[0] + n_actions, 64),
                nn.ReLU())
            agent_all_nei = nn.Sequential(
                nn.Linear(critic_obs[1], 64),
                nn.ReLU())
            agent_grids = nn.Sequential(
                nn.Linear(critic_obs[2], 64),
                nn.ReLU())
            combine_agent_fea = nn.Sequential(nn.Linear(64 + 64 + 64, 256), nn.ReLU())
            self.agents_obs.append(agent_obs)
            self.agents_all_neigh.append(agent_all_nei)
            self.agents_grids.append(agent_grids)
            self.combined_agent_feature.append(combine_agent_fea)
        self.all_agent_combined_features_q = nn.Sequential(nn.Linear(n_agents*256, 1024), nn.ReLU(),
                                                         nn.Linear(1024, 512), nn.ReLU(),
                                                         nn.Linear(512, 1))
        # TOO large !!!!, N * 512 =.=
        # V2 simplified critic
        # self.agents_obs = nn.ModuleList()
        # self.combined_agent_feature = nn.ModuleList()
        # # Define similar network structures for each agent
        # for _ in range(n_agents):
        #     agent_obs = nn.Sequential(
        #         nn.Linear(critic_obs[0] + n_actions, 64),
        #         nn.ReLU())
        #     combine_agent_fea = nn.Sequential(nn.Linear(64, 128), nn.ReLU())
        #     self.agents_obs.append(agent_obs)
        #     self.combined_agent_feature.append(combine_agent_fea)
        # self.all_agent_combined_features_q = nn.Sequential(nn.Linear(n_agents*128, 2048), nn.ReLU(),
        #                                                  nn.Linear(2048, 512), nn.ReLU(),
        #                                                  nn.Linear(512, 1))
        # V3 combine all critic obs directly then we use mlp to process these information
        # self.all_agent_combined_features_q = nn.Sequential(nn.Linear(n_agents * (critic_obs[0] + n_actions), 512), nn.ReLU(),
        #                                                    nn.Linear(512, 512), nn.ReLU(),
        #                                                    nn.Linear(512, 256), nn.ReLU(),
        #                                                    nn.Linear(256, 1))
        # V3.1 (calculate next step drone position)
        # self.all_agent_combined_features_q = nn.Sequential(nn.Linear(n_agents * (critic_obs[0] + 2 + n_actions), 512), nn.ReLU(),
        #                                                    nn.Linear(512, 512), nn.ReLU(),
        #                                                    nn.Linear(512, 256), nn.ReLU(),
        #                                                    nn.Linear(256, 1))

    def forward(self, combine_state, combine_action):
        all_agent_merge_feat = []
        for i in range(len(self.agents_obs)):
            one_agent_obsWaction = self.agents_obs[i](torch.concatenate((combine_state[0][:, i, :], combine_action[:, i, :]), dim=1))
            one_agent_all_nei = self.agents_all_neigh[i](combine_state[1][:, i, :])
            one_agent_grid = self.agents_grids[i](combine_state[2][:, i, :])
            merge_fea = torch.concatenate((one_agent_obsWaction, one_agent_all_nei, one_agent_grid), dim=1)
            one_agent_combine_feat = self.combined_agent_feature[i](merge_fea)
            all_agent_merge_feat.append(one_agent_combine_feat)
        concated_all_agent_feat = torch.concatenate(all_agent_merge_feat,dim=1)
        q = self.all_agent_combined_features_q(concated_all_agent_feat)
        # V2
        # all_agent_merge_feat = []
        # for i in range(len(self.agents_obs)):
        #     one_agent_obsWaction = self.agents_obs[i](torch.concatenate((combine_state[0][:, i, :], combine_action[:, i, :]), dim=1))
        #     one_agent_combine_feat = self.combined_agent_feature[i](one_agent_obsWaction)
        #     all_agent_merge_feat.append(one_agent_combine_feat)
        # concated_all_agent_feat = torch.concatenate(all_agent_merge_feat,dim=1)
        # q = self.all_agent_combined_features_q(concated_all_agent_feat)
        # V3
        # all_agent_merge_feat = []
        # actual_prob_coord = self.normalizer.reverse_nmlz_pos(norm_intersection_obstacle)
        # for i in range(combine_action.shape[1]):
        #     one_agent_obsWaction = torch.concatenate((combine_state[0][:, i, :], combine_action[:, i, :]), dim=1)
        #     all_agent_merge_feat.append(one_agent_obsWaction)
        # concated_all_agent_feat = torch.concatenate(all_agent_merge_feat,dim=1)
        # q = self.all_agent_combined_features_q(concated_all_agent_feat)
        # V3.1 (calculate next step drone position)
        # combine state [px, py, vx, vy, ... ]
        # all_agent_merge_feat = []
        # for i in range(combine_action.shape[1]):
        #     batch_newPos = []
        #     for row_idx, row in enumerate(combine_state[0][:, i, :]):
        #         actual_pos = self.normlize_tool.reverse_nmlz_pos(combine_state[0][:, i, :][row_idx, :][0:2].cpu())
        #         actual_vel = self.normlize_tool.reverse_nmlz_vel(combine_state[0][:, i, :][row_idx, :][2:4].cpu())
        #         ax = combine_action[:, 0, :][row_idx, 0] * self.normlize_tool.acc_max
        #         ay = combine_action[:, 0, :][row_idx, 1] * self.normlize_tool.acc_max
        #         curVelx = actual_vel[0] + ax.cpu() * 0.5
        #         curVely = actual_vel[1] + ay.cpu() * 0.5
        #         next_heading = math.atan2(curVely, curVelx)
        #         if np.linalg.norm([curVelx.clone().detach().numpy(), curVely.clone().detach().numpy()]) >= self.normlize_tool.spd_max:
        #             # update host velocity when chosen speed has exceeded the max speed
        #             hvx = self.normlize_tool.spd_max * math.cos(next_heading)
        #             hvy = self.normlize_tool.spd_max * math.sin(next_heading)
        #             new_vel = np.array([hvx, hvy])
        #         else:
        #             # update host velocity when max speed is not exceeded
        #             new_vel = np.array([curVelx.clone().detach().numpy(), curVely.clone().detach().numpy()])
        #         # update the drone's position based on the update velocities
        #         delta_x = new_vel[0] * 0.5
        #         delta_y = new_vel[1] * 0.5
        #         new_pos = np.array([actual_pos[0] + delta_x, actual_pos[1] + delta_y])
        #         norm_new_pos = self.normlize_tool.nmlz_pos(new_pos)
        #         batch_newPos.append(norm_new_pos)
        #     tensor_batch_newPos = torch.from_numpy(np.array(batch_newPos)).to(self.device)
        #
        #     new_state = torch.concatenate((combine_state[0][:, i, :], tensor_batch_newPos), dim=1)
        #     one_agent_obsWaction = torch.concatenate((new_state, combine_action[:, i, :]), dim=1)
        #     all_agent_merge_feat.append(one_agent_obsWaction)
        # concated_all_agent_feat = torch.concatenate(all_agent_merge_feat, dim=1)
        # q = self.all_agent_combined_features_q(concated_all_agent_feat)
        return q
