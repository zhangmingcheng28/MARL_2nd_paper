# -*- coding: utf-8 -*-
"""
@Time    : 11/16/2023 9:42 AM
@Author  : Mingcheng
@FileName: 
@Description: 
@Package dependency:
"""
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np


def get_custom_linear_scaling_factor(episode, eps_end, start_scale=1, end_scale=0.03):
    # # Ensure episode is within the range
    # episode = min(max(1, episode), total_episodes)
    # Calculate the slope of the linear decrease only up to eps_end
    if episode <= eps_end:
        slope = (end_scale - start_scale) / (eps_end-1)
        current_scale = start_scale + slope * (episode-1)
    else:
        current_scale = end_scale

    return current_scale

matplotlib.use('TkAgg')
# pre_fix = r'D:\MADDPG_2nd_jp\161123_20_55_14\toplot'
# file_path = pre_fix + r'\all_episode_noise.pickle'
# with open(file_path, 'rb') as handle:
#     combine_noise_1 = pickle.load(handle)
#
# pre_fix = r'D:\MADDPG_2nd_jp\161123_10_19_52\toplot'
# file_path = pre_fix + r'\all_episode_noise.pickle'
# with open(file_path, 'rb') as handle:
#     combine_noise = pickle.load(handle)
#
# # x-acc noise
# x1_acc_noise = []
# x_acc_noise = []
# y1_acc_noise = []
# for each_eps in combine_noise_1:
#     for each_step in each_eps:
#         x1_acc_noise.append(each_step[0])
#         y1_acc_noise.append(each_step[1])
#
# plot1 = plt.plot(x1_acc_noise, label='Initial_var_0.5')
# plot2 = plt.plot(y_acc_noise, label='y_noise')
#
# x_acc_noise = []
# y_acc_noise = []
# for each_eps in combine_noise:
#     for each_step in each_eps:
#         x_acc_noise.append(each_step[0])
#         y_acc_noise.append(each_step[1])
#
# plot2 = plt.plot(x_acc_noise, label='Initial_var_1')

# ---------- test noise---------------
x_acc_noise = []
Total_episode = 50000
eps_end = 5000
var_ = 1
var_2 = 0.5
for each_eps in range(Total_episode):
    # if each_eps > 0:
    input_var = get_custom_linear_scaling_factor(each_eps, eps_end, var_)
    # input_var_2 = get_custom_linear_scaling_factor(each_eps, eps_end, var_2)
    noise_value = np.random.randn(2) * input_var
    # noise_value_2 = np.random.randn(2) * input_var_2
    x_acc_noise.append(noise_value[0])
    # x1_acc_noise.append(noise_value_2[0])

# plot1 = plt.plot(x_acc_noise, label='my_var_ver')
# plot2 = plt.plot(x1_acc_noise, label='Initial_var_0.5')
# ---------- end of test noise---------------

# ------------- get noise end episode/step -------------
ori_var_ = 1
count = 0
x_acc_noise_ori = []
for each_eps in range(Total_episode*50):
    noise_value_ori = np.random.randn(2) * ori_var_
    if ori_var_ > 0.05:  # noise decrease at every step instead of every episode.
        ori_var_ = ori_var_ * 0.999998
    else:
        ori_var_ = 0.05
        # eps = count / 50
    x_acc_noise_ori.append(noise_value_ori[0])
plot2 = plt.plot(x_acc_noise_ori, label='ori_var_ver')

plt.grid(linestyle='-.')
plt.xlabel('steps taken')
plt.ylabel('noise level')
plt.legend()
plt.show()
print('end')