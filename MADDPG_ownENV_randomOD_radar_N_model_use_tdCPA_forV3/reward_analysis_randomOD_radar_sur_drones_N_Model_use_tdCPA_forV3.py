# -*- coding: utf-8 -*-
"""
@Time    : 7/26/2023 9:51 AM
@Author  : Thu Ra
@FileName: 
@Description: 
@Package dependency:
"""
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use('TkAgg')
# pre_fix = r'D:\MADDPG_2nd_jp\141123_10_19_22\toplot'
# file_path = pre_fix + r'\all_episode_reward.pickle'
with open('F:\githubClone\Multi_agent_AAC\MADDPG_ownENV_randomOD_radar_sur_drones\episode_critic_loss_cal_record_1500.pickle', 'rb') as handle:
    combine_reward = pickle.load(handle)

# seperated_reward = [[] for _ in combine_reward[0][0][0]]
# # First_term = []
# # Second_term = []
# # Third_term = []
# sum_reward_last_agent = []
# for per_ep_r in combine_reward:
#     for per_step_r in per_ep_r:
#         for per_agent_r in per_step_r:
#             if per_agent_r is None:
#                 # print("None is found in a step of episode, skip it")
#                 # we are only saving the step reward, other term like goal reaching or crashing is not recorded here
#                 continue
#             for individual_list_idx in range(len(seperated_reward)):
#                 seperated_reward[individual_list_idx].append(per_agent_r[individual_list_idx])
#             sum_reward_last_agent.append(sum(per_agent_r))
#             # seperated_reward[0].append()
#             # seperated_reward[0].append()
#             # seperated_reward[0].append()
#             # seperated_reward[0].append()
#             # CT_term.append(per_agent_r[0])
#             # g_term.append(per_agent_r[1])
#             # alive_term.append(per_agent_r[2])

# ---- start preprocess for calculate critic loss ---------------
compile_all_tar_Q_before_rew = []
compile_all_reward_cal = []
compile_all_tar_Q_after_rew = []
compile_all_cal_loss_Q = []

compile_all_tar_Q_before_rew_max = []
compile_all_tar_Q_before_rew_min = []
compile_all_reward_cal_max = []
compile_all_reward_cal_min = []
compile_all_cal_loss_Q_max = []
compile_all_cal_loss_Q_min = []
compile_all_tar_Q_after_rew_max = []
compile_all_tar_Q_after_rew_min = []

for each_eps_critic_cal in combine_reward:
    if len(each_eps_critic_cal) == 0:
        continue
    for ea_step_critic_cal in each_eps_critic_cal:

        tar_Q_before_rew = ea_step_critic_cal[0]
        compile_all_tar_Q_before_rew.append(tar_Q_before_rew)
        compile_all_tar_Q_before_rew_max.append(ea_step_critic_cal[4][0])
        compile_all_tar_Q_before_rew_min.append(ea_step_critic_cal[4][1])

        reward_cal = ea_step_critic_cal[1]
        compile_all_reward_cal.append(reward_cal)
        compile_all_reward_cal_max.append(ea_step_critic_cal[5][1])
        compile_all_reward_cal_min.append(ea_step_critic_cal[5][0])

        tar_Q_after_rew = ea_step_critic_cal[2]
        compile_all_tar_Q_after_rew.append(tar_Q_after_rew)
        compile_all_tar_Q_after_rew_max.append(ea_step_critic_cal[6][1])
        # if ea_step_critic_cal[6][0] != tar_Q_after_rew.min():
        #     print("check")
        # compile_all_tar_Q_after_rew_max.append(tar_Q_after_rew.max())
        compile_all_tar_Q_after_rew_min.append(ea_step_critic_cal[6][0])
        # compile_all_tar_Q_after_rew_min.append(tar_Q_after_rew.min())

        cal_loss_Q = ea_step_critic_cal[3]
        compile_all_cal_loss_Q.append(cal_loss_Q)
        compile_all_cal_loss_Q_max.append(ea_step_critic_cal[6][1])
        compile_all_cal_loss_Q_min.append(ea_step_critic_cal[6][0])




# ----- end of preprocess for calculate critic loss ----------------

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
fig, ax = plt.subplots(1, 1)

# n_bins = 10
# plot1 = plt.hist(seperated_reward[-1], bins=n_bins)
# plo1 = plt.boxplot(seperated_reward[-1])

# compile_all_tar_Q_before_rew = np.concatenate(compile_all_tar_Q_before_rew)
# histogram_values, bin_edges = np.histogram(compile_all_tar_Q_before_rew, bins=100)  # You can specify the number of bins or use 'auto' for automatic binning.
# plt.hist(compile_all_tar_Q_before_rew, bins=100)


# plot1 = plt.plot(sum_reward_last_agent, linestyle='-', label='overall')

# plot3 = plt.plot(seperated_reward[0], linestyle='-.', label='alive-term')

# plot1 = plt.plot(compile_all_reward_cal_max, linestyle='-', label='all_reward_cal_max')
# plot2 = plt.plot(compile_all_reward_cal_min, linestyle='--', label='all_reward_cal_min')

# plot1 = plt.plot(compile_all_tar_Q_before_rew_max, linestyle='-', color='#ff7f0e', label='tar_Q_bf_rw_max')
# plot1 = plt.plot(compile_all_tar_Q_before_rew_max, linestyle='-', label='tar_Q_bf_rw_max')
# plot2 = plt.plot(compile_all_tar_Q_before_rew_min, linestyle='-', color='#1f77b4', label='tar_Q_bf_rw_min')
# plot2 = plt.plot(compile_all_tar_Q_before_rew_min, linestyle='-', label='tar_Q_bf_rw_min')

# plot1 = plt.plot(compile_all_tar_Q_after_rew_max, linestyle='-', color='#1f77b4', label='tar_Q_af_rw_max')
# plot2 = plt.plot(compile_all_tar_Q_after_rew_min, linestyle='-', color='#ff7f0e', label='tar_Q_af_rw_min')

data_to_show = np.concatenate(compile_all_tar_Q_after_rew)
plt.hist(data_to_show, bins=50)


plt.grid()
# plt.xlabel("Occurrence count")
# plt.ylabel("reward")
# plt.ylabel("tar_Q_before_reward")
# plt.ylabel("tar_Q_after_reward")
# plt.xlabel("target_Q_val_before_reward")
plt.xlabel("target_Q_val_after_reward")
# plt.legend()
plt.show()
print('done')