# -*- coding: utf-8 -*-
"""
@Time    : 11/27/2023 2:29 PM
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

matplotlib.use('TkAgg')
pre_fix = r'D:\MADDPG_2nd_jp\281123_14_05_09\toplot'
file_path = pre_fix + r'\all_episode_time.pickle'
with open(file_path, 'rb') as handle:
    combine_time = pickle.load(handle)

plt.figure(1)  # each episode reset time consumed in milliseconds
reset_milliseconds = []
for ea_eps_reset in combine_time:
    reset_milliseconds.append(ea_eps_reset[0])
plot1 = plt.plot(reset_milliseconds, label='episode reset time')
plt.title('Plot of reset time vs episode number')
plt.xlabel('number of episodes')
plt.ylabel('reset time used in each episode (milliseconds)')


plt.figure(2)  # each episode elapsed time
episode_time = [eps_time_record[1] for eps_time_record in combine_time]
episode_step_num = [len(eps_time_record[2]) for eps_time_record in combine_time]
# Create the plot
plt.scatter(episode_time, episode_step_num, color='blue')
# Title and labels
plt.title('Plot of episode_time vs episode_step_num')
plt.xlabel('episode time (seconds)')
plt.ylabel('episode step number')

plt.figure(3)  # mean running time.
plt.plot(np.array(episode_time)/np.array(episode_step_num), color='blue')
plt.title('Plot of mean time take for each step in episode')
plt.xlabel('episode time (seconds)')
plt.ylabel('episode number')

fig, axs = plt.subplots(2, 5)
# 3rd list holder contains generate_action_time, step_transition_time, reward_generation_time, update_time_used, whole_step_time

generate_action_time = []
step_transition_time = []
reward_generation_time = []
update_time_used = []
whole_step_time = []

generate_action_time_vs_eps = []
step_transition_time_vs_eps = []
reward_generation_time_vs_eps = []
update_time_used_vs_eps = []
whole_step_time_vs_eps = []

for eps_idx, eps_time_record in enumerate(combine_time):
    for ea_step_time in eps_time_record[2]:
        generate_action_time.append(ea_step_time[0])
        step_transition_time.append(ea_step_time[1])
        reward_generation_time.append(ea_step_time[2])
        update_time_used.append(ea_step_time[3])
        whole_step_time.append(ea_step_time[4])

        generate_action_time_vs_eps.append([eps_idx, ea_step_time[0]])
        step_transition_time_vs_eps.append([eps_idx, ea_step_time[1]])
        reward_generation_time_vs_eps.append([eps_idx, ea_step_time[2]])
        update_time_used_vs_eps.append([eps_idx, ea_step_time[3]])
        whole_step_time_vs_eps.append([eps_idx, ea_step_time[4]])

plot_holder = [generate_action_time, step_transition_time, reward_generation_time, update_time_used, whole_step_time,
               np.array(generate_action_time_vs_eps), np.array(step_transition_time_vs_eps), np.array(reward_generation_time_vs_eps),
               np.array(update_time_used_vs_eps), np.array(whole_step_time_vs_eps)]
plot_holder_title = ["action generate time per step", "step transition time", "reward generation time", "step update time", "time taken for entire step",
                     "action generation time vs episode number", "step transition time vs episode number", "reward generation time vs episode number",
                     "step update time vs episode number", "whole step time vs episode number"]
for i, ax in enumerate(axs.flatten(), start=1):
    if isinstance(plot_holder[i-1], list):
        ax.plot(plot_holder[i-1])
        ax.set_title(plot_holder_title[i-1])
        ax.set_ylabel('milliseconds')
        ax.set_xlabel('step number')
    else:
        ax.scatter(plot_holder[i-1][:, 0], plot_holder[i-1][:, 1], color='blue')
        # ax.set_title(plot_holder_title[i-1])
        ax.set_ylabel('milliseconds')
        ax.set_xlabel('episode number')




plt.show()
