"""
@Time    : 13/8/2024 1:28 PM
@Author  : Bizhao Pang
@FileName:
@Description: plot all results for the special issue paper
@___Parameters___ to tune the training curve's shape and noise variance:
    1) window_size
    2) step
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Specify the file path
file_path = r'D:\MADDPG_2nd_jp\160824_18_24_23\GFG.csv'

# Read the data from the CSV file, no header since the file contains only reward values
data = pd.read_csv(file_path, header=None)

# Convert the DataFrame to a numpy array for easier manipulation
rewards = data.values  # No need to transpose if shape is (4, 16384)

# Smoothing function (moving average)
def moving_average(data, window_size=100):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Apply smoothing
smoothed_rewards = np.array([moving_average(reward, window_size=500) for reward in rewards])

# Calculate mean and confidence intervals for smoothed data
mean_rewards = smoothed_rewards.mean(axis=1)
std_rewards = smoothed_rewards.std(axis=1)

# X-axis values (episodes)
episodes = np.arange(1, len(smoothed_rewards[0]) + 1, step=200)  # Plot every 10th episode
smoothed_rewards = smoothed_rewards[:, ::200]
# Plotting
plt.figure(figsize=(8, 6))

# Model 1
plt.plot(episodes, smoothed_rewards[0], 'r-', label='Model 1')
plt.fill_between(episodes, smoothed_rewards[0] - std_rewards[0], smoothed_rewards[0] + std_rewards[0], color='red', alpha=0.2)

# Model 2
plt.plot(episodes, smoothed_rewards[1], 'navy', label='Model 2')
plt.fill_between(episodes, smoothed_rewards[1] - std_rewards[1], smoothed_rewards[1] + std_rewards[1], color='navy', alpha=0.2)

# Model 3
plt.plot(episodes, smoothed_rewards[2], 'purple', label='Model 3')
plt.fill_between(episodes, smoothed_rewards[2] - std_rewards[2], smoothed_rewards[2] + std_rewards[2], color='purple', alpha=0.2)

# Model 4
plt.plot(episodes, smoothed_rewards[3], 'brown', label='Model 4')
plt.fill_between(episodes, smoothed_rewards[3] - std_rewards[3], smoothed_rewards[3] + std_rewards[3], color='brown', alpha=0.2)

# Adding labels and title
plt.xlabel('Training episode')
plt.ylabel('Reward')
plt.title('Training Curve with Confidence Interval')
plt.legend()

# Show plot
plt.show()
