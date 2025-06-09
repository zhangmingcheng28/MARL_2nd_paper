# -*- coding: utf-8 -*-
"""
@Time    : 11/27/2023 5:08 PM
@Author  : Mingcheng
@FileName: 
@Description: 
@Package dependency:
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import matplotlib

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
matplotlib.use('TkAgg')
# Set up the figure, the axis, and the plot element to animate
fig, ax = plt.subplots()

x = np.linspace(0, 2*np.pi, 100)
line, = ax.plot(x, np.sin(x))

# Initialization function: plot the background of each frame
def init():
    line.set_ydata([np.nan] * len(x))
    return line,

# Animation function: this is called sequentially
def animate(i):
    line.set_ydata(np.sin(x + i / 10.0))  # update the data
    return line,

# Call the animator
ani = animation.FuncAnimation(fig, animate, init_func=init, frames=200, interval=20, blit=True)

# Save the animation as a GIF file
ani.save('F:\githubClone\Multi_agent_AAC\MADDPG_ownENV_randomOD_tidy/sine_wave_animation.gif', writer='imagemagick')

plt.close(fig)  # Closing the figure so it won't display in the output

# '/mnt/data/sine_wave_animation.gif'  # Return the path to the saved GIF file
