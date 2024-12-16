# -*- coding: utf-8 -*-
"""
@Time    : 8/17/2024 7:39 PM
@Author  : Thu Ra
@FileName: 
@Description: 
@Package dependency:
"""
# Find the contours manually without plotting
# from skimage import measure
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.path as mpath
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import PathPatch
from scipy.ndimage import gaussian_filter
import os
import matplotlib

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
matplotlib.use('TkAgg')
fig, ax = plt.subplots()
center_x, center_y = 0, 0
num_points_per_cluster = 5000
num_clusters = 15
x_range = (-10, 10)
y_range = (-10, 10)

cluster_centers_x = np.random.uniform(center_x + x_range[0], center_x + x_range[1], num_clusters)
cluster_centers_y = np.random.uniform(center_y + y_range[0], center_y + y_range[1], num_clusters)
cluster_centers = np.column_stack((cluster_centers_x, cluster_centers_y))

# Generate points for each cluster with controlled density
x, y = [], []
for cx, cy in cluster_centers:
    angles = np.random.uniform(0, 2 * np.pi, num_points_per_cluster)
    radii = np.random.normal(0, 0.1, num_points_per_cluster)  # Decrease spread for higher density
    x.extend(cx + radii * np.cos(angles))
    y.extend(cy + radii * np.sin(angles))
x = np.array(x)
y = np.array(y)
# Create a 2D histogram to serve as the contour data
margin = 25
contour_min_x = center_x + x_range[0] - margin
contour_max_x = center_x + x_range[1] + margin
contour_min_y = center_y + y_range[0] - margin
contour_max_y = center_y + y_range[1] + margin
hist, xedges, yedges = np.histogram2d(x, y, bins=(100, 100),
                                      range=[[contour_min_x, contour_max_x], [contour_min_y, contour_max_y]])
# Smooth the histogram to create a more organic shape
hist = gaussian_filter(hist, sigma=5)  # Adjust sigma for better control

# Create the custom colormap from green to yellow to red
cmap = LinearSegmentedColormap.from_list('green_yellow_red', ['green', 'yellow', 'red'])
X, Y = np.meshgrid(xedges[:-1] + 0.5 * (xedges[1] - xedges[0]), yedges[:-1] + 0.5 * (yedges[1] - yedges[0]))
contour_levels = np.linspace(hist.min(), hist.max(), 10)
contour = ax.contourf(X, Y, hist, levels=contour_levels, cmap=cmap)
level_color = cmap((contour_levels[1] - contour_levels.min()) / (contour_levels.max() - contour_levels.min()))
# Extract the outermost contour path and overlay it with a black line
outermost_contour = ax.contour(X, Y, hist, levels=[contour_levels[1]], colors=[level_color], linewidths=1)  # this line must be present to

# Extract the vertices of the outermost contour path
outermost_path = outermost_contour.collections[0].get_paths()[0]
vertices = outermost_path.vertices
x_clip, y_clip = vertices[:, 0], vertices[:, 1]
# ax.plot(x_clip, y_clip, color="crimson")
coordinates = np.column_stack((x_clip, y_clip))
clippath = Path(coordinates)
patch = PathPatch(clippath, facecolor='none')
ax.add_patch(patch)
for c in contour.collections:
    c.set_clip_path(patch)

# Add labels and title
ax.set_title('Contour Plot with Correct Outermost Circumference')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.legend()

# Show the plot
plt.show()