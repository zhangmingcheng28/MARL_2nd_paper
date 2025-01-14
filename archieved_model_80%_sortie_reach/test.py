# -*- coding: utf-8 -*-
"""
@Time    : 1/15/2024 2:51 PM
@Author  : Thu Ra
@FileName: 
@Description: 
@Package dependency:
"""
# import numpy as np
# import heapq
# import os
# import matplotlib.pyplot as plt
# import matplotlib
# from matplotlib.pyplot import figure

##############################################################################

# plot grid
# https://www.analytics-link.com/post/2018/09/14/applying-the-a-path-finding-algorithm-in-python-part-1-2d-square-grid
##############################################################################


# grid = np.array([
#
#     [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#
#     [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#
#     [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#
#     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#
#     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#
#     [0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#
#     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1],
#
#     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
#
#     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
#
#     [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
#
#     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#
#     [1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
#
#     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
#
#     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#
#     [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0],
#
#     [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#
#     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#
#     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
#
# # start point and goal
#
# start = (0, 0)
#
# goal = (0, 19)
#
#
# ##############################################################################
#
# # heuristic function for path scoring
#
# ##############################################################################
#
#
# def heuristic(a, b):
#     return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)
#
#
# ##############################################################################
#
# # path finding function
#
# ##############################################################################
#
#
# def astar(array, start, goal):
#     neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
#
#     close_set = set()
#
#     came_from = {}
#
#     gscore = {start: 0}
#
#     fscore = {start: heuristic(start, goal)}
#
#     oheap = []
#
#     heapq.heappush(oheap, (fscore[start], start))
#
#     while oheap:
#
#         current = heapq.heappop(oheap)[1]
#
#         if current == goal:
#
#             data = []
#
#             while current in came_from:
#                 data.append(current)
#
#                 current = came_from[current]
#
#             return data
#
#         close_set.add(current)
#
#         for i, j in neighbors:
#
#             neighbor = current[0] + i, current[1] + j
#
#             tentative_g_score = gscore[current] + heuristic(current, neighbor)
#
#             if 0 <= neighbor[0] < array.shape[0]:
#
#                 if 0 <= neighbor[1] < array.shape[1]:
#
#                     if array[neighbor[0]][neighbor[1]] == 1:
#                         continue
#
#                 else:
#
#                     # array bound y walls
#
#                     continue
#
#             else:
#
#                 # array bound x walls
#
#                 continue
#
#             if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
#                 continue
#
#             if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1] for i in oheap]:
#                 came_from[neighbor] = current
#
#                 gscore[neighbor] = tentative_g_score
#
#                 fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
#
#                 heapq.heappush(oheap, (fscore[neighbor], neighbor))
#
#     return False
#
#
# route = astar(grid, start, goal)
#
# route = route + [start]
#
# route = route[::-1]
#
# print(route)
#
# ##############################################################################
#
# # plot the path
#
# ##############################################################################
#
#
# # extract x and y coordinates from route list
#
# x_coords = []
#
# y_coords = []
#
# for i in (range(0, len(route))):
#     x = route[i][0]
#
#     y = route[i][1]
#
#     x_coords.append(x)
#
#     y_coords.append(y)
#
# # plot map and path
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# matplotlib.use('TkAgg')
# fig, ax = plt.subplots(figsize=(20, 20))
#
# ax.imshow(grid, cmap=plt.cm.Dark2)
#
# ax.scatter(start[1], start[0], marker="*", color="yellow", s=200)
#
# ax.scatter(goal[1], goal[0], marker="*", color="red", s=200)
#
# ax.plot(y_coords, x_coords, color="black")
#
# plt.show()

# import matplotlib.pyplot as plt
# import os
# import matplotlib
# import numpy as np
#
# def line_intersection(line1, line2):
#     xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
#     ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])
#     def det(a, b):
#         return a[0] * b[1] - a[1] * b[0]
#     div = det(xdiff, ydiff)
#     if div == 0:
#        raise Exception('Lines do not intersect')
#     d = (det(*line1), det(*line2))
#     x = det(d, xdiff) / div
#     y = det(d, ydiff) / div
#     return x, y
#
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# matplotlib.use('TkAgg')
# # Define the circle parameters
# radius = 2.5
# center = (0, 0)
#
# # Angles for the lines in radians (converting from degrees)
# angle_20 = np.deg2rad(20)
# angle_340 = np.deg2rad(340)
# angle_0 = np.deg2rad(0)  # North direction
#
# # Calculate the end points of the lines from the center
# line_20_end = (center[0] + radius * np.cos(angle_20), center[1] + radius * np.sin(angle_20))
# line_340_end = (center[0] + radius * np.cos(angle_340), center[1] + radius * np.sin(angle_340))
#
# # Tangent lines parallel to the 0 degree line
# tangent_1_start = (center[0] + radius * np.cos(np.pi/2), center[1] + radius * np.sin(np.pi/2))
# tangent_2_start = (center[0] - radius * np.cos(np.pi/2), center[1] - radius * np.sin(np.pi/2))
# tangent_1_end = (tangent_1_start[0] + 10 * np.cos(angle_0), tangent_1_start[1] + 10 * np.sin(angle_0))
# tangent_2_end = (tangent_2_start[0] + 10 * np.cos(angle_0), tangent_2_start[1] + 10 * np.sin(angle_0))
#
# # Calculate the intersection points
# intersection_20_tangent_1 = line_intersection((center, line_20_end), (tangent_1_start, tangent_1_end))
# intersection_340_tangent_2 = line_intersection((center, line_340_end), (tangent_2_start, tangent_2_end))
#
# distance_to_20_tangent_1 = np.linalg.norm(np.array(intersection_20_tangent_1)-np.array(center))
# distance_to_340_tangent_1 = np.linalg.norm(np.array(intersection_340_tangent_2)-np.array(center))
#
# # Start plotting
# fig, ax = plt.subplots()
#
# # Draw the circle
# circle = plt.Circle(center, radius, fill=False, color='blue', linestyle='dashed')
# ax.add_artist(circle)
#
# # Draw the lines from the center
# plt.plot([center[0], line_20_end[0]], [center[1], line_20_end[1]], 'r')
# plt.plot([center[0], line_340_end[0]], [center[1], line_340_end[1]], 'r')
#
# # Draw the tangent lines
# plt.plot([tangent_1_start[0], tangent_1_end[0]], [tangent_1_start[1], tangent_1_end[1]], 'g')
# plt.plot([tangent_2_start[0], tangent_2_end[0]], [tangent_2_start[1], tangent_2_end[1]], 'g')
#
# # Plot the intersections
# plt.scatter(*intersection_20_tangent_1, color='purple', zorder=5)
# plt.scatter(*intersection_340_tangent_2, color='purple', zorder=5)
#
# # Draw the lines from the center as arrows
# ax.annotate('', xy=line_20_end, xytext=center, arrowprops=dict(arrowstyle='->', color='red'))
# ax.annotate('', xy=line_340_end, xytext=center, arrowprops=dict(arrowstyle='->', color='red'))
# # Arrow representing the line from center at 0 degrees
# ax.annotate('', xy=(center[0], center[1] + 2 * radius), xytext=center, arrowprops=dict(arrowstyle='->', color='black'))
#
# # Set equal aspect ratio
# ax.set_aspect('equal')
#
# # Set labels and title
# plt.xlabel('X coordinate')
# plt.ylabel('Y coordinate')
# plt.title('Circle with Tangent Lines and Intersections')
#
# # Show the plot with a grid
# plt.grid(True)
# plt.show()

# import matplotlib.pyplot as plt
# import numpy as np
# import os
# import matplotlib
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# matplotlib.use('TkAgg')
# def line_intersection(line1, line2):
#     xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
#     ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])
#
#     def det(a, b):
#         return a[0] * b[1] - a[1] * b[0]
#
#     div = det(xdiff, ydiff)
#     if div == 0:
#         raise Exception('lines do not intersect')
#
#     d = (det(*line1), det(*line2))
#     x = det(d, xdiff) / div
#     y = det(d, ydiff) / div
#     return x, y
#
# # Define the circle parameters
# radius = 2.5
# center = (0, 0)
#
# # Angles for the lines in radians (converting from degrees)
# angle_20 = np.deg2rad(20)
# angle_340 = np.deg2rad(340)
# angle_0 = np.deg2rad(0)  # North direction
#
# # Calculate the end points of the lines from the center
# line_20_end = (center[0] + radius * np.cos(angle_20), center[1] + radius * np.sin(angle_20))
# line_340_end = (center[0] + radius * np.cos(angle_340), center[1] + radius * np.sin(angle_340))
#
# # Extend the lines far beyond the circle for visual clarity
# extended_line_20_end = (center[0] + 2 * radius * np.cos(angle_20), center[1] + 2 * radius * np.sin(angle_20))
# extended_line_340_end = (center[0] + 2 * radius * np.cos(angle_340), center[1] + 2 * radius * np.sin(angle_340))
#
# # Tangent lines parallel to the 0 degree line
# tangent_1_start = (center[0] + radius * np.cos(np.pi/2), center[1] + radius * np.sin(np.pi/2))
# tangent_2_start = (center[0] - radius * np.cos(np.pi/2), center[1] - radius * np.sin(np.pi/2))
# tangent_1_end = (tangent_1_start[0] + 10 * np.cos(angle_0), tangent_1_start[1] + 10 * np.sin(angle_0))
# tangent_2_end = (tangent_2_start[0] + 10 * np.cos(angle_0), tangent_2_start[1] + 10 * np.sin(angle_0))
#
# # Calculate the intersection points
# intersection_20_tangent_1 = line_intersection((center, extended_line_20_end), (tangent_1_start, tangent_1_end))
# intersection_340_tangent_2 = line_intersection((center, extended_line_340_end), (tangent_2_start, tangent_2_end))
#
# # Start plotting
# fig, ax = plt.subplots()
#
# # Draw the circle
# circle = plt.Circle(center, radius, fill=False, color='blue', linestyle='dashed')
# ax.add_artist(circle)
#
# # Draw the lines from the center as arrows
# ax.annotate('', xy=line_20_end, xytext=center, arrowprops=dict(arrowstyle='->', color='red'))
# ax.annotate('', xy=line_340_end, xytext=center, arrowprops=dict(arrowstyle='->', color='red'))
# # Arrow representing the line from center at 0 degrees
# # Adjust the arrow length if necessary
# arrow_length = radius * 1.1  # Slightly longer than the radius
#
# # Plot the arrow representing the line from center at 0 degrees
# ax.annotate('', xy=(center[0], center[1] + arrow_length), xytext=center,
#             arrowprops=dict(arrowstyle='->', lw=2, color='black', zorder=5), label='0-degree Line')
#
# # Draw the tangent lines
# plt.plot([tangent_1_start[0], tangent_1_end[0]], [tangent_1_start[1], tangent_1_end[1]], 'g--', label='Tangent Lines')
# plt.plot([tangent_2_start[0], tangent_2_end[0]], [tangent_2_start[1], tangent_2_end[1]], 'g--')
#
# # Plot the intersections
# plt.scatter(*intersection_20_tangent_1, color='purple', zorder=5, label='Intersection Points')
# plt.scatter(*intersection_340_tangent_2, color='purple', zorder=5)
#
# # Set equal aspect ratio
# ax.set_aspect('equal')
#
# # Set labels and title
# plt.xlabel('X coordinate')
# plt.ylabel('Y coordinate')
# plt.title('Circle with Lines and Intersections')
#
# # Add a legend
# plt.legend()
#
# # Show the plot with a grid
# plt.grid(True)
# plt.show()


import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
matplotlib.use('TkAgg')

def potential_flow_around_circular_obstacle(stream_velocity, obstacle_radius, grid_size):
    Y, X = np.mgrid[-grid_size:grid_size:100j, -grid_size:grid_size:100j]
    U = stream_velocity - (obstacle_radius ** 2) * (X ** 2 - Y ** 2) / (X ** 2 + Y ** 2) ** 2
    V = -2 * obstacle_radius ** 2 * X * Y / (X ** 2 + Y ** 2) ** 2

    speed = np.sqrt(U ** 2 + V ** 2)

    # Streamlines
    strm = plt.streamplot(X, Y, U, V, color=speed, linewidth=1, cmap='autumn')

    # Add circle to represent the rock
    circle = plt.Circle((0, 0), obstacle_radius, color='blue', alpha=0.7)
    plt.gca().add_patch(circle)

    plt.colorbar(strm.lines)
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.title('Potential Flow around a Circular Obstacle')
    plt.axis('equal')
    plt.show()


# Define parameters
stream_velocity = 1.0  # velocity of the stream
obstacle_radius = 1.0  # radius of the circular obstacle, representing a rock
grid_size = 5.0  # size of the grid for plotting

potential_flow_around_circular_obstacle(stream_velocity, obstacle_radius, grid_size)