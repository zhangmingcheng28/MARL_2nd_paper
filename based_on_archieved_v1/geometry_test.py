# -*- coding: utf-8 -*-
"""
@Time    : 8/11/2023 3:12 PM
@Author  : Mingcheng
@FileName: 
@Description: 
@Package dependency:
"""
import numpy as np
from shapely.geometry import LineString, Point, Polygon


goal = Point(np.array([536, 356])).buffer(1, cap_style='round')
pre_pos = np.array([530.81, 353.08])
cur_pos = np.array([534.12, 355.86])