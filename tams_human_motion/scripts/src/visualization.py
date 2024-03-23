#!/usr/bin/env python

# Human motion reconstruction
# 2022, Philipp Ruppel

import tractor as tr
import tractor.types_double as tt
import numpy as np
import colorsys
from src.configuration import *

# Helper function to visualize keypoint datasets
def visualize_data(data):

    assert(len(data.shape) == 3)
    assert(data.shape[2] == 3)

    frame_count = data.shape[0]
    point_count = data.shape[1]
    # point_indices = [11, 12, 23, 24, 27, 28, 15, 16, ]
    if config["network"]["mediapipe"]:
        point_indices = [27, 28, 15, 16, ] # mediapipe
    else:
        point_indices = [7, 8, 20, 21, ] # smpl


    points = [ ]
    colors = [ ]
    if config["visualization"]["keypoint_size"] > 0:
        for frame_index in range(0, frame_count):
            for index in point_indices:
                if (index%2) == 0:
                    color = colorsys.hsv_to_rgb(index*1.0/point_count,1,1)+(1,)
                else:
                    color = colorsys.hsv_to_rgb(index*1.0/point_count,0.6,0.7)+(1,)
                # color = colorsys.hsv_to_rgb(index*1.0/point_count,0.8,0.8)+(1,)
                colors.append(color)
                points.append(data[frame_index,index])
    tr.visualize_points("keypoints", config["visualization"]["keypoint_size"], colors, points)

    points = [ ]
    colors = [ ]
    if config["visualization"]["trail_width"] > 0:
        for frame_index in range(1, frame_count):
            for index in point_indices:
                if (index%2) == 0:
                    color = colorsys.hsv_to_rgb(index*1.0/point_count,1,1)+(1,)
                else:
                    color = colorsys.hsv_to_rgb(index*1.0/point_count,0.6,0.7)+(1,)
                # color = colorsys.hsv_to_rgb(index*1.0/point_count,0.8,0.8)+(1,)
                colors.append(color)
                colors.append(color)
                points.append(data[frame_index-1,index])
                points.append(data[frame_index-0,index])
    tr.visualize_lines("lines", config["visualization"]["trail_width"], colors, points)
