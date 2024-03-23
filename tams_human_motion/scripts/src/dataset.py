#!/usr/bin/env python

# Human motion reconstruction
# 2022, Philipp Ruppel

import tractor as tr
import tractor.types_double as tt
import numpy as np
from src.configuration import *

# This class loads and represents keypoint trajectories from the motion
# tracking networks
class Dataset:

    # Loads or wraps a dataset
    # set path to load a dataset from a file
    # set data to wrap an array that already exists in memory
    def __init__(self, time_step, path = False, data = False):

        # currently we've always got 33 points
        point_count = config["retargeting"]["keypoint_count"]
        print("point count", point_count)

        # load dataset from file, if desired
        if (data is False) and (path is not False):

            # Load data
            print("loading data file", path)
            data = np.genfromtxt(path, delimiter=',')

            # Turn into tensor
            # 1st dimension: time step
            # 2nd dimension: marker index
            # 3rd dimension: coordinate (x/y/z)
            frame_count = data.shape[0]
            data = data[0:frame_count,0:point_count*3]
            data = data.reshape([frame_count,point_count,3])

            if config["network"]["YZswitch"]:
                data = data[:,:,[0,2,1]] # for mediapipe

        # Allow manually scaling dataset along each axis
        data[:,:,0] *= config["recording"]["scale"]["x"]
        data[:,:,1] *= config["recording"]["scale"]["y"]
        data[:,:,2] *= config["recording"]["scale"]["z"]

        # Store data as member variables
        self.data = data
        self.frame_count = data.shape[0]
        self.point_count = point_count

        # Load mapping between tracking markers, URDF links and marker indices
        # in txt file
        self.link_mappings = config["retargeting"]["mapping"]
        print("link mappings", self.link_mappings)
        self.point_index_to_link_name_map = { }
        for lm in self.link_mappings:
            self.point_index_to_link_name_map[lm[0]] = lm[2]

        # Store time between two consecutive time frames
        self.time_step = time_step

    # Map keypoint indices to URDF link names according to table loaded above
    def point_index_to_link_name(self, index):
        return self.point_index_to_link_name_map[index]

    # Clamp time frame index if it lies before the first or after the last frame
    def clamp_index(self, i):
        return max(0, min(self.frame_count - 1, i))

    # Return position of a tracking marker
    def point(self, time_step, point_index):
        return tt.Vector3(self.data[self.clamp_index(time_step), point_index])
