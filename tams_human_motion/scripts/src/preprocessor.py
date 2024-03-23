#!/usr/bin/env python

# Human motion reconstruction
# 2022, Philipp Ruppel

import tractor as tr
import tractor.types_double as tt
import numpy as np
from src.dataset import *
from src.configuration import *

# Preprocess motion tracking data
class Preprocessor:

    # Preprocess dataset and return results
    def process(self, dataset, robot_model):
        data = dataset.data

        # Scale to match size of human model
        measurement_links = config["preprocessing"]["size_reference_links"]
        joint_states = tt.JointStates(robot_model)
        link_states = robot_model.forward_kinematics(joint_states)
        ff = [ ]
        for mlink in measurement_links:
            l0 = tr.translation(link_states.link_pose(dataset.point_index_to_link_name(mlink[0])))
            l1 = tr.translation(link_states.link_pose(dataset.point_index_to_link_name(mlink[1])))
            pd = data[:,mlink[0],:] - data[:,mlink[1],:]
            pd = np.mean([tr.norm(tt.Vector3(d)).value for d in pd])
            f = tr.norm(l0 - l1).value / pd
            ff.append(f)
        f = np.mean(ff)

        if config["preprocessing"]["scale"]:
            data = data * f * config["preprocessing"]["scale"]

        # Center data
        if config["preprocessing"]["center_data"]:
            data -= np.mean(data, axis=(0,1), keepdims=True)

        # Optionally decimate the data a bit for faster processing
        if data.shape[0] > config["preprocessing"]["max_frames"]:
            decimation = config["preprocessing"]["decimation"]
        else:
            decimation = 1
        data = data[::decimation,:,:]

        # Put data just above floor
        z0 = np.min(data[:,:,2])
        print("z0", z0)
        data[:,:,2] -= z0
        data[:,:,2] -= config["preprocessing"]["ground_offset"]

        # Wrap processed data as new Dataset object
        return Dataset(data=data, time_step=dataset.time_step * dataset.data.shape[0] / data.shape[0])
