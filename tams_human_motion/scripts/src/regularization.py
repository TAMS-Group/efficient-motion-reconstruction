#!/usr/bin/env python

# Human motion reconstruction
# 2022, Philipp Ruppel

import tractor as tr
import tractor.types_double as tt
import numpy as np
from src.configuration import *

# Regularizers to prefer nice and smooth motions
class Regularization:

    def __init__(self):
        self.weight_parameter = tt.Scalar(1)

    def visualize(self, reconstruction):
        pass

    def make_variables(self, time_frame):
        pass

    def apply(self, reconstruction):

        tr.parameter(self.weight_parameter)

        frame_range = range(-1, reconstruction.frame_count)

        weights = config["regularization"]["weights"]

        # Prefer smooth motions by minimizing joint accelerations
        if 1:
            revolute_weight = weights["revolute_joint_acceleration"]
            floating_weight = weights["floating_joint_acceleration"]
            for frame_index in frame_range:
                for joint_index in range(reconstruction.robot_model.joint_count):
                    joint_model = reconstruction.robot_model.joint(joint_index)
                    if revolute_weight > 0:
                        if revolute_weight != 0:
                            if isinstance(joint_model, tt.ScalarJointModelBase):
                                pos0 = reconstruction.joint_states(frame_index-2).joint_state(joint_index).position
                                pos1 = reconstruction.joint_states(frame_index-1).joint_state(joint_index).position
                                pos2 = reconstruction.joint_states(frame_index-0).joint_state(joint_index).position
                                tr.goal((pos1 + pos1 - pos0 - pos2) * (self.weight_parameter * tt.Scalar(revolute_weight)))
                    if floating_weight > 0:
                        if floating_weight != 0:
                            if isinstance(joint_model, tt.FloatingJointModel):
                                pose0 = reconstruction.joint_states(frame_index-2).joint_state(joint_index).pose
                                pose1 = reconstruction.joint_states(frame_index-1).joint_state(joint_index).pose
                                pose2 = reconstruction.joint_states(frame_index-0).joint_state(joint_index).pose
                                pose1_inv = tr.inverse(pose1)
                                v0n = tr.residual(pose1_inv * pose0)
                                v1p = tr.residual(pose1_inv * pose2)
                                tr.goal((v0n + v1p) * (self.weight_parameter * tt.Scalar(floating_weight)))

        # Avoid unnecessary motions by minimizing joint velocities
        if 1:
            revolute_weight = weights["revolute_joint_velocity"]
            floating_weight = weights["floating_joint_velocity"]
            for frame_index in frame_range:
                for joint_index in range(reconstruction.robot_model.joint_count):
                    joint_model = reconstruction.robot_model.joint(joint_index)
                    if revolute_weight > 0:
                        if isinstance(joint_model, tt.ScalarJointModelBase):
                            pos0 = reconstruction.joint_states(frame_index-2).joint_state(joint_index).position
                            pos1 = reconstruction.joint_states(frame_index-1).joint_state(joint_index).position
                            tr.goal((pos1 - pos0) * (self.weight_parameter * tt.Scalar(revolute_weight)))
                    if floating_weight > 0:
                        if isinstance(joint_model, tt.FloatingJointModel):
                            pose0 = reconstruction.joint_states(frame_index-2).joint_state(joint_index).pose
                            pose1 = reconstruction.joint_states(frame_index-1).joint_state(joint_index).pose
                            v0 = tr.residual(tr.inverse(pose0) * pose1)
                            tr.goal(v0 * (self.weight_parameter * tt.Scalar(floating_weight)))

        # Prefer smooth motions by minimizing link accelerations
        if 1:
            weight = weights["cartesian_link_acceleration"]
            if weight > 0:
                for frame_index in frame_range:
                    for link_name in reconstruction.robot_model.link_names:
                        p0 = tr.position(reconstruction.link_states(frame_index - 0).link_pose(link_name))
                        p1 = tr.position(reconstruction.link_states(frame_index - 1).link_pose(link_name))
                        p2 = tr.position(reconstruction.link_states(frame_index - 2).link_pose(link_name))
                        tr.goal((p1 + p1 - p2 - p0) * (self.weight_parameter * tt.Scalar(weight)))

        # Avoid unnecessary motions by minimizing link velocities
        if 1:
            weight = weights["cartesian_link_velocity"]
            if weight > 0:
                for frame_index in frame_range:
                    for link_name in reconstruction.robot_model.link_names:
                        p0 = tr.position(reconstruction.link_states(frame_index - 0).link_pose(link_name))
                        p1 = tr.position(reconstruction.link_states(frame_index - 1).link_pose(link_name))
                        tr.goal((p0 - p1) * (self.weight_parameter * tt.Scalar(weight)))
