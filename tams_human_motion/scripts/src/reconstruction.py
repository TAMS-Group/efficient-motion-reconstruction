#!/usr/bin/env python

# Human motion reconstruction
# 2022, Philipp Ruppel

import tractor as tr
import tractor.types_double as tt
import numpy as np
from src.configuration import *

# This class stores the current motion reconstruction and is modified during the
# optimization process
class Reconstruction:

    def __init__(self, robot_model, frame_count, time_step):
        self.frame_count = frame_count
        self.robot_model = robot_model
        self.time_step = time_step

        # For each time step, we store / optimize positions for all the joints
        self.joint_trajectory = [tt.JointStates(robot_model) for i in range(frame_count)]

    # Clamp frame index if it is before the first or after the last time frame
    def clamp_trajectory_index(self, i):
        return max(0, min(len(self.joint_trajectory) - 1, i))

    # Return joint states for specific time frame
    def joint_states(self, i):
        return self.joint_trajectory[self.clamp_trajectory_index(i)]

    # Map from joint angles to Cartesian link poses and store link poses for
    # later access
    def compute_kinematics(self):
        for time_step in range(self.frame_count):
            self.apply_mimic(time_step)
        self.link_trajectory = [self.robot_model.forward_kinematics(joint_states) for joint_states in self.joint_trajectory]

    # Return link poses for specific time frame
    def link_states(self, i):
        return self.link_trajectory[self.clamp_trajectory_index(i)]

    # Return pose of a specific link at a specific time frame
    def link_pose(self, frame, name):
        return self.link_trajectory[self.clamp_trajectory_index(frame)].link_pose(name)

    # Return position of a specific link at a specific time frame
    def link_position(self, frame, name):
        return tr.position(self.link_pose(frame, name))

    # Return linear and angular velocity of a specific link at a specific time
    # frame
    def link_velocity(self, frame, name):
        #return tr.residual(tr.inverse(self.link_pose(frame - 1, name)) * self.link_pose(frame + 1, name)) * tt.Scalar(0.5 / self.time_step)
        #return tr.residual(tr.inverse(self.link_pose(frame - 1, name)) * self.link_pose(frame, name)) * tt.Scalar(1.0 / self.time_step)
        #return tr.residual(tr.inverse(self.link_pose(frame, name)) * self.link_pose(frame - 1, name)) * tt.Scalar(-1.0 / self.time_step)
        va = tr.residual(tr.inverse(self.link_pose(frame, name)) * self.link_pose(frame - 1, name)) * tt.Scalar(-1.0 / self.time_step)
        vb = tr.residual(tr.inverse(self.link_pose(frame, name)) * self.link_pose(frame + 1, name)) * tt.Scalar(1.0 / self.time_step)
        return (va + vb) * tt.Scalar(0.5)

    def apply_mimic(self, time_step):
        for mimic_group in config["joints"]["mimic"]:
            for i in range(1, len(mimic_group)):
                a = mimic_group[i]
                b = mimic_group[0]
                self.joint_states(time_step).joint_state(a).position = self.joint_states(time_step).joint_state(b).position * tt.Scalar(1)

    # Create free variables for all joints
    def make_variables(self, time_step):

        joint_margins = config["joints"]["position_limit_margin"]
        joint_limit_cost = config["joints"]["position_limit_penalty"]

        velocity_limit = config["joints"]["velocity_limit"]
        velocity_penalty = config["joints"]["velocity_penalty"]

        acceleration_limit = config["joints"]["acceleration_limit"]
        acceleration_penalty = config["joints"]["acceleration_penalty"]

        zeroing = dict(config["joints"]["zero"])

        is_fixed = dict([[j.name, False] for j in self.robot_model.joints])

        mimic_div = dict([[j.name, 1.0] for j in self.robot_model.joints])
        for mimic_group in config["joints"]["mimic"]:
            for n in mimic_group:
                mimic_div[n] = len(mimic_group)
            for n in mimic_group[1:]:
                is_fixed[n] = True

        for i in range(self.robot_model.joint_count):
            joint_model = self.robot_model.joint(i)
            joint_state = self.joint_states(time_step).joint_state(i)
            if not is_fixed[joint_model.name]:
                if isinstance(joint_model, tt.ScalarJointModelBase):
                    tr.variable(joint_state.position)
                if isinstance(joint_model, tt.FloatingJointModel):
                    tr.variable(joint_state.pose)

        self.apply_mimic(time_step)

        for i in range(self.robot_model.joint_count):
            joint_model = self.robot_model.joint(i)
            joint_state = self.joint_states(time_step).joint_state(i)
            mimic_f = 1.0 / mimic_div[joint_model.name]
            if isinstance(joint_model, tt.ScalarJointModelBase):
                if joint_model.limits:
                    tr.goal(tr.relu(joint_model.limits.lower - joint_state.position + tt.Scalar(joint_margins)) * tt.Scalar(mimic_f * joint_limit_cost))
                    tr.goal(tr.relu(joint_state.position - joint_model.limits.upper + tt.Scalar(joint_margins)) * tt.Scalar(mimic_f * joint_limit_cost))
                    if joint_model.name in zeroing:
                        tr.goal(joint_state.position * tt.Scalar(zeroing[joint_model.name]))
                    if time_step > 0:
                        prev_joint_state = self.joint_states(time_step - 1).joint_state(i)
                        joint_velocity = (joint_state.position - prev_joint_state.position) * tt.Scalar(1.0 / self.time_step)
                        if velocity_limit > 0 and velocity_penalty > 0:
                            tr.goal(tr.relu(joint_velocity - tt.Scalar(velocity_limit)) * tt.Scalar(mimic_f * velocity_penalty))
                            tr.goal(tr.relu(-joint_velocity - tt.Scalar(velocity_limit)) * tt.Scalar(mimic_f * velocity_penalty))
                        if time_step > 1:
                            if acceleration_limit > 0 and acceleration_penalty > 0:
                                prev_joint_state_2 = self.joint_states(time_step - 2).joint_state(i)
                                joint_velocity_2 = (prev_joint_state.position - prev_joint_state_2.position) * tt.Scalar(1.0 / self.time_step)
                                joint_acceleration = (joint_velocity_2 - joint_velocity) * tt.Scalar(1.0 / self.time_step)
                                tr.goal(tr.relu(joint_acceleration - tt.Scalar(acceleration_limit)) * tt.Scalar(mimic_f * acceleration_penalty))
                                tr.goal(tr.relu(-joint_acceleration - tt.Scalar(acceleration_limit)) * tt.Scalar(mimic_f * acceleration_penalty))
