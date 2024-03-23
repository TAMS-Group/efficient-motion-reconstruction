#!/usr/bin/env python

# Human motion reconstruction
# 2022, Philipp Ruppel

import tractor as tr
import tractor.types_double as tt
import numpy as np
from src.configuration import *
import colorsys
import tf.transformations as transformations
# Loss functions to make the reconstructions follow the recordings
class Retargeting:

    def __init__(self, dataset):
        self.dataset = dataset

    def visualize(self, reconstruction):
        # Fetch parameters
        reconstruct_line_width = config["visualization"]["reconstruct_line_width"]
        reconstruct_point_size = config["visualization"]["reconstruct_point_size"]

        frame_count = self.dataset.frame_count
        point_count = self.dataset.data.shape[1]
        if config["network"]["mediapipe"]:
            point_indices = [27, 28, 15, 16, ] # mediapipe
        else:
            point_indices = [7, 8, 20, 21, ] # SMPL
        frame_range = range(self.dataset.frame_count)


        # Visualize reconstruct point
        if reconstruct_point_size > 0:
            points = [ ]
            colors = [ ]
            for frame_index in frame_range:
                link_states = reconstruction.link_states(frame_index)
                for index in point_indices:
                    link = self.dataset.point_index_to_link_name(index)
                    point = tr.position(link_states.link_pose(link))
                    if (index%2) == 0:
                        color = colorsys.hsv_to_rgb(index*1.0/point_count,1,1)+(1,)
                    else:
                        color = colorsys.hsv_to_rgb(index*1.0/point_count,1,1)+(0.8,)
                    colors.append(color)
                    points.append(point.value)
            tr.visualize_points("reconstruct_points", reconstruct_point_size, colors, points)

        # Visualize reconstruct trajectory
        if reconstruct_line_width > 0:
            points = [ ]
            colors = [ ]
            for frame_index in range(1, frame_count):
                link_states_0 = reconstruction.link_states(frame_index - 1)
                link_states_1 = reconstruction.link_states(frame_index - 0)
                for index in point_indices:
                    link = self.dataset.point_index_to_link_name(index)
                    point_0 = tr.position(link_states_0.link_pose(link))
                    point_1 = tr.position(link_states_1.link_pose(link))
                    if (index%2) == 0:
                        color = colorsys.hsv_to_rgb(index*1.0/point_count,1,1)+(1,)
                    else:
                        color = colorsys.hsv_to_rgb(index*1.0/point_count,1,1)+(0.8,)
                    colors.append(color)
                    colors.append(color)
                    points.append(point_0.value)
                    points.append(point_1.value)
            tr.visualize_lines("reconstruct_trajectory", reconstruct_line_width, colors, points)


    def make_variables(self, time_frame):
        pass

    def apply(self, reconstruction):
        frame_range = range(self.dataset.frame_count)

        # Get camera orientation
        angles = config["camera"]["orientation"]
        camera_orientation = tt.Orientation(transformations.quaternion_from_euler(angles["roll"], angles["pitch"], angles["yaw"]))

        # Get camera position
        xyz = config["camera"]["position"]
        camera_position = tt.Vector3(xyz["x"], xyz["y"], xyz["z"])

        # We can optionally try to optimize the camera orientation
        orientation_uncertainty = config["camera"]["orientation_uncertainty"]
        rx = tt.Scalar()
        ry = tt.Scalar()
        if orientation_uncertainty > 0:
            # rx = tt.Scalar()
            # ry = tt.Scalar()
            tr.variable(rx)
            tr.variable(ry)
            tr.goal(rx * tt.Scalar(1.0 / orientation_uncertainty))
            tr.goal(ry * tt.Scalar(1.0 / orientation_uncertainty))
            camera_orientation += tt.Vector3(rx, ry, tt.Scalar(0))

        # We can optionally try to optimize the camera position (height above
        # ground)
        z = tt.Scalar()
        height_uncertainty = config["camera"]["height_uncertainty"]
        if height_uncertainty > 0:
            tr.variable(z)
            tr.goal(z * tt.Scalar(1.0 / height_uncertainty))

        # Assemble camera pose
        camera_pose = tt.Pose(camera_position + tt.Vector3(tt.Scalar(0), tt.Scalar(0), z), camera_orientation)

        # Allow different weights along each axis
        wxyz = config["retargeting"]["axis_weights"]
        weight_matrix = tt.Matrix3([[wxyz["x"],0,0], [0,wxyz["y"],0], [0,0,wxyz["z"]]])

        # Optionally increase the weight of the last sample by this factor
        end_weight = config["retargeting"]["end_weight"]

        # Loss function to try and match link directions with relative
        # directions between keypoints
        if 1:
            weight = config["retargeting"]["link_weight"]
            if weight > 0:
                connections = config["retargeting"]["links"]
                for frame_index in frame_range:
                    if frame_index == self.dataset.frame_count - 1: weight *= end_weight
                    link_states = reconstruction.link_states(frame_index)
                    for conn in connections:
                        link_0 = self.dataset.point_index_to_link_name(conn[0][0])
                        link_1 = self.dataset.point_index_to_link_name(conn[0][1])
                        actual = tr.position(link_states.link_pose(link_1)) - tr.position(link_states.link_pose(link_0))
                        target = camera_pose * self.dataset.point(frame_index, conn[0][1]) - camera_pose * self.dataset.point(frame_index, conn[0][0])
                        tr.goal((tr.normalized(target) - tr.normalized(actual)) * tt.Scalar(conn[1] * weight))

        # Loss function to try and match link positions with keypoint locations
        if 1:
            weight = config["retargeting"]["point_weight"]
            if weight > 0:
                links = dict(config["retargeting"]["points"])
                for frame_index in frame_range:
                    if frame_index == self.dataset.frame_count - 1: weight *= end_weight
                    for m in self.dataset.link_mappings:
                        if m[1] in links:
                            w = links[m[1]]
                            actual = tr.translation(reconstruction.link_states(frame_index).link_pose(m[2]))
                            target = camera_pose * self.dataset.point(frame_index, m[0])
                            tr.goal(weight_matrix * (target - actual) * tt.Scalar(w * weight))

        # Loss function to try and match Cartesian velocities
        if 1:
            weight = config["retargeting"]["velocity_weight"]
            if weight > 0:
                links = dict(config["retargeting"]["points"])
                for frame_index in frame_range:
                    if frame_index == self.dataset.frame_count - 1: weight *= end_weight
                    for m in self.dataset.link_mappings:
                        if m[1] in links:
                            w = links[m[1]]
                            i = reconstruction.clamp_trajectory_index(frame_index - 1)
                            j = reconstruction.clamp_trajectory_index(frame_index + 1)
                            actual = reconstruction.link_position(i, m[2]) - reconstruction.link_position(j, m[2])
                            target = camera_pose * self.dataset.point(i, m[0]) - camera_pose * self.dataset.point(j, m[0])
                            tr.goal(weight_matrix * (target - actual) * tt.Scalar(w * weight / (j - i) / self.dataset.time_step))
