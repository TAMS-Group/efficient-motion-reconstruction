#!/usr/bin/env python

# Human motion reconstruction
# 2022, Philipp Ruppel

import tractor as tr
import tractor.types_double as tt
import numpy as np
from src.configuration import *

# Collision avoidance
class CollisionAvoidance:

    def __init__(self, collision_robot):
        self.collision_robot = collision_robot
        self.collision_links = [l for l in self.collision_robot.links if l.shape_count > 0]
        self.weight_parameter = tt.Scalar(0)

    def visualize(self, reconstruction):
        pass

    def make_variables(self, time_frame):
        pass

    def apply(self, reconstruction):

        tr.parameter(self.weight_parameter)

        frame_range = range(reconstruction.frame_count)

        # Prevent penetrating ground plane
        weight = config["collision"]["ground"]["penalty"]
        if weight > 0:
            for frame_index in frame_range:
                for link_name in reconstruction.robot_model.link_names:
                    p = tr.position(reconstruction.link_pose(frame_index, link_name))
                    tr.goal(tr.relu(tr.unpack(p)[2] * tt.Scalar(-weight)))

        # Avoid self-collision between robot links
        padding = config["collision"]["self"]["padding"]
        weight = config["collision"]["self"]["penalty"]
        pairs = config["collision"]["self"]["pairs"]
        continuous = config["collision"]["self"]["continuous"]
        wf = self.weight_parameter * tt.Scalar(weight)
        pad = tt.Scalar(padding)
        if weight > 0:
            print("available collision links", [l.name for l in self.collision_links])
            collision_links = dict([(l.name, l) for l in self.collision_robot.links])
            for frame_index in frame_range:
                frame_index_1 = frame_index
                frame_index_2 = frame_index + 1
                if frame_index_1 in frame_range and frame_index_2 in frame_range:
                    for pair in pairs:
                        for i in range(2):
                            if pair[i] not in collision_links:
                                raise Exception("collision link not found: " + pair[i])
                            if collision_links[pair[i]].shape_count == 0:
                                raise Exception("empty collision link: " + pair[i])
                        link_a = collision_links[pair[0]]
                        link_b = collision_links[pair[1]]
                        for shape_a in link_a.shapes:
                            for shape_b in link_b.shapes:
                                if continuous:
                                    collision = tr.collide(
                                        reconstruction.link_pose(frame_index_1, link_a.name),
                                        reconstruction.link_pose(frame_index_2, link_a.name),
                                        shape_a,
                                        reconstruction.link_pose(frame_index_1, link_b.name),
                                        reconstruction.link_pose(frame_index_2, link_b.name),
                                        shape_b
                                        )
                                    tr.goal(tr.relu(wf * (pad + tr.dot(collision.point_b_0, collision.normal) - tr.dot(collision.point_a_0, collision.normal))))
                                    tr.goal(tr.relu(wf * (pad + tr.dot(collision.point_b_1, collision.normal) - tr.dot(collision.point_a_1, collision.normal))))
                                    tr.goal(tr.relu(wf * (pad + tr.dot(collision.point_b_0, collision.normal) - tr.dot(collision.point_a_1, collision.normal))))
                                    tr.goal(tr.relu(wf * (pad + tr.dot(collision.point_b_1, collision.normal) - tr.dot(collision.point_a_0, collision.normal))))
                                else:
                                    collision = tr.collide(
                                        reconstruction.link_pose(frame_index_1, link_a.name),
                                        shape_a,
                                        reconstruction.link_pose(frame_index_2, link_b.name),
                                        shape_b
                                        )
                                    tr.goal(tr.relu(wf * (pad + tr.dot(collision.point_b - collision.point_a, collision.normal))))
