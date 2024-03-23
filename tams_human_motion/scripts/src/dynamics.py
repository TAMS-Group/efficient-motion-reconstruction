#!/usr/bin/env python

# Human motion reconstruction
# 2022, Philipp Ruppel

import tractor as tr
import tractor.types_double as tt
import numpy as np
from src.configuration import *


# Helper class to store information related to the dynamics of a joint
class JointWrench:
    def __init__(self, time_frame, parent_link, joint_model, child_link):
        self.time_frame = time_frame
        self.joint_model = joint_model
        self.wrench_variable = tt.Twist()
        self.child_link = child_link
        self.parent_link = parent_link


# Helper class to store information about a contact
class Contact:
    def __init__(self, time_frame, body_link, target_link, point):
        self.time_frame = time_frame
        self.body_link = body_link
        self.target_link = target_link
        self.force_variable = tt.Vector3()
        self.point = point


# This class defines loss terms to encourage physically consistent motion
# reconstructions
class Dynamics:


    def __init__(self, robot_model, frame_count, time_step):

        # Store parameters
        self.time_step = time_step
        self.frame_min = 0
        self.frame_max = frame_count
        self.frame_range = range(self.frame_min, self.frame_max)
        self.frame_count = frame_count

        # Find joints and attached bodies
        self.joint_wrenches = { }
        for time_frame in self.frame_range:
            self.joint_wrenches[time_frame] = [ ]
            for joint_model in robot_model.joints:
                if isinstance(joint_model, tt.RevoluteJointModel):
                    parent_link = joint_model.parent_link
                    while (parent_link.inertia.mass.value <= 0) or isinstance(parent_link.parent_joint, tt.FixedJointModel):
                        parent_link = parent_link.parent_joint.parent_link
                    child_link = joint_model.child_link
                    if (child_link is not None) and (parent_link is not None) and (child_link.inertia.mass.value > 0):
                        self.joint_wrenches[time_frame].append(JointWrench(time_frame, parent_link, joint_model, child_link))

        # Define points where we might expect contacts
        # (currently hardcoded, TODO: move to config files)
        foot_z = -0.064752
        toe_x = 0.14456
        foot_w = 0.05
        self.contacts = { }
        for time_frame in self.frame_range:
            self.contacts[time_frame] = [ ]

            self.contacts[time_frame].append(Contact(time_frame, robot_model.link("LeftLowerLeg_f1"), robot_model.link("LeftFoot"), tt.Vector3(toe_x,+foot_w,foot_z)))
            self.contacts[time_frame].append(Contact(time_frame, robot_model.link("LeftLowerLeg_f1"), robot_model.link("LeftFoot"), tt.Vector3(toe_x,-foot_w,foot_z)))
            self.contacts[time_frame].append(Contact(time_frame, robot_model.link("LeftLowerLeg_f1"), robot_model.link("LeftFoot"), tt.Vector3(0,+foot_w,foot_z)))
            self.contacts[time_frame].append(Contact(time_frame, robot_model.link("LeftLowerLeg_f1"), robot_model.link("LeftFoot"), tt.Vector3(0,-foot_w,foot_z)))

            self.contacts[time_frame].append(Contact(time_frame, robot_model.link("RightLowerLeg_f1"), robot_model.link("RightFoot"), tt.Vector3(toe_x,+foot_w,foot_z)))
            self.contacts[time_frame].append(Contact(time_frame, robot_model.link("RightLowerLeg_f1"), robot_model.link("RightFoot"), tt.Vector3(toe_x,-foot_w,foot_z)))
            self.contacts[time_frame].append(Contact(time_frame, robot_model.link("RightLowerLeg_f1"), robot_model.link("RightFoot"), tt.Vector3(0,+foot_w,foot_z)))
            self.contacts[time_frame].append(Contact(time_frame, robot_model.link("RightLowerLeg_f1"), robot_model.link("RightFoot"), tt.Vector3(0,-foot_w,foot_z)))

            self.contacts[time_frame].append(Contact(time_frame, robot_model.link("LeftForeArm"), robot_model.link("LeftHand"), tt.Vector3(+0.11145/2,+0.16718,0)))
            self.contacts[time_frame].append(Contact(time_frame, robot_model.link("LeftForeArm"), robot_model.link("LeftHand"), tt.Vector3(-0.11145/2,+0.16718,0)))
            self.contacts[time_frame].append(Contact(time_frame, robot_model.link("RightForeArm"), robot_model.link("RightHand"), tt.Vector3(+0.11145/2,-0.16718,0)))
            self.contacts[time_frame].append(Contact(time_frame, robot_model.link("RightForeArm"), robot_model.link("RightHand"), tt.Vector3(-0.11145/2,-0.16718,0)))

        # Find root link of each physics body
        # (multiple links connected via fixed joints will be merged into a
        # single body)
        self.body_links = [joint.child_link for joint in robot_model.joints if ((not isinstance(joint, tt.FixedJointModel)) and (joint.child_link.inertia.mass.value > 0))]
        print("body_links", [l.name for l in self.body_links])

        # Adjustable parameter to adjust weight of the physical consistency loss
        self.weight_parameter = tt.Scalar(0)


    # Visualize forces and torques using TAMSVIZ
    def visualize(self, reconstruction):

        # Print current gravity value in case we don't know the frame rate of
        # the recording and are solving for gravity
        # print "gravity", self.gravity_variable.value

        # Fetch parameters
        contact_force_length = config["visualization"]["force_scaling"] * config["dynamics"]["scaling"]
        joint_force_length = config["visualization"]["force_scaling"] * config["dynamics"]["scaling"]
        joint_torque_length = config["visualization"]["torque_scaling"] * config["dynamics"]["scaling"]
        line_width = config["visualization"]["line_width"]

        # Visualize joint forces
        points = [ ]
        for time_frame in self.frame_range:
            for w in self.joint_wrenches[time_frame]:
                point = reconstruction.link_position(w.time_frame, w.child_link.name)
                force = tr.translation(w.wrench_variable) * tt.Scalar(joint_force_length * 0.5)
                points.append((point-force).value)
                points.append((point+force).value)
        tr.visualize_lines("joint_forces", line_width, (1,0,0,1), points)

        # Visualize joint torques
        points = [ ]
        for time_frame in self.frame_range:
            for w in self.joint_wrenches[time_frame]:
                point = reconstruction.link_position(w.time_frame, w.child_link.name)
                torque = tr.rotation(w.wrench_variable) * tt.Scalar(joint_torque_length * 0.5)
                points.append((point-torque).value)
                points.append((point+torque).value)
        tr.visualize_lines("joint_torques", line_width, (0,0,1,1), points)

        # Visualize contact forces
        points = [ ]
        for time_frame in self.frame_range:
            for c in self.contacts[time_frame]:
                point = reconstruction.link_pose(c.time_frame, c.target_link.name) * c.point
                force = c.force_variable * tt.Scalar(contact_force_length)
                points.append((point).value)
                points.append((point - force).value)
        tr.visualize_lines("contact_forces", line_width, (1,1,0,1), points)


    # Create free variables for joint forces, joint torques and contact forces
    def make_variables(self, time_frame):

        for joint_wrench in self.joint_wrenches[time_frame]:
            tr.variable(joint_wrench.wrench_variable)

        for contact in self.contacts[time_frame]:
            tr.variable(contact.force_variable)


    # Apply physical consistency terms
    def apply(self, reconstruction):


        # Allow changing the weight parameter during optimization
        tr.parameter(self.weight_parameter)


        # Store transformations across fixed links for later use
        default_pose = reconstruction.robot_model.forward_kinematics(tt.JointStates(reconstruction.robot_model))


        # Fetch parameters
        cfg = config["dynamics"]

        contact_force_scaling = cfg["scaling"]
        joint_force_scaling = cfg["scaling"]
        joint_torque_scaling = cfg["scaling"]

        contact_force_cost = cfg["contact_force_cost"]
        joint_force_cost = cfg["joint_force_cost"]
        joint_torque_cost = cfg["joint_torque_cost"]

        contact_stick_cost = cfg["contact_stickiness_penalty"]

        contact_force_distance_cost = cfg["contact_force_distance_penalty"]
        contact_force_slip_cost = cfg["contact_force_slip_penalty"]

        contact_slip_frames = cfg["contact_slip_frames"]

        position_cost = cfg["translational_consistency_loss"]
        orientation_cost = cfg["rotational_consistency_loss"]

        substeps = cfg["substeps"]

        steps = cfg["steps"]

        smoothness = cfg["smoothness"]

        friction_cone_penalty = cfg["friction_cone_penalty"]

        stride = cfg["stride"]

        linear_velocity_consistency = cfg["linear_velocity_consistency"]
        angular_velocity_consistency = cfg["angular_velocity_consistency"]

        force_consistency = cfg["force_consistency"]
        torque_consistency = cfg["torque_consistency"]

        end_weight = cfg["end_weight"]

        robot_model = reconstruction.robot_model
        frame_count = reconstruction.frame_count


        # If we don't know the frame rate of the recording, maybe we could
        # simply try to optimize it. This easiest way to do that seems to be to
        # make gravity a free parameter, since gravity is basically the time
        # reference in our case.
        gravity_variable = tt.Scalar(cfg["gravity"])
        if cfg["gravity_uncertainty"] > 0:
            tr.variable(gravity_variable)
            tr.goal((gravity_variable - tt.Scalar(cfg["gravity"])) * tt.Scalar(1.0 / cfg["gravity_uncertainty"]))
        gravity = tt.Vector3(tt.Scalar(0), tt.Scalar(0), gravity_variable)
        self.gravity_variable = gravity_variable


        # Define friction cone for contacts
        # Currently hadcoded (TODO: move to config.yaml)
        # As is common in physics engines, we approximate the friction cone as a
        # pyramid
        friction_cone = [
            tt.Vector3(+1,  0, -1),
            tt.Vector3(-1,  0, -1),
            tt.Vector3( 0, +1, -1),
            tt.Vector3( 0, -1, -1),
        ]
        friction_cone = [tr.normalized(normal) * tt.Scalar(friction_cone_penalty) * self.weight_parameter for normal in friction_cone]


        # We can optionally assign a cost to non-zero joint forces and torques,
        # simulating the human or robot being a bit lazy or avoiding pain.
        # Since we're solving for full 6D wrenches, we can also simulate the
        # human trying to not break their joints.
        for time_frame in self.frame_range:
            for joint_wrench in self.joint_wrenches[time_frame]:
                if joint_torque_cost != 0: tr.goal(tr.rotation(joint_wrench.wrench_variable) * tt.Scalar(joint_torque_cost) * self.weight_parameter)
                if joint_force_cost != 0: tr.goal(tr.translation(joint_wrench.wrench_variable) * tt.Scalar(joint_force_cost) * self.weight_parameter)


        # Create contact terms
        for time_frame in self.frame_range:
            for contact in self.contacts[time_frame]:

                # Get information about link and contact
                contact_point = reconstruction.link_pose(time_frame, contact.target_link.name) * contact.point
                contact_force = contact.force_variable
                contact_force_z = tr.unpack(contact_force)[2]
                contact_height = tr.unpack(contact_point)[2]

                # Forbid contact forces from pulling things together (not needed
                # anymore)
                if contact_stick_cost != 0: tr.goal(tr.relu(contact_force_z * tt.Scalar(-contact_stick_cost)) * self.weight_parameter)

                # There can only be contact forces if to bodies actually touch
                if contact_force_distance_cost != 0: tr.goal(contact_force * contact_height * tt.Scalar(contact_force_distance_cost) * self.weight_parameter)

                # We can optionally apply a penalty to non-zero contact forces /
                # simulate avoiding being hit too hard / pain
                if contact_force_cost != 0: tr.goal(contact_force * tt.Scalar(contact_force_cost) * self.weight_parameter)

                # Avoid slipping
                for i in range(-contact_slip_frames, contact_slip_frames):
                    slip_p0 = reconstruction.link_pose(time_frame + i + 0, contact.target_link.name) * contact.point
                    slip_p1 = reconstruction.link_pose(time_frame + i + 1, contact.target_link.name) * contact.point
                    slip_vector = slip_p0 - slip_p1
                    slip_xyz = tr.unpack(slip_vector)
                    if contact_force_slip_cost > 0:
                        tr.goal(contact_force * slip_xyz[0] * tt.Scalar(contact_force_slip_cost * 1.0) * self.weight_parameter)
                        tr.goal(contact_force * slip_xyz[1] * tt.Scalar(contact_force_slip_cost * 1.0) * self.weight_parameter)
                        #tr.goal(contact_force * slip_xyz[2] * tt.Scalar(contact_force_slip_cost * 1.0) * self.weight_parameter)

                # Enforce friction cones
                if friction_cone_penalty > 0:
                    for normal in friction_cone:
                        tr.goal(tr.relu(tr.dot(normal, contact_force)) * self.weight_parameter)


        # We can optionally prefer smooth force and torque profiles
        if smoothness > 0:
            contact_count = len(self.contacts[self.frame_min])
            for time_frame in range(self.frame_min + 1, self.frame_max):
                for contact_index in range(contact_count):
                    w0 = self.contacts[time_frame - 0][contact_index].force_variable
                    w1 = self.contacts[time_frame - 1][contact_index].force_variable
                    tr.goal((w0 - w1) * tt.Scalar(smoothness) * self.weight_parameter)
            joint_count = len(self.joint_wrenches[self.frame_min])
            for time_frame in range(self.frame_min + 1, self.frame_max):
                for joint_index in range(joint_count):
                    w0 = self.joint_wrenches[time_frame - 0][joint_index].wrench_variable
                    w1 = self.joint_wrenches[time_frame - 1][joint_index].wrench_variable
                    tr.goal((w0 - w1) * tt.Scalar(smoothness) * self.weight_parameter)

        # Simplified dynamics
        #for time_frame in self.frame_range:
        #for time_frame in range(0, self.frame_count - 1):
        for time_frame in range(0, self.frame_count):

            ref_point = tt.Vector3.zero
            ref_div = tt.Scalar(0)
            for link_model in self.body_links:
                link_name = link_model.name
                link_inertia = link_model.inertia
                ref_point += reconstruction.link_position(time_frame, link_name) * link_inertia.mass
                ref_div += link_inertia.mass
            ref_point *= tt.Scalar(1) / ref_div

            inertial_forces = tt.Vector3.zero
            inertial_torques = tt.Vector3.zero

            for link_model in self.body_links:

                link_name = link_model.name
                link_inertia = link_model.inertia

                pose_prev = reconstruction.link_pose(time_frame - 1, link_name)
                pose_curr = reconstruction.link_pose(time_frame + 0, link_name)
                pose_next = reconstruction.link_pose(time_frame + 1, link_name)

                position_curr = tr.position(pose_curr)
                orientation_curr = tr.orientation(pose_curr)

                pose_curr_inv = tr.inverse(pose_curr)

                velocity_curr = tr.residual(pose_curr_inv * pose_prev) * tt.Scalar(-1.0 / self.time_step)
                velocity_next = tr.residual(pose_curr_inv * pose_next) * tt.Scalar(+1.0 / self.time_step)

                velocity_curr_linear = tr.translation(velocity_curr)
                velocity_curr_angular = tr.rotation(velocity_curr)

                velocity_next_linear = tr.translation(velocity_next)
                velocity_next_angular = tr.rotation(velocity_next)

                momentum_curr_linear = orientation_curr * (link_inertia.mass * velocity_curr_linear)
                momentum_curr_angular = orientation_curr * (link_inertia.moment * velocity_curr_angular)

                momentum_next_linear = orientation_curr * (link_inertia.mass * velocity_next_linear)
                momentum_next_angular = orientation_curr * (link_inertia.moment * velocity_next_angular)

                center = pose_curr * link_inertia.center

                force = (momentum_next_linear - momentum_curr_linear) * tt.Scalar(-1.0 / self.time_step)
                torque = (momentum_next_angular - momentum_curr_angular) * tt.Scalar(-1.0 / self.time_step)

                #force = tt.Vector3.zero
                #torque = tt.Vector3.zero

                force += gravity * link_inertia.mass

                inertial_forces += force
                inertial_torques += torque - tr.cross(ref_point - center, force)

            contact_forces = tt.Vector3.zero
            contact_torques = tt.Vector3.zero

            for contact in self.contacts[time_frame]:

                contact_point = reconstruction.link_pose(time_frame, contact.target_link.name) * contact.point
                contact_force = contact.force_variable

                contact_forces += contact_force
                contact_torques += tr.cross(contact_point - ref_point, contact_force)

            w = 1.0
            if time_frame == self.frame_count - 1:
                w = end_weight

            if force_consistency > 0: tr.goal((inertial_forces + contact_forces) * tt.Scalar(w * force_consistency) * self.weight_parameter)
            if torque_consistency > 0: tr.goal((inertial_torques + contact_torques) * tt.Scalar(w * torque_consistency) * self.weight_parameter)


        # Create rigid body objects to simulate dynamics
        bodies = { }
        for time_frame in range(self.frame_min, self.frame_max, stride):
            bodies[time_frame] = { }
            for link_model in self.body_links:
                current_pose = reconstruction.link_pose(time_frame, link_model.name)
                current_velocity = reconstruction.link_velocity(time_frame, link_model.name)
                bodies[time_frame][link_model] = tt.RigidBody(current_pose, link_model.inertia, current_velocity);


        # We unroll short trajectories using rigid body physics and then
        # minimize deviations from our motion reconstruction
        for dstep in range(steps):

            # Apply joint forces and joint torques
            for time_frame in self.frame_range:
                for joint_wrench in self.joint_wrenches[time_frame]:
                    wrench = joint_wrench.wrench_variable
                    wrench = tt.Twist(tr.translation(wrench) * tt.Scalar(joint_force_scaling), tr.rotation(wrench) * tt.Scalar(joint_torque_scaling))
                    body_time_frame = max(self.frame_min, joint_wrench.time_frame - dstep)
                    if body_time_frame in bodies:
                        parent_body = bodies[body_time_frame][joint_wrench.parent_link]
                        child_body = bodies[body_time_frame][joint_wrench.child_link]
                        # reference_pose = tt.Pose(child_body.position, tt.Orientation.identity)
                        # parent_body.apply_wrench(reference_pose, wrench * tt.Scalar(+1))
                        # child_body.apply_wrench(reference_pose, wrench * tt.Scalar(-1))
                        pos1 = child_body.position
                        pos2 = parent_body.pose * (tr.inverse(default_pose.link_pose(joint_wrench.parent_link.name)) * tr.translation(default_pose.link_pose(joint_wrench.child_link.name)))
                        reference_pose = tt.Pose((pos1 + pos2) * tt.Scalar(0.5), tt.Orientation.identity)
                        child_body.apply_wrench(reference_pose, wrench * tt.Scalar(-1))
                        parent_body.apply_wrench(reference_pose, wrench * tt.Scalar(+1))
                        # reference_pose = child_body.pose
                        # child_body.apply_wrench(reference_pose, wrench * tt.Scalar(-1))
                        # reference_pose = parent_body.pose * tr.inverse(default_pose.link_pose(joint_wrench.parent_link.name)) * default_pose.link_pose(joint_wrench.child_link.name)
                        # parent_body.apply_wrench(reference_pose, wrench * tt.Scalar(+1))

            # Apply contact forces
            for time_frame in self.frame_range:
                for contact in self.contacts[time_frame]:
                    body_time_frame = max(self.frame_min, contact.time_frame - dstep)
                    if body_time_frame in bodies:
                        body = bodies[body_time_frame][contact.body_link]
                        contact_point = body.pose * (tr.inverse(default_pose.link_pose(contact.body_link.name)) * default_pose.link_pose(contact.target_link.name) * contact.point)
                        contact_force = contact.force_variable
                        body.apply_force(contact_point, contact_force * tt.Scalar(contact_force_scaling))

            # Apply gravity and simulate rigid body dynamics
            for body_map in bodies.values():
                for body in body_map.values():

                    # Apply gravity
                    body.apply_acceleration(gravity)

                    # Simulate rigid body dynamics
                    for i in range(substeps):
                        body.integrate(self.time_step / substeps)

            # Try to minimize deviations between motion reconstructions and
            # rigid body dynamics
            for time_frame in self.frame_range:
                if time_frame in bodies:
                    for link_model in self.body_links:
                        body = bodies[time_frame][link_model]
                        next_time_frame = time_frame + 1 + dstep
                        if next_time_frame in self.frame_range:

                            # Match Cartesian positions and orientations
                            if position_cost > 0: tr.goal((body.position - reconstruction.link_position(next_time_frame, link_model.name)) * tt.Scalar(position_cost) * self.weight_parameter)
                            if orientation_cost > 0: tr.goal(tr.residual(tr.inverse(tr.orientation(reconstruction.link_pose(next_time_frame, link_model.name))) * body.orientation) * tt.Scalar(orientation_cost) * self.weight_parameter)

                            # Match linear and angular velocities
                            body_velocity = body.local_velocity
                            link_velocity = reconstruction.link_velocity(next_time_frame, link_model.name)
                            if linear_velocity_consistency > 0: tr.goal((tr.translation(body_velocity) - tr.translation(link_velocity)) * tt.Scalar(linear_velocity_consistency) * self.weight_parameter)
                            if angular_velocity_consistency > 0: tr.goal((tr.rotation(body_velocity) - tr.rotation(link_velocity)) * tt.Scalar(angular_velocity_consistency) * self.weight_parameter)
