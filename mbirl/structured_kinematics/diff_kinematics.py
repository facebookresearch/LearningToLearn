import os
import torch
import math
import numpy as np

import structured_kinematics.utils as utils
from urdf_parser_py.urdf import URDF

robot_description_folder = './'

torch.set_default_tensor_type(torch.DoubleTensor)
is_measuring_computation_time = False


def x_rot(angle):
    if len(angle.shape) == 0:
        angle = angle.unsqueeze(0)
    angle = utils.convert_into_at_least_2d_pytorch_tensor(angle).squeeze(1)
    batch_size = angle.shape[0]
    R = torch.zeros((batch_size, 3, 3))
    R[:, 0, 0] = torch.ones(batch_size)
    R[:, 1, 1] = torch.cos(angle)
    R[:, 1, 2] = -torch.sin(angle)
    R[:, 2, 1] = torch.sin(angle)
    R[:, 2, 2] = torch.cos(angle)
    return R


def y_rot(angle):
    if len(angle.shape) == 0:
        angle = angle.unsqueeze(0)
    angle = utils.convert_into_at_least_2d_pytorch_tensor(angle).squeeze(1)
    batch_size = angle.shape[0]
    R = torch.zeros((batch_size, 3, 3))
    R[:, 0, 0] = torch.cos(angle)
    R[:, 0, 2] = torch.sin(angle)
    R[:, 1, 1] = torch.ones(batch_size)
    R[:, 2, 0] = -torch.sin(angle)
    R[:, 2, 2] = torch.cos(angle)
    return R


def z_rot(angle):
    if len(angle.shape) == 0:
        angle = angle.unsqueeze(0)
    angle = utils.convert_into_at_least_2d_pytorch_tensor(angle).squeeze(1)
    batch_size = angle.shape[0]
    R = torch.zeros((batch_size, 3, 3))
    R[:, 0, 0] = torch.cos(angle)
    R[:, 0, 1] = -torch.sin(angle)
    R[:, 1, 0] = torch.sin(angle)
    R[:, 1, 1] = torch.cos(angle)
    R[:, 2, 2] = torch.ones(batch_size)
    return R


def cross_product(vec3a, vec3b):
    vec3a = utils.convert_into_at_least_2d_pytorch_tensor(vec3a)
    vec3b = utils.convert_into_at_least_2d_pytorch_tensor(vec3b)
    skew_symm_mat_a = utils.vector3_to_skew_symm_matrix(vec3a)
    return (skew_symm_mat_a @ vec3b.unsqueeze(2)).squeeze(2)


class CoordinateTransform(object):
    def __init__(self, rot=None, trans=None, device='cpu'):

        self._device = device

        if rot is None:
            self._rot = torch.eye(3).to(self._device)
        else:
            self._rot = rot.to(self._device)
        if len(self._rot.shape) == 2:
            self._rot = self._rot.unsqueeze(0)

        if trans is None:
            self._trans = torch.zeros(3).to(self._device)
        else:
            self._trans = trans.to(self._device)
        if len(self._trans.shape) == 1:
            self._trans = self._trans.unsqueeze(0)

    def set_translation(self, t):
        self._trans = t.to(self._device)
        if len(self._trans.shape) == 1:
            self._trans = self._trans.unsqueeze(0)
        return

    def set_rotation(self, rot):
        self._rot = rot.to(self._device)
        if len(self._rot.shape) == 2:
            self._rot = self._rot.unsqueeze(0)
        return

    def rotation(self):
        return self._rot

    def translation(self):
        return self._trans

    def inverse(self):
        rot_transpose = self._rot.transpose(-2, -1)
        return CoordinateTransform(rot_transpose, -(rot_transpose @ self._trans.unsqueeze(2)).squeeze(2),
                                   device=self._device)

    def multiply_transform(self, coordinate_transform):
        new_rot = self._rot @ coordinate_transform.rotation()
        new_trans = (self._rot @ coordinate_transform.translation().unsqueeze(2)).squeeze(2) + self._trans
        return CoordinateTransform(new_rot, new_trans, device=self._device)

    def trans_cross_rot(self):
        return utils.vector3_to_skew_symm_matrix(self._trans) @ self._rot

    def get_quaternion(self):
        M = torch.zeros((4, 4))
        M[:3, :3] = self._rot
        M[:3, 3] = self._trans[:, 0]
        M[3, 3] = 1
        q = torch.empty((4, ))
        t = torch.trace(M)
        if t > M[3, 3]:
            q[3] = t
            q[2] = M[1, 0] - M[0, 1]
            q[1] = M[0, 2] - M[2, 0]
            q[0] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 0, 1, 2
            if M[1, 1] > M[0, 0]:
                i, j, k = 1, 2, 0
            if M[2, 2] > M[i, i]:
                i, j, k = 2, 0, 1
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
            #q = q[[3, 0, 1, 2]]
        q *= 0.5 / math.sqrt(t * M[3, 3])

        return q


class Joint(object):
    def __init__(self, device='cpu'):

        self._device = device

        self.pose = CoordinateTransform(device=self._device)

        # local velocities and accelerations (w.r.t. joint coordinate frame):
        # in spatial vector terminology: linear velocity v
        self._lin_vel = torch.zeros((1, 3)).to(self._device)
        # in spatial vector terminology: angular velocity w
        self._ang_vel = torch.zeros((1, 3)).to(self._device)
        # in spatial vector terminology: linear acceleration vd
        self._lin_acc = torch.zeros((1, 3)).to(self._device)
        # in spatial vector terminology: angular acceleration wd
        self._ang_acc = torch.zeros((1, 3)).to(self._device)

        self.set_limits(0.0, 0.0, 0.0, 0.0)

        # damping
        self.damping = 0.0

    def set_limits(self, effort, lower, upper, vel):
        self.max_effort = effort
        self.lower_limit = lower
        self.upper_limit = upper
        self.max_velocity = vel


class RevoluteJoint(Joint):
    def __init__(self, rot_angles, trans, device='cpu'):

        self._device = device

        super().__init__(device=self._device)
        self._q = 0.0
        self._qd = 0.0
        self._qdd = 0.0

        # keeping those around for debugging purposes
        self._roll = rot_angles[0]
        self._pitch = rot_angles[1]
        self._yaw = rot_angles[2]
        self._trans = trans

        # this one works for the kuka arm and for panda arm
        self._fixed_rotation = (z_rot(self._yaw) @ y_rot(self._pitch)) @ x_rot(self._roll)
        self.pose.set_translation(torch.reshape(trans, (1, 3)))

        self.update_state(torch.zeros(1, 1), torch.zeros(1, 1))
        self.update_acc(torch.zeros(1, 1))
        return

    def reset_fixed_rotation(self):
        self._fixed_rotation = (z_rot(self._yaw) @ y_rot(self._pitch)) @ x_rot(self._roll)
        self.pose.set_translation(torch.reshape(self._trans, (1, 3)))
    def update_state(self, q, qd):
        self._q = utils.convert_into_at_least_2d_pytorch_tensor(q).to(self._device)
        self._qd = utils.convert_into_at_least_2d_pytorch_tensor(qd).to(self._device)

        batch_size = self._q.shape[0]

        # local z axis (w.r.t. joint coordinate frame):
        self._z = self._qd.new_zeros((1, 3))
        self._z[:, 2] = 1.0

        self._ang_vel = self._qd @ self._z

        # when we update the joint angle, we also need to update the transformation
        self.pose.set_rotation(self._fixed_rotation.repeat(batch_size, 1, 1) @ z_rot(self._q))
        return

    def update_acc(self, qdd):
        self._qdd = utils.convert_into_at_least_2d_pytorch_tensor(qdd).to(self._device)

        # local z axis (w.r.t. joint coordinate frame):
        self._z = self._qd.new_zeros((1, 3))
        self._z[:, 2] = 1.0

        self._ang_acc = self._qdd @ self._z
        return


class RigidBody(torch.nn.Module):
    """
    Representation of the links.
    """
    def __init__(self, device='cpu'):

        super().__init__()

        self._device = device

        self.pose = CoordinateTransform(device=self._device)

        self._lin_vel = torch.zeros((1, 3)).to(self._device)
        self._ang_vel = torch.zeros((1, 3)).to(self._device)
        self._lin_acc = torch.zeros((1, 3)).to(self._device)
        self._ang_acc = torch.zeros((1, 3)).to(self._device)

        self.joint_id = -1
        self.name = ""
        self.joint = Joint(device=self._device)

        self.inertia = None

        # viscous friction (gsutanto: TODO: isn't this supposed to be a joint property?)
        self.vis_friction = 0.0
        # coloumb friction (gsutanto: TODO: isn't this supposed to be a joint property?)
        self.coul_friction = 0.0

        # in spatial vector terminology this is the "linear force f"
        self._lin_force = torch.zeros((1, 3)).to(self._device)
        # in spatial vector terminology this is the "couple n"
        self._ang_force = torch.zeros((1, 3)).to(self._device)

        return


class RobotModelTorch(torch.nn.Module):

    def __init__(self, rel_urdf_path, is_using_damping=False,device=None, gpu_name=None, name=''):

        super().__init__()

        self.name = name

        device = device
        self.gpu_name = gpu_name

        if device == "gpu" and torch.cuda.is_available():
            self._device = torch.device('cuda:0')
        else:
            self._device = 'cpu'

        urdf_path = os.path.join(robot_description_folder, rel_urdf_path)
        self._urdf_model = URDF.from_xml_file(urdf_path)

        self._bodies = torch.nn.ModuleList()
        self._n_dofs = 0
        self._controlled_joints = []
        self._joint_velocity_limits = []
        self._joint_effort_limits = []
        self._joint_upper_angle_limits = []
        self._joint_lower_angle_limits = []
        self._joint_damping = []

        # here we're making the joint a part of the rigid body
        # while urdfs model joints and rigid bodies separately
        # add initial body with joint.
        self._name_to_idx_map = dict()

        for (i, link) in enumerate(self._urdf_model.links):

            body = RigidBody(device=self._device)
            body.joint_id = i
            body.name = link.name

            urdf_jid = self._find_joint_of_body(body.name)
            urdf_joint_type = self._urdf_model.joints[urdf_jid].type

            if i == 0:
                rot_angles = torch.zeros(3).to(self._device)
                trans = torch.zeros(3).to(self._device)
                name = "base_joint"
                body.joint = RevoluteJoint(rot_angles, trans, device=self._device)
                body.rot_angles = rot_angles
                body.trans = trans
            else:
                jid = self._find_joint_of_body(body.name)
                joint = self._urdf_model.joints[jid]
                # find joint that is the "child" of this body according to urdf

                rpy = torch.tensor(joint.origin.rotation)
                rot_angles = torch.tensor([rpy[0], rpy[1], rpy[2]],requires_grad=True)
                trans = torch.tensor(torch.tensor(joint.origin.position))
                name = joint.name
                body.joint = RevoluteJoint(rot_angles, trans, device=self._device)
                # storing for debugging purposes
                body.rot_angles = rpy
                # storing for debugging purposes
                body.trans = trans
                if joint.type != "fixed":
                    self._n_dofs += 1
                    self._controlled_joints.append(i)
                    body.joint.set_limits(joint.limit.effort,
                                          joint.limit.lower, joint.limit.upper,
                                          joint.limit.velocity)
                    body.joint.damping = joint.dynamics.damping
                    self._joint_effort_limits.append(joint.limit.effort)
                    self._joint_velocity_limits.append(joint.limit.velocity)
                    self._joint_upper_angle_limits.append(joint.limit.upper)
                    self._joint_lower_angle_limits.append(joint.limit.lower)
                    self._joint_damping.append(joint.dynamics.damping)

            body.joint.name = name

            self._bodies.append(body)
            self._name_to_idx_map[body.name] = i

        self._is_using_damping = is_using_damping
        self._damping_net = None

    def _find_joint_of_body(self, body_name):
        for (i, joint) in enumerate(self._urdf_model.joints):
            if joint.child == body_name:
                return i
        return -1


    def update_kinematic_state(self, q, qd):
        """
        Updating the linear and angular positions and velocities of the links.
        :param joint_angles:
        :param joint_vel:
        :return:
        """
        q = utils.convert_into_at_least_2d_pytorch_tensor(q).to(self._device)
        qd = utils.convert_into_at_least_2d_pytorch_tensor(qd).to(self._device)
        batch_size = q.shape[0]

        # update the state of the joints
        for i in range(q.shape[1]):
            idx = self._controlled_joints[i]
            self._bodies[idx].joint.update_state(q[:, i].unsqueeze(1), qd[:, i].unsqueeze(1))

        base_lin_vel = torch.zeros((batch_size, 3), dtype=q.dtype).to(self._device)
        base_ang_vel = torch.zeros((batch_size, 3), dtype=q.dtype).to(self._device)
        # propagate the new joint state through the kinematic chain to update bodies position/velocities
        for i, body in enumerate(self._bodies):
            if i == 0:  # base
                body._lin_vel = base_lin_vel
                body._ang_vel = base_ang_vel
            else:
                body.pose = parent_body.pose.multiply_transform(body.joint.pose)

                inv_pose = body.joint.pose.inverse()

                new_ang_vel = (inv_pose.rotation() @ parent_body._ang_vel.unsqueeze(2)).squeeze(2)
                body._ang_vel = body.joint._ang_vel + new_ang_vel

                new_lin_vel = ((inv_pose.trans_cross_rot() @ parent_body._ang_vel.unsqueeze(2)).squeeze(2) +
                               (inv_pose.rotation() @ parent_body._lin_vel.unsqueeze(2)).squeeze(2))

                body._lin_vel = body.joint._lin_vel + new_lin_vel

            parent_body = body
        return

    def forward_kinematics(self, joint_angles, link_name):
        joint_vels = torch.zeros_like(joint_angles)
        self.update_kinematic_state(joint_angles, joint_vels)
        pose = self._bodies[self._name_to_idx_map[link_name]].pose
        pos = pose.translation()
        rot = pose.get_quaternion()
        return pos[0, :], rot

    def get_link_names(self):
        link_names = []
        for i in range(len(self._bodies)):
            link_names.append(self._bodies[i].name)
        return link_names