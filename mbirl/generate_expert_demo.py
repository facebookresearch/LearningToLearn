import os, sys
import random
import torch
import numpy as np
import dill as pickle
import pybullet_utils.bullet_client as bc
import pybullet
import time
import mbirl
import matplotlib.pyplot as plt
from os.path import dirname, abspath

from differentiable_robot_model import DifferentiableRobotModel

_ROOT_DIR = dirname(abspath(__file__))
sys.path.append(_ROOT_DIR)

traj_data_dir = os.path.join(_ROOT_DIR, 'traj_data')


class GroundTruthForwardModel(torch.nn.Module):
    def __init__(self, model):
        super(GroundTruthForwardModel, self).__init__()
        self.robot_model = model

    # x=current joint state, u=change in joint state
    def forward(self, x, u):
        xdesired = x + u

        keypoints = []
        for link in [1, 2]:#, 3]:
            kp_pos, kp_rot = self.robot_model.compute_forward_kinematics(xdesired, 'kp_link_' + str(link))
            keypoints += 100.0*kp_pos

        return torch.stack(keypoints).squeeze()

    def forward_kin(self, x):
        keypoints = []
        for link in [1, 2]:  # , 3]:
            kp_pos, kp_rot = self.robot_model.compute_forward_kinematics(x, 'kp_link_' + str(link))
            keypoints += 100.0*kp_pos

        return torch.stack(keypoints).squeeze()


def visualize_traj(traj_data, robot_id, sim):
    qs = traj_data['q'].squeeze()
    print(qs.shape)
    for j, q in enumerate(qs):
        for i in range(7):
            sim.resetJointState(bodyUniqueId=robot_id,
                                jointIndex=i,
                                targetValue=q[i],
                                targetVelocity=0)
        if j == 0:
            print(q)
            cur_ee = dmodel.forward_kin(torch.Tensor(q).unsqueeze(dim=0))
            print(cur_ee)
            time.sleep(0.2)
        sim.stepSimulation()

        time.sleep(1.0)


def show_goal_trajectory(goal_ee_list, data_type, save=True):
    fig = plt.figure(figsize=(10, 30))
    for i in range(3):
        ax = fig.add_subplot(1, 3, i + 1, projection='3d')
        ax.plot(goal_ee_list[:, i, 0], goal_ee_list[:, i, 1], goal_ee_list[:, i, 2])
        ax.scatter(goal_ee_list[:, i, 0], goal_ee_list[:, i, 1], goal_ee_list[:, i, 2],
                   color='blue')
        ax.scatter(goal_ee_list[0, i, 0], goal_ee_list[0, i, 1], goal_ee_list[0, i, 2],
                   color='red')
        ax.scatter(goal_ee_list[-1, i, 0], goal_ee_list[-1, i, 1], goal_ee_list[-1, i, 2],
                   color='green')
        range_x = goal_ee_list[:, i, 0].max() - goal_ee_list[:, i, 0].min()
        range_y = goal_ee_list[:, i, 1].max() - goal_ee_list[:, i, 1].min()
        range_z = goal_ee_list[:, i, 2].max() - goal_ee_list[:, i, 2].min()
        max_range = max(range_x, range_y, range_z)
        ax.set_xlim([goal_ee_list[:, i, 0].min(), goal_ee_list[:, i, 0].min() + max_range])
        ax.set_ylim([goal_ee_list[:, i, 1].min(), goal_ee_list[:, i, 1].min() + max_range])
        ax.set_zlim([goal_ee_list[:, i, 2].min(), goal_ee_list[:, i, 2].min() + max_range])
        ax.set_title(f"Trajectory {i}")

    plt.tight_layout()
    if save:
        plt.savefig(f"{traj_data_dir}/traj_goal_{data_type}.png")
    else:
        plt.show()


if __name__ == '__main__':

    random.seed(10)
    np.random.seed(10)
    torch.manual_seed(0)
    curr_dir = os.path.dirname(__file__)

    rel_urdf_path = 'env/kuka_iiwa/urdf/iiwa7_ft_with_obj_keypts.urdf'
    urdf_path = os.path.join(mbirl.__path__[0], rel_urdf_path)
    robot_model = DifferentiableRobotModel(urdf_path=urdf_path, name="kuka_w_obj_keypts")

    dmodel = GroundTruthForwardModel(robot_model)

    rest_pose = [0.0, 0.0, 0.0, 1.57079633, 0.0, 1.03672558, 0.0]
    rest_pose = torch.Tensor(rest_pose).unsqueeze(dim=0)

    data_type = 'placing'
    # data_type = 'reaching'

    regenerate_data = True

    if not os.path.exists(traj_data_dir):
        os.makedirs(traj_data_dir)

    joint_limits = [2.967, 2.094, 2.967, 2.094, 2.967, 2.094, 3.054]
    if regenerate_data or not os.path.exists(f'{traj_data_dir}/traj_data_{data_type}.pkl'):
        trajectories = []
        for traj_it in range(6):
            print(traj_it)
            traj_data = {}
            start_pose = rest_pose.clone()
            # start_pose[0, 3] += torch.randn(1)[0]*0.03
            # if torch.abs(start_pose[0, 3]) > joint_limits[3]:
            #     start_pose[0, 3] = torch.Tensor(joint_limits[3])
            start_keypts = dmodel.forward_kin(start_pose)
            print(f"cur keypts: {start_keypts}")
            goal_keypts1 = start_keypts[-3:].clone()
            goal_keypts1[:, 0] = goal_keypts1[:, 0] + torch.Tensor([-20.0]) + torch.randn(1)[0]#torch.Tensor(np.random.uniform(-0.25, -0.15, 1)*100.0)
            goal_keypts2 = goal_keypts1.clone()
            goal_keypts2[:, 2] = goal_keypts2[:, 2] + torch.Tensor([-30.0]) + torch.randn(1)[0]#torch.Tensor(np.random.uniform(-0.3, -0.2, 1)*100.0)

            if data_type == 'reaching':
                goal_ee_list = torch.stack([start_keypts.clone() for i in range(10)])
            else:
                goal_ee_list = torch.stack([start_keypts.clone() for i in range(5)] + [goal_keypts1.clone() for i in range(5)])

            for kp_idx in range(2):
                if data_type == 'reaching':
                    goal_ee_list[:, kp_idx, 0] = torch.linspace(start_keypts[kp_idx, 0], goal_keypts1[kp_idx, 0], 10)
                else:
                    goal_ee_list[:5, kp_idx, 0] = torch.linspace(start_keypts[kp_idx, 0], goal_keypts1[kp_idx, 0], 5)
                    goal_ee_list[5:, kp_idx, 2] = torch.linspace(goal_keypts1[kp_idx, 2], goal_keypts2[kp_idx, 2], 5)

            traj_data['start_joint_config'] = start_pose
            traj_data['desired_keypoints'] = goal_ee_list
            trajectories.append(traj_data)

        with open(f'{traj_data_dir}/traj_data_{data_type}.pkl', "wb") as fp:
            pickle.dump(trajectories, fp, protocol=pickle.HIGHEST_PROTOCOL)

    # visualization - matplotlib

    with open(f'{traj_data_dir}/traj_data_{data_type}.pkl', 'rb') as f:
        trajs = pickle.load(f)

    n_trajs = len(trajs)

    fig = plt.figure(figsize=(2 * 5, int(np.ceil(n_trajs/2)) * 5))
    for i, traj in enumerate(trajs):
        ax = fig.add_subplot(2, int(np.ceil(n_trajs/2)), i + 1, projection='3d')
        ax.plot(trajs[i]['desired_keypoints'][:, 0, 0], trajs[i]['desired_keypoints'][:, 0, 1], trajs[i]['desired_keypoints'][:, 0, 2])
        ax.scatter(trajs[i]['desired_keypoints'][:, 0, 0], trajs[i]['desired_keypoints'][:, 0, 1], trajs[i]['desired_keypoints'][:, 0, 2],
                   color='blue')
        ax.scatter(start_keypts[0, 0], start_keypts[0, 1], start_keypts[0, 2],
                   color='red')
        ax.scatter(trajs[i]['desired_keypoints'][-1, 0, 0], trajs[i]['desired_keypoints'][-1, 0, 1], trajs[i]['desired_keypoints'][-1, 0, 2],
                   color='green')
        min_x = -100.0; max_x = -50.0
        min_y = 0.0; max_y = 30
        min_z = 50; max_z = 100
        ax.set_xlim([min_x, max_x])
        ax.set_ylim([min_y, max_y])
        ax.set_zlim([min_z, max_z])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_title(f"Trajectory {i}")

    plt.tight_layout()
    plt.savefig(f'{traj_data_dir}/traj_data_{data_type}.png')
    plt.show()

    # pybullet visualization
    pybullet_viz = False
    if pybullet_viz:

        rel_urdf_path = 'env/kuka_iiwa/urdf/iiwa7_ft_with_obj_keypts.urdf'
        urdf_path = os.path.join(mbirl.__path__[0], rel_urdf_path)
        robot_model = DifferentiableRobotModel(urdf_path=urdf_path, name="kuka_w_obj_keypts")

        sim = bc.BulletClient(connection_mode=pybullet.GUI)
        robot_id = sim.loadURDF(urdf_path, basePosition=[0, 0, 0], useFixedBase=True,
                                flags=pybullet.URDF_USE_INERTIA_FROM_FILE)

        sim.setGravity(0, 0, -9.81)

        sim.setRealTimeSimulation(0)

        for link_idx in range(8):
            sim.changeDynamics(robot_id, link_idx, linearDamping=0.0, angularDamping=0.0, jointDamping=0.0)
            sim.changeDynamics(robot_id, link_idx, maxJointVelocity=200)

        for traj_data in trajs:
            visualize_traj(traj_data, robot_id, sim)
