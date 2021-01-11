# Copyright (c) Facebook, Inc. and its affiliates.
import os
import random
import torch
import numpy as np
import mbirl
import matplotlib.pyplot as plt

from differentiable_robot_model import DifferentiableRobotModel

EXP_FOLDER = os.path.join(mbirl.__path__[0], "experiments")
traj_data_dir = os.path.join(EXP_FOLDER, 'traj_data')


class GroundTruthForwardModel(torch.nn.Module):
    def __init__(self, model):
        super(GroundTruthForwardModel, self).__init__()
        self.robot_model = model

    def forward_kin(self, x):
        keypoints = []
        for link in [1, 2]:  # , 3]:
            kp_pos, kp_rot = self.robot_model.compute_forward_kinematics(x, 'kp_link_' + str(link))
            keypoints += 100.0*kp_pos

        return torch.stack(keypoints).squeeze()


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

    experiment_type = 'placing'

    regenerate_data = True

    if not os.path.exists(traj_data_dir):
        os.makedirs(traj_data_dir)

    joint_limits = [2.967, 2.094, 2.967, 2.094, 2.967, 2.094, 3.054]
    if regenerate_data or not os.path.exists(f'{traj_data_dir}/traj_data_{experiment_type}.pt'):
        trajectories = []
        for traj_it in range(6):
            print(traj_it)
            traj_data = {}
            start_pose = rest_pose.clone()
            start_keypts = dmodel.forward_kin(start_pose)
            print(f"cur keypts: {start_keypts}")
            goal_keypts1 = start_keypts[-3:].clone()
            goal_keypts1[:, 0] = goal_keypts1[:, 0] + torch.Tensor([-20.0]) + torch.randn(1)[0]
            goal_keypts2 = goal_keypts1.clone()
            goal_keypts2[:, 2] = goal_keypts2[:, 2] + torch.Tensor([-30.0]) + torch.randn(1)[0]

            desired_keypt_traj = torch.stack([start_keypts.clone() for i in range(5)] + [goal_keypts1.clone() for i in range(5)])

            for kp_idx in range(2):
                desired_keypt_traj[:5, kp_idx, 0] = torch.linspace(start_keypts[kp_idx, 0], goal_keypts1[kp_idx, 0], 5)
                desired_keypt_traj[5:, kp_idx, 2] = torch.linspace(goal_keypts1[kp_idx, 2], goal_keypts2[kp_idx, 2], 5)

            traj_data['start_joint_config'] = start_pose
            traj_data['desired_keypoints'] = desired_keypt_traj
            trajectories.append(traj_data)

        torch.save(trajectories, f"{traj_data_dir}/traj_data_{experiment_type}.pt")

    # visualization - matplotlib
    trajs = torch.load(f"{traj_data_dir}/traj_data_{experiment_type}.pt")

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
    plt.savefig(f'{traj_data_dir}/traj_data_{experiment_type}.png')
    plt.show()

