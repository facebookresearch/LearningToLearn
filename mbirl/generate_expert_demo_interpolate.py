import os, sys
import random
import torch
import numpy as np
import hydra
import dill as pickle
import pybullet_utils.bullet_client as bc
import pybullet_data
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
        for link in [1, 2, 3]:
            kp_pos, kp_rot = self.robot_model.compute_forward_kinematics(xdesired, 'kp_link_' + str(link))
            keypoints += kp_pos

        return torch.stack(keypoints).squeeze()

    def forward_kin(self, x):
        keypoints = []
        for link in [1, 2, 3]:
            kp_pos, kp_rot = self.robot_model.compute_forward_kinematics(x, 'kp_link_' + str(link))
            keypoints += kp_pos

        return torch.stack(keypoints).squeeze()


class ActionNetwork(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.action = torch.nn.Parameter(torch.Tensor(np.zeros([25, 7])))
        # torch.nn.Module.register_paramete(self.action)
        self.model = model
        self.optimizer = torch.optim.SGD(self.parameters(), lr=1e-1)
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, joint_state, ac):
        ee_pred = self.model(joint_state, ac)
        return ee_pred

    def roll_out(self, joint_state):
        qs = []
        key_pos = []
        qs.append(joint_state)
        key_pos.append(self.model.forward_kin(joint_state))
        for t in range(25):
            ac = self.action[t]
            ee_pred = self.forward(joint_state.detach(), ac)
            joint_state = (joint_state.detach() + ac).clone()
            qs.append(joint_state.clone())
            key_pos.append(ee_pred.clone())

        return torch.stack(qs), torch.stack(key_pos)


def generate_demo_traj(rest_pose, goal_ee, policy):
    joint_state = rest_pose
    for epoch in range(100):
        qs, key_pos = policy.roll_out(joint_state.clone())
        loss = ((key_pos[1:, -3:] - torch.Tensor(goal_ee)) ** 2).mean(dim=0)
        loss = loss.mean() + 0.5*(policy.action ** 2).mean()
        policy.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        policy.optimizer.step()
    print('keypoint', key_pos[12, -3:])
    print('goal1', goal_ee1)
    print('keypoint', key_pos[-1, -3:])
    print('goal2', goal_ee2)

    # collect trajectory info
    qs, key_pos = policy.roll_out(joint_state.clone())
    print('roll_out', key_pos[-1, -3:])
    return qs.detach().numpy(), key_pos.detach().numpy(), policy.action.detach().numpy()


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
    policy = ActionNetwork(dmodel)

    rest_pose = [0.0, 0.0, 0.0, 1.57079633, 0.0, 1.03672558, 0.0]
    rest_pose = torch.Tensor(rest_pose).unsqueeze(dim=0)

    # update kinematic state
    _ = dmodel.forward_kin(rest_pose)

    cur_ee = dmodel.forward_kin(rest_pose)
    print(f"cur_ee: {cur_ee}")

    keypt_centroid = cur_ee.mean(dim=0)
    print(f"keypt_centroid: {keypt_centroid}")

    # data_type = 'reaching'
    data_type = 'placing'

    generate_new_data = False

    if not os.path.exists(traj_data_dir):
        os.makedirs(traj_data_dir)

    if generate_new_data or not os.path.exists(f'{traj_data_dir}/traj_data_{data_type}.pkl'):
        trajectories = []
        for traj_it in range(6):
            print(traj_it)
            traj_data = {}
            policy = ActionNetwork(dmodel)
            goal_ee1 = cur_ee[-3:].clone()
            goal_ee1[:, 0] = goal_ee1[:, 0] + torch.Tensor(np.random.uniform(-0.4, -0.3, 1))
            goal_ee2 = goal_ee1.clone()
            goal_ee2[:, 2] = goal_ee2[:, 2] + torch.Tensor(np.random.uniform(-0.5, -0.4, 1))

            if data_type == 'reaching':
                goal_ee_list = torch.stack([cur_ee.clone() for i in range(25)])
            else:
                goal_ee_list = torch.stack([cur_ee.clone() for i in range(12)] + [goal_ee1.clone() for i in range(13)])
            for i in range(3):
                if data_type == 'reaching':
                    goal_ee_list[:, i, 0] = torch.linspace(cur_ee[i, 0], goal_ee1[i, 0], 25)
                else:
                    goal_ee_list[:12, i, 0] = torch.linspace(cur_ee[i, 0], goal_ee1[i, 0], 12)
                    goal_ee_list[12:, i, 2] = torch.linspace(goal_ee1[i, 2], goal_ee2[i, 2], 13)

            # print(goal_ee_list)
            # show_goal_trajectory(goal_ee_list, data_type, save=True)

            qs, keypoints, actions = generate_demo_traj(rest_pose, goal_ee_list, policy)
            traj_data['q'] = qs
            traj_data['keypoints'] = keypoints
            traj_data['actions'] = actions
            traj_data['desired'] = goal_ee_list
            trajectories.append(traj_data)

        with open(f'{traj_data_dir}/traj_data_{data_type}.pkl', "wb") as fp:
            pickle.dump(trajectories, fp, protocol=pickle.HIGHEST_PROTOCOL)

    # visualization - matplotlib

    with open(f'{traj_data_dir}/traj_data_{data_type}.pkl', 'rb') as f:
        trajs = pickle.load(f)

    n_trajs = len(trajs)

    fig = plt.figure(figsize=(2 * 5, np.ceil(n_trajs/2) * 5))
    for i, traj in enumerate(trajs):
        ax = fig.add_subplot(2, np.ceil(n_trajs/2), i + 1, projection='3d')
        ax.plot(trajs[i]['keypoints'][1:, 0, 0], trajs[i]['keypoints'][1:, 0, 1], trajs[i]['keypoints'][1:, 0, 2])
        ax.scatter(trajs[i]['keypoints'][1:, 0, 0], trajs[i]['keypoints'][1:, 0, 1], trajs[i]['keypoints'][1:, 0, 2],
                   color='blue')
        ax.plot(trajs[i]['desired'][:, 0, 0], trajs[i]['desired'][:, 0, 1], trajs[i]['desired'][:, 0, 2], color='orange')
        ax.scatter(trajs[i]['desired'][:, 0, 0], trajs[i]['desired'][:, 0, 1], trajs[i]['desired'][:, 0, 2],
                   color='orange')
        ax.scatter(trajs[i]['keypoints'][1, 0, 0], trajs[i]['keypoints'][1, 0, 1], trajs[i]['keypoints'][1, 0, 2],
                   color='red')
        ax.scatter(trajs[i]['keypoints'][-1, 0, 0], trajs[i]['keypoints'][-1, 0, 1], trajs[i]['keypoints'][-1, 0, 2],
                   color='green')
        max_x = max(trajs[i]['keypoints'][1:, 0, 0].max(), trajs[i]['desired'][:, 0, 0].max())
        min_x = min(trajs[i]['keypoints'][1:, 0, 0].min(), trajs[i]['desired'][:, 0, 0].min())
        max_y = max(trajs[i]['keypoints'][1:, 0, 1].max(), trajs[i]['desired'][:, 0, 1].max())
        min_y = min(trajs[i]['keypoints'][1:, 0, 1].min(), trajs[i]['desired'][:, 0, 1].min())
        max_z = max(trajs[i]['keypoints'][1:, 0, 2].max(), trajs[i]['desired'][:, 0, 2].max())
        min_z = min(trajs[i]['keypoints'][1:, 0, 2].min(), trajs[i]['desired'][:, 0, 2].min())
        range_x = max_x - min_x
        range_y = max_y - min_y
        range_z = max_z - min_z
        max_range = max(range_x, range_y, range_z)
        ax.set_xlim([min_x, min_x + max_range])
        ax.set_ylim([min_y, min_y + max_range])
        ax.set_zlim([min_z, min_z + max_range])
        ax.set_title(f"Trajectory {i}")

    plt.tight_layout()
    plt.savefig(f'{traj_data_dir}/traj_data_{data_type}.png')
    plt.show()

    # pybullet visualization

    rel_urdf_path = 'env/kuka_iiwa/urdf/iiwa7_ft_with_obj_keypts.urdf'
    urdf_path = os.path.join(mbirl.__path__[0], rel_urdf_path)
    robot_model = DifferentiableRobotModel(urdf_path=urdf_path, name="kuka_w_obj_keypts")

    sim = bc.BulletClient(connection_mode=pybullet.GUI)
    robot_id = sim.loadURDF(urdf_path, basePosition=[0, 0, 0], useFixedBase=True,
                            flags=pybullet.URDF_USE_INERTIA_FROM_FILE)

    sim.setGravity(0, 0, -9.81)

    sim.setRealTimeSimulation(0)

    # for testing purposes we set joint damping to zero, because in pybullet the forward dynamics (used for simulation)
    # does use joint damping, but the inverse dynamics call does not use joint damping - which makes it hard to test both
    # with the same robot model if joint damping is not zero
    for link_idx in range(8):
        sim.changeDynamics(robot_id, link_idx, linearDamping=0.0, angularDamping=0.0, jointDamping=0.0)
        sim.changeDynamics(robot_id, link_idx, maxJointVelocity=200)

    # for traj_data in trajs:
    #     visualize_traj(traj_data, robot_id, sim)
