import os
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

from differentiable_robot_model import DifferentiableRobotModel


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
            ee_pred = self.forward(joint_state, ac)
            joint_state = (joint_state + ac).clone()
            qs.append(joint_state.clone())
            key_pos.append(ee_pred.clone())

        return torch.stack(qs), torch.stack(key_pos)


def generate_demo_traj(rest_pose, goal_ee1, goal_ee2, policy):
    joint_state = rest_pose
    for epoch in range(100):
        qs, key_pos = policy.roll_out(joint_state.clone())
        loss = ((key_pos[1:12, -3:] - torch.Tensor(goal_ee1)) ** 2).mean(dim=0) + (
                    (key_pos[12:, -3:] - torch.Tensor(goal_ee2)) ** 2).mean(dim=0)
        loss = loss.mean() + (policy.action ** 2).mean()
        policy.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        policy.optimizer.step()
    print('keypoint', key_pos[11, -3:])
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
    for q in qs:
        for i in range(7):
            sim.resetJointState(bodyUniqueId=robot_id,
                                jointIndex=i,
                                targetValue=q[i],
                                targetVelocity=0)
        sim.stepSimulation()

        time.sleep(0.2)


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
    cur_ee = dmodel.forward_kin(rest_pose)

    data_type = 'reaching' #'placing'

    generate_new_data = False

    if not os.path.exists(f'traj_data'):
        os.makedirs('traj_data')

    if generate_new_data or not os.path.exists(f'traj_data/traj_data_{data_type}.pkl'):
        print('cur_ee', cur_ee[-3:])
        trajectories = []
        for _ in range(6):
            print(_)
            traj_data = {}
            policy = ActionNetwork(dmodel)
            goal_ee1 = cur_ee[-3:].clone()
            if data_type == 'reaching':
                multiplier = 2
            else:
                multiplier = 1
            goal_ee1[0] = goal_ee1[0] + multiplier * torch.Tensor(np.random.uniform(-0.3, 0.3, 1))
            goal_ee2 = goal_ee1.clone()
            goal_ee2[2] = goal_ee2[2] + multiplier * torch.Tensor(np.random.uniform(-0.4, 0.0, 1))
            if data_type == 'reaching':
                chosen_goal = np.random.choice([1, 2])
                if chosen_goal == 2:
                    goal_ee1 = goal_ee2.clone()
                else:
                    goal_ee2 = goal_ee1.clone()
            qs, keypoints, actions = generate_demo_traj(rest_pose, goal_ee1, goal_ee2, policy)
            traj_data['q'] = qs
            traj_data['keypoints'] = keypoints
            traj_data['actions'] = actions
            trajectories.append(traj_data)

        with open(f'traj_data/traj_data_{data_type}.pkl', "wb") as fp:
            pickle.dump(trajectories, fp, protocol=pickle.HIGHEST_PROTOCOL)

    # visualization - matplotlib

    with open(f'traj_data/traj_data_{data_type}.pkl', 'rb') as f:
        trajs = pickle.load(f)

    n_trajs = len(trajs)
    n_trajs_sqrt = int(np.sqrt(n_trajs))
    if n_trajs_sqrt ** 2 < n_trajs:
        n_trajs_sqrt += 1

    fig = plt.figure(figsize=(n_trajs_sqrt ** 2, n_trajs_sqrt ** 2))
    for i, traj in enumerate(trajs):
        ax = fig.add_subplot(n_trajs_sqrt, n_trajs_sqrt, i + 1, projection='3d')
        ax.plot(trajs[i]['keypoints'][1:, 0, 0], trajs[i]['keypoints'][1:, 0, 1], trajs[i]['keypoints'][1:, 0, 2])
        ax.scatter(trajs[i]['keypoints'][1:, 0, 0], trajs[i]['keypoints'][1:, 0, 1], trajs[i]['keypoints'][1:, 0, 2],
                   color='blue')
        ax.scatter(trajs[i]['keypoints'][1, 0, 0], trajs[i]['keypoints'][1, 0, 1], trajs[i]['keypoints'][1, 0, 2],
                   color='red')
        ax.scatter(trajs[i]['keypoints'][-1, 0, 0], trajs[i]['keypoints'][-1, 0, 1], trajs[i]['keypoints'][-1, 0, 2],
                   color='green')
        range_x = trajs[i]['keypoints'][1:, 0, 0].max() - trajs[i]['keypoints'][1:, 0, 0].min()
        range_y = trajs[i]['keypoints'][1:, 0, 1].max() - trajs[i]['keypoints'][1:, 0, 1].min()
        range_z = trajs[i]['keypoints'][1:, 0, 2].max() - trajs[i]['keypoints'][1:, 0, 2].min()
        range = max(range_x, range_y, range_z)
        ax.set_xlim([trajs[i]['keypoints'][1:, 0, 0].min(), trajs[i]['keypoints'][1:, 0, 0].min() + range])
        ax.set_ylim([trajs[i]['keypoints'][1:, 0, 1].min(), trajs[i]['keypoints'][1:, 0, 1].min() + range])
        ax.set_zlim([trajs[i]['keypoints'][1:, 0, 2].min(), trajs[i]['keypoints'][1:, 0, 2].min() + range])
        ax.set_title(f"Trajectory {i}")

    plt.savefig(f'traj_data/traj_data_{data_type}.png')
