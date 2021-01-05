import os
import random
import numpy as np
import torch
import dill as pickle
from differentiable_robot_model import DifferentiableRobotModel

from mbirl.learnable_costs import LearnableWeightedCost, LearnableTimeDepWeightedCost
from mbirl.keypoint_mpc import KeypointMPCWrapper
import logging; logging.disable(logging.CRITICAL);

import mbirl

import cvxpy as cp

##Code adapted from:  https://github.com/reinforcement-learning-kr/lets-do-irl


class QPoptimizer(object):
    def __call__(self, feature_num, learner, expert):
        w = cp.Variable(feature_num)
        obj_func = cp.Minimize(cp.norm(w))

        constraints = [(expert.detach().numpy()-learner.detach().numpy()) @ w >= 1]

        prob = cp.Problem(obj_func, constraints)
        prob.solve()

        if prob.status == "optimal":
            weights = np.squeeze(np.asarray(w.value))
            return weights, prob.value
        else:
            weights = np.zeros(feature_num)
            return weights, prob.status

class IRLLoss(object):
    def __call__(self, pred_traj, target_traj):
        loss = ((pred_traj[:,-9:] - target_traj[:,-9:])**2).sum(dim=0)
        return loss.mean()


def extract_feature_expectations(keypoint_trajectory, goal_keypts):
    T, num_feat = keypoint_trajectory.shape
    exp_feat = torch.zeros(num_feat)
    gamma = 0.9

    # we use the keypoints distance to goal keypots as features
    # this is analogous to our learnable cost function
    for t in range(T):
        exp_feat += gamma**t * (keypoint_trajectory[t, :] - goal_keypts)**2

    return exp_feat


def irl_training(learnable_cost, robot_model, irl_loss_fn, expert_demo, n_outer_iter, n_inner_iter):

    irl_losses = []

    cost_optimizer = QPoptimizer()
    keypoint_mpc_wrapper = KeypointMPCWrapper(robot_model)

    start_joint_state = expert_demo[0, :7].clone()
    goal_keypts = expert_demo[-1, -9:].clone()

    expert_features = extract_feature_expectations(expert_demo[:, -9:], goal_keypts)

    # unroll and extract expected features
    pred_traj = keypoint_mpc_wrapper.roll_out(start_joint_state.clone())
    phi = extract_feature_expectations(pred_traj[:, -9:], goal_keypts)

    # get initial irl loss
    irl_losses.append(irl_loss_fn(pred_traj, expert_demo).item())
    print("irl loss start, loss: {}".format(irl_losses[-1]))

    # get initial weights
    W, _ = cost_optimizer(len(expert_features), expert_features, phi)

    for outer_iter in range(n_outer_iter):

        keypoint_mpc_wrapper = KeypointMPCWrapper(robot_model)
        action_optimizer = torch.optim.SGD(keypoint_mpc_wrapper.parameters(), lr=0.001)

        weight = torch.Tensor(W)

        for idx in range(10):

            # unroll and extract expected features
            pred_traj = keypoint_mpc_wrapper.roll_out(start_joint_state.clone())
            phi = extract_feature_expectations(pred_traj[:, -9:], goal_keypts)

            # reset gradients
            action_optimizer.zero_grad()

            # update the actions, given the current weights
            cost = (phi*weight).sum()
            cost.backward(retain_graph=True)
            action_optimizer.step()

        irl_losses.append(irl_loss_fn(pred_traj, expert_demo).item())
        print("irl loss outer iter: {} loss: {}".format(outer_iter, irl_losses[-1]))

        # compute optimal weights with convex optimization (Abbeel et al.)
        W, _ = cost_optimizer(len(expert_features), expert_features, phi)
        hyper_distance = np.abs(np.dot(W, expert_features.detach()-phi.detach())) #hyperdistance = t
        if hyper_distance <= params['epsilon']: # terminate if the point reached close enough
            break


if __name__ == '__main__':
    random.seed(10)
    np.random.seed(10)
    torch.manual_seed(0)
    rest_pose = [0.0, 0.0, 0.0, 1.57079633, 0.0, 1.03672558, 0.0]
    joint_limits = [2.967,2.094,2.967,2.094,2.967,2.094,3.054]

    rel_urdf_path = 'env/kuka_iiwa/urdf/iiwa7_ft_with_obj_keypts.urdf'
    urdf_path = os.path.join(mbirl.__path__[0], rel_urdf_path)
    robot_model = DifferentiableRobotModel(urdf_path=urdf_path, name="kuka_w_obj_keypts")

    params = {
        'gamma': 0.9,
        'epsilon': 0.001,
        'n_inner_iter': 1,
        'time_horizon': 25
    }

    data_type = 'reaching'  # 'placing'
    with open(f'traj_data/traj_data_{data_type}.pkl', 'rb') as f:
        trajs = pickle.load(f)
    if data_type == 'reaching':
        traj = trajs[4]
    else:
        traj = trajs[0]

    traj_len = len(traj['q'])

    expert_demo = np.concatenate([traj['q'].reshape(traj_len, -1), traj['keypoints'].reshape(traj_len, -1)], axis=-1)
    expert_demo = torch.Tensor(expert_demo)
    print(expert_demo.shape)

    no_demos = 1

    # this loss measures the distance between the expert demo and generated trajectory, we use this loss only to
    # compare progress to our algorithm
    irl_loss_fn = IRLLoss()

    learnable_cost = LearnableWeightedCost()

    irl_training(learnable_cost, robot_model, irl_loss_fn, expert_demo, n_outer_iter=200, n_inner_iter=1)


