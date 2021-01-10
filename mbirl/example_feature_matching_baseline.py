import os, sys
import random
import numpy as np
import torch
import cvxpy as cp
import dill as pickle
from differentiable_robot_model import DifferentiableRobotModel
from os.path import dirname, abspath

_ROOT_DIR = dirname(abspath(__file__))
sys.path.append(_ROOT_DIR)

from mbirl.keypoint_mpc import KeypointMPCWrapper
import mbirl

traj_data_dir = os.path.join(_ROOT_DIR, 'traj_data')
model_data_dir = os.path.join(_ROOT_DIR, 'model_data')


class QPoptimizer(object):
    # Code adapted from:  https://github.com/reinforcement-learning-kr/lets-do-irl
    def __call__(self, feature_num, learner, expert):
        w = cp.Variable(feature_num)
        obj_func = cp.Minimize(cp.norm(w))
        constraints = [(expert.detach().numpy() - learner.detach().numpy()) @ w >= 1]

        prob = cp.Problem(obj_func, constraints)
        prob.solve()

        if prob.status == "optimal":
            weights = np.squeeze(np.asarray(w.value))
            norm = np.linalg.norm(weights)
            weights = weights / norm
            return weights, prob.value
        else:
            weights = np.zeros(feature_num)
            return weights, prob.status


class IRLLoss(object):
    def __call__(self, pred_traj, target_traj):
        loss = ((pred_traj[:, -6:] - target_traj[:, -6:]) ** 2).sum(dim=0)
        return loss.mean()


def learned_loss(pred_traj, target_traj, weights):
    assert pred_traj.dim() == 2
    mse = ((pred_traj[:, -6:] - target_traj[-6:]) ** 2).squeeze()

    # weighted mse
    wmse = mse * weights
    return wmse.mean()


def extract_feature_expectations(keypoint_trajectory, goal_keypts):
    T, num_feat = keypoint_trajectory.shape
    exp_feat = torch.zeros(num_feat)
    gamma = 0.9

    # we use the keypoints distance to goal keypots as features
    # this is analogous to our learnable cost function
    for t in range(T):
        exp_feat += gamma ** t * (keypoint_trajectory[t, :] - goal_keypts) ** 2

    return exp_feat


def evaluate_action_optimization(weights, robot_model, irl_loss_fn, trajs, n_inner_iter):
    # np.random.seed(cfg.random_seed)
    # torch.manual_seed(cfg.random_seed)

    eval_costs = []
    for i, traj in enumerate(trajs):

        traj_len = len(traj['desired_keypoints'])

        start_pose = traj['start_joint_config'].squeeze()
        expert_demo = traj['desired_keypoints'].reshape(traj_len, -1)
        expert_demo = torch.Tensor(expert_demo)
        time_horizon, n_keypt_dim = expert_demo.shape
        keypoint_mpc_wrapper = KeypointMPCWrapper(robot_model, time_horizon=time_horizon - 1, n_keypt_dim=n_keypt_dim)
        action_optimizer = torch.optim.SGD(keypoint_mpc_wrapper.parameters(), lr=0.001)

        for i in range(n_inner_iter):
            action_optimizer.zero_grad()

            pred_traj = keypoint_mpc_wrapper.roll_out(start_pose.clone())
            # use the learned loss to update the action sequence
            learned_cost_val = learned_loss(pred_traj, expert_demo[-1], weights)
            learned_cost_val.backward()
            action_optimizer.step()

        # Actually take the next step after optimizing the action
        pred_state_traj_new = keypoint_mpc_wrapper.roll_out(start_pose.clone())
        eval_costs.append(irl_loss_fn(pred_state_traj_new, expert_demo).mean())

    return torch.stack(eval_costs).detach()


def irl_training(robot_model, irl_loss_fn, expert_demo, start_joint_state, test_trajs, n_outer_iter, n_inner_iter):
    irl_cost_tr = []
    irl_cost_eval = []

    cost_optimizer = QPoptimizer()
    time_horizon, n_keypt_dim = expert_demo.shape
    keypoint_mpc_wrapper = KeypointMPCWrapper(robot_model, time_horizon=time_horizon - 1, n_keypt_dim=n_keypt_dim)
    action_optimizer = torch.optim.SGD(keypoint_mpc_wrapper.parameters(), lr=1.0)

    goal_keypts = expert_demo[-1, -n_keypt_dim:].clone()

    expert_features = extract_feature_expectations(expert_demo, goal_keypts)

    # unroll and extract expected features
    pred_traj = keypoint_mpc_wrapper.roll_out(start_joint_state.clone())
    phi = extract_feature_expectations(pred_traj[:, -n_keypt_dim:], goal_keypts)

    # get initial irl loss
    irl_cost_tr.append(irl_loss_fn(pred_traj, expert_demo).mean())
    print("irl loss outer iter: {} loss: {}".format(0, irl_cost_tr[-1]))

    # get initial weights
    W, _ = cost_optimizer(len(expert_features), expert_features, phi)

    for outer_iter in range(n_outer_iter):

        weight = torch.Tensor(W)
        # we reset the actions to optimize new action sequence from scratch with the new learned weights
        keypoint_mpc_wrapper.reset_actions()

        # we use the current "reward/cost" weight vector to optimize an action sequence
        for idx in range(n_inner_iter):
            # unroll and extract expected features
            pred_traj = keypoint_mpc_wrapper.roll_out(start_joint_state.clone())
            phi = extract_feature_expectations(pred_traj[:, -n_keypt_dim:], goal_keypts)

            # reset gradients
            action_optimizer.zero_grad()

            # update the actions, given the current weights
            cost = (phi * weight).sum()
            cost.backward(retain_graph=True)
            action_optimizer.step()

        # compute optimal weights with convex optimization (Abbeel et al.)
        W, _ = cost_optimizer(len(expert_features), expert_features, phi)

        # compute irl loss, solely for comparing to our method
        irl_cost_tr.append(irl_loss_fn(pred_traj, expert_demo).mean())
        print("irl loss outer iter: {} loss: {}".format(outer_iter + 1, irl_cost_tr[-1]))

        # check convergence
        hyper_distance = np.abs(np.dot(W, expert_features.detach() - phi.detach()))  # hyperdistance = t
        if hyper_distance <= 0.001:  # terminate if the point reached close enough
            break

        eval_costs = evaluate_action_optimization(torch.Tensor(W), robot_model, irl_loss_fn, test_trajs,
                                                  n_inner_iter)
        irl_cost_eval.append(eval_costs)

    return torch.stack(irl_cost_tr), torch.stack(irl_cost_eval), {'weights': torch.Tensor(W)}, pred_traj


if __name__ == '__main__':
    random.seed(10)
    np.random.seed(10)
    torch.manual_seed(0)

    rest_pose = [0.0, 0.0, 0.0, 1.57079633, 0.0, 1.03672558, 0.0]
    joint_limits = [2.967, 2.094, 2.967, 2.094, 2.967, 2.094, 3.054]

    rel_urdf_path = 'env/kuka_iiwa/urdf/iiwa7_ft_with_obj_keypts.urdf'
    urdf_path = os.path.join(mbirl.__path__[0], rel_urdf_path)
    robot_model = DifferentiableRobotModel(urdf_path=urdf_path, name="kuka_w_obj_keypts")

    # params = {
    #     'gamma': 0.9,
    #     'epsilon': 0.001,
    #     'n_inner_iter': 1,
    #     'time_horizon': 25
    # }

    # data_type = 'reaching'
    data_type = 'placing'
    with open(f'{traj_data_dir}/traj_data_{data_type}.pkl', 'rb') as f:
        trajs = pickle.load(f)

    traj = trajs[0]
    traj_len = len(traj['desired_keypoints'])

    start_q = traj['start_joint_config'].squeeze()
    expert_demo = traj['desired_keypoints'].reshape(traj_len, -1)
    expert_demo = torch.Tensor(expert_demo)
    print('expert demo shape', expert_demo.shape, traj['desired_keypoints'].shape)

    no_demos = 1

    # this loss measures the distance between the expert demo and generated trajectory, we use this loss only to
    # compare progress to our algorithm
    irl_loss_fn = IRLLoss()

    n_outer_iter = 200
    n_inner_iter = 5
    time_horizon = 10
    n_test_traj = 5
    irl_cost_tr, irl_cost_eval, learnable_cost_params, pred_traj = irl_training(robot_model, irl_loss_fn, expert_demo,
                                                                                start_q,
                                                                                trajs[1:1 + n_test_traj], n_outer_iter,
                                                                                n_inner_iter)

    if not os.path.exists(model_data_dir):
        os.makedirs(model_data_dir)

    torch.save({
        'irl_cost_tr': irl_cost_tr,
        'irl_cost_eval': irl_cost_eval,
        'cost_parameters': learnable_cost_params,
        'fina_pred_traj': pred_traj
    }, f=f'{model_data_dir}/{data_type}_Abbeel')
