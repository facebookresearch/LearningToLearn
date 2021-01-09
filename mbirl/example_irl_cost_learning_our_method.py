# Copyright (c) Facebook, Inc. and its affiliates.
import random
import os, sys
import torch
import numpy as np
import higher
import mbirl
import dill as pickle
from os.path import dirname, abspath
import matplotlib.pyplot as plt

from differentiable_robot_model import DifferentiableRobotModel

_ROOT_DIR = dirname(abspath(__file__))
sys.path.append(_ROOT_DIR)

from mbirl.learnable_costs import LearnableWeightedCost, LearnableTimeDepWeightedCost, LearnableRBFWeightedCost
from mbirl.keypoint_mpc import KeypointMPCWrapper


traj_data_dir = os.path.join(_ROOT_DIR, 'traj_data')
model_data_dir = os.path.join(_ROOT_DIR, 'model_data')


# The IRL Loss, the learning objective for the learnable cost functions.
# The IRL loss measures the distance between the demonstrated trajectory and predicted trajectory
class IRLLoss(object):
    def __call__(self, pred_traj, target_traj):
        loss = ((pred_traj[:, -6:] - target_traj[:, -6:]) ** 2).sum(dim=0)
        return loss.mean()


def evaluate_action_optimization(learned_cost, robot_model, irl_loss_fn, trajs, n_inner_iter):
    # np.random.seed(cfg.random_seed)
    # torch.manual_seed(cfg.random_seed)

    eval_costs = []
    for i, traj in enumerate(trajs):

        traj_len = len(traj['desired_keypoints'])
        start_pose = traj['start_joint_config'].squeeze()
        expert_demo = traj['desired_keypoints'].reshape(traj_len, -1)
        expert_demo = torch.Tensor(expert_demo)
        time_horizon, n_keypt_dim = expert_demo.shape

        keypoint_mpc_wrapper = KeypointMPCWrapper(robot_model, time_horizon=time_horizon-1, n_keypt_dim=n_keypt_dim)
        action_optimizer = torch.optim.SGD(keypoint_mpc_wrapper.parameters(), lr=0.001)

        for i in range(n_inner_iter):
            action_optimizer.zero_grad()

            pred_traj = keypoint_mpc_wrapper.roll_out(start_pose.clone())
            # use the learned loss to update the action sequence
            learned_cost_val = learned_cost(pred_traj, expert_demo[-1])
            learned_cost_val.backward(retain_graph=True)
            action_optimizer.step()

        # Actually take the next step after optimizing the action
        pred_state_traj_new = keypoint_mpc_wrapper.roll_out(start_pose.clone())
        eval_costs.append(irl_loss_fn(pred_state_traj_new, expert_demo).mean())

    return torch.stack(eval_costs).detach()


# Helper function for the irl learning loop
def irl_training(learnable_cost, robot_model, irl_loss_fn, expert_demo, start_pose, test_trajs, n_outer_iter, n_inner_iter):

    time_horizon, n_keypt_dim = expert_demo.shape
    #learnable_cost_opt = torch.optim.RMSprop(learnable_cost.parameters(), lr=1e-2)
    learnable_cost_opt = torch.optim.Adam(learnable_cost.parameters(), lr=1e-2)#, amsgrad=True)
    keypoint_mpc_wrapper = KeypointMPCWrapper(robot_model, time_horizon=time_horizon-1, n_keypt_dim=n_keypt_dim)
    action_optimizer = torch.optim.SGD(keypoint_mpc_wrapper.parameters(), lr=0.001)

    irl_cost_tr = []
    irl_cost_eval = []

    # unroll and extract expected features
    pred_traj = keypoint_mpc_wrapper.roll_out(start_pose.clone())

    print("Cost function parameters to be optimized:")
    for name, param in learnable_cost.named_parameters():
        print(name)
        print(param)

    # get initial irl loss
    irl_loss = irl_loss_fn(pred_traj, expert_demo).mean()
    irl_cost_tr.append(irl_loss)
    print("irl cost training iter: {} loss: {}".format(0, irl_loss.item()))

    # start of inverse RL loop
    for outer_i in range(n_outer_iter):

        learnable_cost_opt.zero_grad()
        # re-initialize action parameters for each outer iteration

        for _ in range(n_inner_iter):
            action_optimizer.zero_grad()
            keypoint_mpc_wrapper.reset_actions()

            with higher.innerloop_ctx(keypoint_mpc_wrapper, action_optimizer) as (fpolicy, diffopt):
                pred_traj = fpolicy.roll_out(start_pose.clone())

                # use the learned loss to update the action sequence
                learned_cost_val = learnable_cost(pred_traj, expert_demo[-1])
                diffopt.step(learned_cost_val)

                pred_traj = fpolicy.roll_out(start_pose)
                # compute task loss
                irl_loss = irl_loss_fn(pred_traj, expert_demo).mean()

                print("irl cost training iter: {} loss: {}".format(outer_i+1, irl_loss.item()))

                # backprop gradient of learned cost parameters wrt irl loss
                irl_loss.backward(retain_graph=True)
                irl_cost_tr.append(irl_loss.detach())

        learnable_cost_opt.step()

        if outer_i % 25 == 0:
            plt.figure()
            plt.plot(pred_traj[:, 7].detach(), pred_traj[:, 9].detach(), 'o')
            plt.plot(expert_demo[:, 0], expert_demo[:, 2], 'o')
            plt.title("outer i: {}".format(outer_i))
            plt.show()

        eval_costs = evaluate_action_optimization(learnable_cost.eval(), robot_model, irl_loss_fn, test_trajs,
                                                  n_inner_iter)
        irl_cost_eval.append(eval_costs)
        learnable_cost_params = {}
        for name, param in learnable_cost.named_parameters():
            learnable_cost_params[name] = param

    return torch.stack(irl_cost_tr), torch.stack(irl_cost_eval), learnable_cost_params, pred_traj


if __name__ == '__main__':
    random.seed(10)
    np.random.seed(10)
    torch.manual_seed(0)

    rest_pose = [0.0, 0.0, 0.0, 1.57079633, 0.0, 1.03672558, 0.0]

    rel_urdf_path = 'env/kuka_iiwa/urdf/iiwa7_ft_with_obj_keypts.urdf'
    urdf_path = os.path.join(mbirl.__path__[0], rel_urdf_path)
    robot_model = DifferentiableRobotModel(urdf_path=urdf_path, name="kuka_w_obj_keypts")

    # data_type = 'reaching'
    data_type = 'placing'
    with open(f'{traj_data_dir}/traj_data_{data_type}.pkl', 'rb') as f:
        trajs = pickle.load(f)

    traj = trajs[0]
    traj_len = len(traj['desired_keypoints'])

    start_q = traj['start_joint_config'].squeeze()
    expert_demo = traj['desired_keypoints'].reshape(traj_len, -1)
    expert_demo = torch.Tensor(expert_demo)
    print(expert_demo.shape)
    n_keypt_dim = expert_demo.shape[1]
    time_horizon = expert_demo.shape[0]

    # type of cost
    # cost_type = 'Weighted'
    # cost_type = 'TimeDep'
    cost_type = 'RBF'

    learnable_cost = None

    if cost_type == 'Weighted':
        learnable_cost = LearnableWeightedCost(dim=n_keypt_dim)
    elif cost_type == 'TimeDep':
        learnable_cost = LearnableTimeDepWeightedCost(time_horizon=time_horizon, dim=n_keypt_dim)
    elif cost_type == 'RBF':
        learnable_cost = LearnableRBFWeightedCost(time_horizon=time_horizon, dim=n_keypt_dim)
    else:
        print('Cost not implemented')

    irl_loss_fn = IRLLoss()

    n_outer_iter = 1000
    n_inner_iter = 1
    n_test_traj = 2
    train_trajs = trajs[0:1]
    test_trajs = trajs[1:1+n_test_traj]
    irl_cost_tr, irl_cost_eval, learnable_cost_params, pred_traj = irl_training(learnable_cost, robot_model, irl_loss_fn,
                                                                     expert_demo, start_q, test_trajs,
                                                                     n_outer_iter, n_inner_iter)


    if not os.path.exists(model_data_dir):
        os.makedirs(model_data_dir)

    torch.save({
        'irl_cost_tr': irl_cost_tr,
        'irl_cost_eval': irl_cost_eval,
        'cost_parameters': learnable_cost_params,
        'fina_pred_traj': pred_traj
    }, f=f'{model_data_dir}/{data_type}_{cost_type}')
