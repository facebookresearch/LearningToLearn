# Copyright (c) Facebook, Inc. and its affiliates.
import random
import os
import torch
import numpy as np
import higher
import mbirl
import matplotlib.pyplot as plt

from differentiable_robot_model import DifferentiableRobotModel

from mbirl.learnable_costs import LearnableWeightedCost, LearnableTimeDepWeightedCost, LearnableRBFWeightedCost
from mbirl.keypoint_mpc import KeypointMPCWrapper

EXP_FOLDER = os.path.join(mbirl.__path__[0], "experiments")
traj_data_dir = os.path.join(EXP_FOLDER, 'traj_data')
model_data_dir = os.path.join(EXP_FOLDER, 'traj_data')


# The IRL Loss, the learning objective for the learnable cost functions.
# The IRL loss measures the distance between the demonstrated trajectory and predicted trajectory
class IRLLoss(object):
    def __call__(self, pred_traj, target_traj):
        loss = ((pred_traj[:, -6:] - target_traj[:, -6:]) ** 2).sum(dim=0)
        return loss.mean()


def evaluate_action_optimization(learned_cost, robot_model, irl_loss_fn, trajs, n_inner_iter, action_lr=0.001):
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
        action_optimizer = torch.optim.SGD(keypoint_mpc_wrapper.parameters(), lr=action_lr)

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
def irl_training(learnable_cost, robot_model, irl_loss_fn, train_trajs, test_trajs, n_outer_iter, n_inner_iter,
                 data_type, cost_type, cost_lr=1e-2, action_lr=1e-3):
    irl_loss_on_train = []
    irl_loss_on_test = []

    learnable_cost_opt = torch.optim.Adam(learnable_cost.parameters(), lr=cost_lr)

    irl_loss_dems = []
    # initial loss before training

    plots_dir = os.path.join(model_data_dir, data_type, cost_type)

    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    for demo_i in range(len(train_trajs)):
        expert_demo_dict = train_trajs[demo_i]

        start_pose = expert_demo_dict['start_joint_config'].squeeze()
        expert_demo = expert_demo_dict['desired_keypoints'].reshape(traj_len, -1)
        expert_demo = torch.Tensor(expert_demo)
        time_horizon, n_keypt_dim = expert_demo.shape

        keypoint_mpc_wrapper = KeypointMPCWrapper(robot_model, time_horizon=time_horizon - 1, n_keypt_dim=n_keypt_dim)
        # unroll and extract expected features
        pred_traj = keypoint_mpc_wrapper.roll_out(start_pose.clone())

        # get initial irl loss
        irl_loss = irl_loss_fn(pred_traj, expert_demo).mean()
        irl_loss_dems.append(irl_loss.item())

    irl_loss_on_train.append(torch.Tensor(irl_loss_dems).mean())
    print("irl cost training iter: {} loss: {}".format(0, irl_loss_on_train[-1]))

    print("Cost function parameters to be optimized:")
    for name, param in learnable_cost.named_parameters():
        print(name)
        print(param)

    # start of inverse RL loop
    for outer_i in range(n_outer_iter):
        irl_loss_dems = []

        for demo_i in range(len(train_trajs)):
            learnable_cost_opt.zero_grad()
            expert_demo_dict = train_trajs[demo_i]

            start_pose = expert_demo_dict['start_joint_config'].squeeze()
            expert_demo = expert_demo_dict['desired_keypoints'].reshape(traj_len, -1)
            expert_demo = torch.Tensor(expert_demo)
            time_horizon, n_keypt_dim = expert_demo.shape

            keypoint_mpc_wrapper = KeypointMPCWrapper(robot_model, time_horizon=time_horizon - 1,
                                                      n_keypt_dim=n_keypt_dim)
            action_optimizer = torch.optim.SGD(keypoint_mpc_wrapper.parameters(), lr=action_lr)

            with higher.innerloop_ctx(keypoint_mpc_wrapper, action_optimizer) as (fpolicy, diffopt):
                pred_traj = fpolicy.roll_out(start_pose.clone())

                # use the learned loss to update the action sequence
                learned_cost_val = learnable_cost(pred_traj, expert_demo[-1])
                diffopt.step(learned_cost_val)

                pred_traj = fpolicy.roll_out(start_pose)
                # compute task loss
                irl_loss = irl_loss_fn(pred_traj, expert_demo).mean()
                # backprop gradient of learned cost parameters wrt irl loss
                irl_loss.backward(retain_graph=True)
                irl_loss_dems.append(irl_loss.detach())

            learnable_cost_opt.step()

            if outer_i % 25 == 0:
                plt.figure()
                plt.plot(pred_traj[:, 7].detach(), pred_traj[:, 9].detach(), 'o')
                plt.plot(expert_demo[:, 0], expert_demo[:, 2], 'x')
                plt.title("outer i: {}".format(outer_i))
                plt.savefig(os.path.join(plots_dir, f'{demo_i}_{outer_i}.png'))

        irl_loss_on_train.append(torch.Tensor(irl_loss_dems).mean())
        test_irl_losses = evaluate_action_optimization(learnable_cost.eval(), robot_model, irl_loss_fn, test_trajs,
                                                  n_inner_iter)
        print("irl loss (on train) training iter: {} loss: {}".format(outer_i + 1, irl_loss_on_train[-1]))
        print("irl loss (on test) training iter: {} loss: {}".format(outer_i + 1, test_irl_losses.mean().item()))
        print("")
        irl_loss_on_test.append(test_irl_losses)
        learnable_cost_params = {}
        for name, param in learnable_cost.named_parameters():
            learnable_cost_params[name] = param

        if len(learnable_cost_params) == 0:
            # For RBF Weighted Cost
            for name, param in learnable_cost.weights_fn.named_parameters():
                learnable_cost_params[name] = param

    plt.figure()
    plt.plot(pred_traj[:, 7].detach(), pred_traj[:, 9].detach(), 'o')
    plt.plot(expert_demo[:, 0], expert_demo[:, 2], 'x')
    plt.title("final")
    plt.savefig(os.path.join(plots_dir, f'{demo_i}_final.png'))

    return torch.stack(irl_loss_on_train), torch.stack(irl_loss_on_test), learnable_cost_params, pred_traj


if __name__ == '__main__':
    random.seed(10)
    np.random.seed(10)
    torch.manual_seed(0)

    rest_pose = [0.0, 0.0, 0.0, 1.57079633, 0.0, 1.03672558, 0.0]

    rel_urdf_path = 'env/kuka_iiwa/urdf/iiwa7_ft_with_obj_keypts.urdf'
    urdf_path = os.path.join(mbirl.__path__[0], rel_urdf_path)
    robot_model = DifferentiableRobotModel(urdf_path=urdf_path, name="kuka_w_obj_keypts")

    data_type = 'placing'
    trajs = torch.load(f'{traj_data_dir}/traj_data_{data_type}.pt')

    traj = trajs[0]
    traj_len = len(traj['desired_keypoints'])

    start_q = traj['start_joint_config'].squeeze()
    expert_demo = traj['desired_keypoints'].reshape(traj_len, -1)
    expert_demo = torch.Tensor(expert_demo)
    print(expert_demo.shape)
    n_keypt_dim = expert_demo.shape[1]
    time_horizon = expert_demo.shape[0]

    # type of cost
    #cost_type = 'Weighted'
    #cost_type = 'TimeDep'
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

    cost_lr = 1e-2
    action_lr = 1e-3
    n_outer_iter = 100
    n_inner_iter = 1
    n_test_traj = 2
    train_trajs = trajs[0:3]
    test_trajs = trajs[3:3 + n_test_traj]
    irl_loss_train, irl_loss_test, learnable_cost_params, pred_traj = irl_training(learnable_cost, robot_model,
                                                                                   irl_loss_fn,
                                                                                   train_trajs, test_trajs,
                                                                                   n_outer_iter, n_inner_iter,
                                                                                   cost_type=cost_type,
                                                                                   data_type=data_type,
                                                                                   cost_lr=cost_lr,
                                                                                   action_lr=action_lr)

    if not os.path.exists(model_data_dir):
        os.makedirs(model_data_dir)

    torch.save({
        'irl_loss_train': irl_loss_train,
        'irl_loss_test': irl_loss_test,
        'cost_parameters': learnable_cost_params,
        'fina_pred_traj': pred_traj,
        'n_inner_iter': n_inner_iter,
        'action_lr': action_lr
    }, f=f'{model_data_dir}/{data_type}_{cost_type}')
