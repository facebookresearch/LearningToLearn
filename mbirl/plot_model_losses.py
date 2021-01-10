import os, sys
import torch
import numpy as np
import dill as pickle
import matplotlib.pyplot as plt
from os.path import dirname, abspath
from differentiable_robot_model import DifferentiableRobotModel

_ROOT_DIR = dirname(abspath(__file__))
sys.path.append(_ROOT_DIR)

traj_data_dir = os.path.join(_ROOT_DIR, 'traj_data')
model_data_dir = os.path.join(_ROOT_DIR, 'model_data')
from mbirl.keypoint_mpc import KeypointMPCWrapper
from mbirl.learnable_costs import *
import mbirl


def evaluate_action_optimization(learned_loss, robot_model, irl_loss_fn, trajs, n_inner_iter):
    # np.random.seed(cfg.random_seed)
    # torch.manual_seed(cfg.random_seed)

    print(f'Weights: {learned_loss.weights}')

    eval_costs = []
    predicted_trajs = []
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
            learned_cost_val = learned_loss(pred_traj, expert_demo[-1])
            learned_cost_val.backward()
            action_optimizer.step()

        # Actually take the next step after optimizing the action
        pred_state_traj_new = keypoint_mpc_wrapper.roll_out(start_pose.clone())
        eval_costs.append(irl_loss_fn(pred_state_traj_new, expert_demo).mean())
        predicted_trajs.append(pred_state_traj_new)

    return torch.stack(eval_costs).detach(), torch.stack(predicted_trajs).detach()


for data_type in ['placing', 'reaching']:
    if not os.path.exists(
            f"{model_data_dir}/{data_type}_TimeDep") or not os.path.exists(
        f"{model_data_dir}/{data_type}_Weighted") or not os.path.exists(f"{model_data_dir}/{data_type}_RBF"):
        continue

    # if not os.path.exists(f"{model_data_dir}/{data_type}_Abbeel") or not os.path.exists(
    #         f"{model_data_dir}/{data_type}_TimeDep") or not os.path.exists(
    #     f"{model_data_dir}/{data_type}_Weighted") or not os.path.exists(f"{model_data_dir}/{data_type}_RBF"):
    #     continue

    #baseline = torch.load(f"{model_data_dir}/{data_type}_Abbeel")
    timedep = torch.load(f"{model_data_dir}/{data_type}_TimeDep")
    weighted = torch.load(f"{model_data_dir}/{data_type}_Weighted")
    rbf = torch.load(f"{model_data_dir}/{data_type}_RBF")

    # IRL Cost

    plt.figure()

    #plt.plot(baseline['irl_cost_tr'].detach(), color='red', alpha=0.5, label="Baseline")
    plt.plot(weighted['irl_cost_tr'].detach(), color='orange', label="Weighted Ours")
    plt.plot(timedep['irl_cost_tr'].detach(), color='green', label="Time Dep Weighted Ours")
    plt.plot(rbf['irl_cost_tr'].detach(), color='violet', label="RBF Weighted Ours")
    plt.xlabel("iterations")
    plt.ylabel("IRL Cost")
    plt.legend()

    plt.show()
    plt.savefig(f"{model_data_dir}/{data_type}_IRL_cost.png")

    # Eval

    plt.figure()
    #baseline_trace = baseline['irl_cost_eval'].detach()
    #b_mean = baseline_trace.mean(dim=-1)
    #b_std = baseline_trace.std(dim=-1)
    weighted_trace = weighted['irl_cost_eval'].detach()
    w_mean = weighted_trace.mean(dim=-1)
    w_std = weighted_trace.std(dim=-1)
    timedep_trace = timedep['irl_cost_eval'].detach()
    t_mean = timedep_trace.mean(dim=-1)
    t_std = timedep_trace.std(dim=-1)
    rbf_trace = rbf['irl_cost_eval'].detach()
    r_mean = rbf_trace.mean(dim=-1)
    r_std = rbf_trace.std(dim=-1)
    #plt.plot(b_mean, color='red', alpha=0.5, label="Baseline")
    #plt.fill_between(np.arange(len(b_mean)), b_mean - b_std, b_mean + b_std, color='red', alpha=0.1)
    plt.plot(w_mean, color='orange', label="Weighted Ours")
    plt.fill_between(np.arange(len(w_mean)), w_mean - w_std, w_mean + w_std, color='orange', alpha=0.1)
    plt.plot(t_mean, color='green', label="Time Dep Weighted Ours")
    plt.fill_between(np.arange(len(t_mean)), t_mean - t_std, t_mean + t_std, color='green', alpha=0.1)
    plt.plot(r_mean, color='violet', label="RBF Weighted Ours")
    plt.fill_between(np.arange(len(r_mean)), r_mean - r_std, r_mean + r_std, color='blueviolet', alpha=0.1)
    plt.xlabel("iterations")
    plt.ylabel("Eval Cost")
    plt.legend()

    plt.show()
    plt.savefig(f"{model_data_dir}/{data_type}_Eval.png")

    # Get Demo Trajectories

    with open(f'{traj_data_dir}/traj_data_{data_type}.pkl', 'rb') as f:
        trajs = pickle.load(f)

    n_test_traj = 3
    n_inner_iter = 1
    train_trajs = trajs[0:1]
    # test_trajs = trajs[0:1]
    test_trajs = trajs[0:0 + n_test_traj]

    traj_len = len(train_trajs[0]['desired_keypoints'])
    time_horizon, n_keypt_dim = test_trajs[0]['desired_keypoints'].reshape(traj_len, -1).shape

    # Plot predicted traj during training

    plt.figure()
    #print(baseline['fina_pred_traj'].detach().shape)

    plt.scatter(x=train_trajs[0]['desired_keypoints'][:, 0, 0], y=train_trajs[0]['desired_keypoints'][:, 0, 2],
                color='blue', label='Demo')
    plt.plot(train_trajs[0]['desired_keypoints'][:, 0, 0],
             train_trajs[0]['desired_keypoints'][:, 0, 2], color='blue')

    #plt.scatter(x=baseline['fina_pred_traj'][:, -n_keypt_dim].detach(),
    #            y=baseline['fina_pred_traj'][:, -n_keypt_dim + 2].detach(), color='red', alpha=0.5, label="Baseline", s=100)
    #plt.plot(baseline['fina_pred_traj'][:, -n_keypt_dim].detach(),
    #         baseline['fina_pred_traj'][:, -n_keypt_dim + 2].detach(), color='red', alpha=0.5)

    plt.scatter(x=weighted['fina_pred_traj'][:, -n_keypt_dim].detach(),
                y=weighted['fina_pred_traj'][:, -n_keypt_dim + 2].detach(), color='orange', label="Weighted Ours")
    plt.plot(weighted['fina_pred_traj'][:, -n_keypt_dim].detach(),
             weighted['fina_pred_traj'][:, -n_keypt_dim + 2].detach(), color='orange')

    plt.scatter(x=timedep['fina_pred_traj'][:, -n_keypt_dim].detach(),
                y=timedep['fina_pred_traj'][:, -n_keypt_dim + 2].detach(), color='green',
                label="Time Dep Weighted Ours")
    plt.plot(timedep['fina_pred_traj'][:, -n_keypt_dim].detach(),
             timedep['fina_pred_traj'][:, -n_keypt_dim + 2].detach(), color='green')

    plt.scatter(x=rbf['fina_pred_traj'][:, -n_keypt_dim].detach(),
                y=rbf['fina_pred_traj'][:, -n_keypt_dim + 2].detach(), color='blueviolet', label="RBF Weighted Ours")
    plt.plot(rbf['fina_pred_traj'][:, -n_keypt_dim].detach(),
             rbf['fina_pred_traj'][:, -n_keypt_dim + 2].detach(), color='blueviolet')

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()

    plt.show()
    plt.savefig(f"{model_data_dir}/{data_type}_pred_train_traj.png")

    # Plot trajectories with 1 eval trajectory

    #b_weights = baseline['cost_parameters']['weights']
    w_weights = weighted["cost_parameters"]['weights']
    t_weights = timedep["cost_parameters"]['weights']
    r_weights = rbf["cost_parameters"]['weights_fn.weights']

    rel_urdf_path = 'env/kuka_iiwa/urdf/iiwa7_ft_with_obj_keypts.urdf'
    urdf_path = os.path.join(mbirl.__path__[0], rel_urdf_path)
    robot_model = DifferentiableRobotModel(urdf_path=urdf_path, name="kuka_w_obj_keypts")

    # b_eval_losses, b_predicted_trajectories = evaluate_action_optimization(
    #     BaselineCost(dim=n_keypt_dim, weights=b_weights),
    #     robot_model=robot_model,
    #     irl_loss_fn=IRLLoss(dim=n_keypt_dim),
    #     trajs=test_trajs,
    #     n_inner_iter=n_inner_iter)
    w_eval_losses, w_predicted_trajectories = evaluate_action_optimization(
        LearnableWeightedCost(dim=n_keypt_dim, weights=w_weights),
        robot_model=robot_model,
        irl_loss_fn=IRLLoss(dim=n_keypt_dim),
        trajs=test_trajs,
        n_inner_iter=n_inner_iter)
    t_eval_losses, t_predicted_trajectories = evaluate_action_optimization(
        LearnableTimeDepWeightedCost(dim=n_keypt_dim, time_horizon=time_horizon, weights=t_weights),
        robot_model=robot_model,
        irl_loss_fn=IRLLoss(dim=n_keypt_dim),
        trajs=test_trajs,
        n_inner_iter=n_inner_iter)
    r_eval_losses, r_predicted_trajectories = evaluate_action_optimization(
        LearnableRBFWeightedCost(dim=n_keypt_dim, time_horizon=time_horizon, weights=r_weights),
        robot_model=robot_model,
        irl_loss_fn=IRLLoss(dim=n_keypt_dim),
        trajs=test_trajs,
        n_inner_iter=n_inner_iter)

    plt.close("all")

    fig = plt.figure(figsize=(n_test_traj * 5, 5))
    for i in range(n_test_traj):
        ax = fig.add_subplot(1, n_test_traj, i + 1, projection='3d')
        print(test_trajs[i]["desired_keypoints"].shape)
        ax.plot(test_trajs[i]["desired_keypoints"][:, 0, 0], test_trajs[i]["desired_keypoints"][:, 0, 1],
                test_trajs[i]["desired_keypoints"][:, 0, 2], color='blue', label='Demonstration')
        #ax.plot(b_predicted_trajectories[i, :, -n_keypt_dim], b_predicted_trajectories[i, :, -n_keypt_dim + 1],
        #        b_predicted_trajectories[i, :, -n_keypt_dim + 2], color='red', alpha=0.5, label='Baseline')
        ax.plot(w_predicted_trajectories[i, :, -n_keypt_dim], w_predicted_trajectories[i, :, -n_keypt_dim + 1],
                w_predicted_trajectories[i, :, -n_keypt_dim + 2], color='orange', label='Weighted Ours')
        ax.plot(t_predicted_trajectories[i, :, -n_keypt_dim], t_predicted_trajectories[i, :, -n_keypt_dim + 1],
                t_predicted_trajectories[i, :, -n_keypt_dim + 2], color='green', label='Time Dependent Ours')
        ax.plot(r_predicted_trajectories[i, :, -n_keypt_dim], r_predicted_trajectories[i, :, -n_keypt_dim + 1],
                r_predicted_trajectories[i, :, -n_keypt_dim + 2], color='blueviolet', label='RBF Ours')
        # range_x = test_trajs[i]["desired_keypoints"][:, 0, 0].max() - test_trajs[i]["desired_keypoints"][:, 0, 0].min()
        # range_y = test_trajs[i]["desired_keypoints"][:, 0, 1].max() - test_trajs[i]["desired_keypoints"][:, 0, 1].min()
        # range_z = test_trajs[i]["desired_keypoints"][:, 0, 2].max() - test_trajs[i]["desired_keypoints"][:, 0, 2].min()
        # max_range = max(range_x, range_y, range_z)
        # ax.set_xlim([test_trajs[i]["desired_keypoints"][:, 0, 0].min(),
        #              test_trajs[i]["desired_keypoints"][:, 0, 0].min() + max_range])
        # ax.set_ylim([test_trajs[i]["desired_keypoints"][:, 0, 1].min(),
        #              test_trajs[i]["desired_keypoints"][:, 0, 1].min() + max_range])
        # ax.set_zlim([test_trajs[i]["desired_keypoints"][:, 0, 2].min(),
        #              test_trajs[i]["desired_keypoints"][:, 0, 2].min() + max_range])
        min_x = -100.0;
        max_x = -50.0
        min_y = 0.0;
        max_y = 30
        min_z = 50;
        max_z = 100
        ax.set_xlim([min_x, max_x])
        ax.set_ylim([min_y, max_y])
        ax.set_zlim([min_z, max_z])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.legend()
    plt.show()

    fig = plt.figure(figsize=(n_test_traj * 5, 5))
    for i in range(n_test_traj):
        ax = fig.add_subplot(1, n_test_traj, i + 1)
        print(test_trajs[i]["desired_keypoints"].shape)
        ax.plot(test_trajs[i]["desired_keypoints"][:, 0, 0], test_trajs[i]["desired_keypoints"][:, 0, 2], color='blue', label='Demonstration')
        ax.plot(w_predicted_trajectories[i, :, -n_keypt_dim], w_predicted_trajectories[i, :, -n_keypt_dim + 2], color='orange', label='Weighted Ours')
        ax.plot(t_predicted_trajectories[i, :, -n_keypt_dim], t_predicted_trajectories[i, :, -n_keypt_dim + 2], color='green', label='Time Dependent Ours')
        ax.plot(r_predicted_trajectories[i, :, -n_keypt_dim], r_predicted_trajectories[i, :, -n_keypt_dim + 2], color='blueviolet', label='RBF Ours')
        min_x = -90.0;
        max_x = -50.0
        # min_y = 0.0;
        # max_y = 30
        min_z = 50;
        max_z = 100
        ax.set_xlim([min_x, max_x])
        ax.set_ylim([min_z, max_z])
        # ax.set_zlim([min_z, max_z])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        # ax.set_zlabel("z")
        ax.legend()
    plt.show()
