# Copyright (c) Facebook, Inc. and its affiliates.
import random
import os
import torch
import numpy as np
import higher
import mbirl

from differentiable_robot_model import DifferentiableRobotModel

from mbirl.learnable_costs import LearnableWeightedCost, LearnableTimeDepWeightedCost
from mbirl.keypoint_mpc import KeypointMPCWrapper


# The IRL Loss, the learning objective for the learnable cost functions.
# The IRL loss measures the distance between the demonstrated trajectory and predicted trajectory
class IRLLoss(object):
    def __call__(self, pred_traj, target_traj):
        loss = ((pred_traj[:,-9:] - target_traj[:,-9:])**2).sum(dim=0)
        return loss.mean()


# Helper function for the irl learning loop
def irl_training(learnable_cost, robot_model, irl_loss_fn, expert_demo, n_outer_iter, n_inner_iter):

    learnable_cost_opt = torch.optim.Adam(learnable_cost.parameters(), lr=1e-2)

    for outer_i in range(n_outer_iter):

        learnable_cost_opt.zero_grad()
        # re-initialize action parameters for each outer iteration
        keypoint_mpc_wrapper = KeypointMPCWrapper(robot_model)
        action_optimizer = torch.optim.SGD(keypoint_mpc_wrapper.parameters(), lr=1.0)

        for _ in range(n_inner_iter):
            action_optimizer.zero_grad()
            keypoint_mpc_wrapper.reset_actions()
            with higher.innerloop_ctx(keypoint_mpc_wrapper, action_optimizer) as (fpolicy, diffopt):
                start_pose = expert_demo[0,:7]
                joint_state = torch.Tensor(start_pose)
                pred_traj = fpolicy.roll_out(joint_state.clone())

                # use the learned loss to update the action sequence
                learned_cost_val = learnable_cost(pred_traj, expert_demo[-1])
                diffopt.step(learned_cost_val)

                joint_state = torch.Tensor(start_pose)
                pred_traj = fpolicy.roll_out(joint_state)
                # compute task loss
                irl_loss = irl_loss_fn(pred_traj, expert_demo).mean()

        if outer_i % 10 == 0:
            print("irl cost training iter: {} loss: {}".format(outer_i, irl_loss.item()))

        # backprop gradient of learned cost parameters wrt irl loss
        learnable_cost_opt.zero_grad()
        learnable_cost.zero_grad()
        irl_loss.backward(retain_graph=True)

        learnable_cost_opt.step()


if __name__ == '__main__':
    random.seed(10)
    np.random.seed(10)
    torch.manual_seed(0)

    rest_pose = [0.0, 0.0, 0.0, 1.57079633, 0.0, 1.03672558, 0.0]

    rel_urdf_path = 'env/kuka_iiwa/urdf/iiwa7_ft_with_obj_keypts.urdf'
    urdf_path = os.path.join(mbirl.__path__[0], rel_urdf_path)
    robot_model = DifferentiableRobotModel(urdf_path=urdf_path, name="kuka_w_obj_keypts")

    #type of cost, either seq=time dependent or fixed
    # TODO: rename into weighted and timedepweighted ? something more descriptive
    cost_type = 'seq' #'fixed'

    expert_demo = torch.Tensor(np.load('expert_demo.npy'))
    learned_cost = None

    if cost_type=='fixed':
        learned_cost = LearnableWeightedCost()
    elif cost_type == 'seq':
        learned_cost = LearnableTimeDepWeightedCost()
    else:
        print('Cost not implemented')

    irl_loss_fn = IRLLoss()

    n_outer_iter = 200
    n_inner_iter = 1
    time_horizon = 25
    irl_training(learned_cost, robot_model, irl_loss_fn, expert_demo, n_outer_iter, n_inner_iter)