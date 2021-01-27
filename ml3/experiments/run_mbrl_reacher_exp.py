# Copyright (c) Facebook, Inc. and its affiliates.
import sys
import os
import numpy as np
import torch
import pybullet
import ml3
from ml3.envs.reacher_sim import ReacherSimulation
from ml3.mbrl_utils import Dynamics
from ml3.learnable_losses import Ml3_loss_reacher as Ml3_loss
from ml3.optimizee import Reacher_Policy as Policy
from ml3.ml3_train import meta_train_mbrl_reacher as meta_train
from ml3.ml3_test import test_ml3_loss_reacher as test_ml3_loss

EXP_FOLDER = os.path.join(ml3.__path__[0], "experiments/data/mbrl_reacher")


class Task_loss(object):
    def __call__(self, a, s, goal):
        loss = 10*torch.norm(s[-1,:2]-goal[:2])+torch.mean(torch.norm(s[:,:2]-goal[:2],dim=1))+0.0001*torch.mean(torch.norm(s[:,2:],dim=1))
        return loss


def random_babbling(env, time_horizon):
    # do random babbling
    actions = np.random.uniform(-1.0, 1.0, [time_horizon, 2])
    states = []
    state = env.reset()
    states.append(state)
    for u in actions:
        state = env.sim_step(state, u)
        states.append(state.copy())

    return np.array(states), actions


if __name__ == '__main__':

    if not os.path.exists(EXP_FOLDER):
        os.makedirs(EXP_FOLDER)

    np.random.seed(0)
    torch.manual_seed(0)

    # create Reacher simulation
    env = ReacherSimulation(gui=False)

    # initialize policy and save initialization for training
    policy = Policy(8, 2, EXP_FOLDER)
    policy.reset()

    # initialize learned loss
    ml3_loss = Ml3_loss(7, 1)
    # initialize task loss for meta training
    task_loss = Task_loss()

    # initialize learned dynamics model
    dmodel = Dynamics(env)

    # generate training task
    num_task = 1
    train_goal = np.array(env.get_target_joint_configuration(np.array([0.02534078, 0.19863741, 0.0])))
    train_goal = np.hstack([train_goal, np.zeros(2)])

    goals = [train_goal]
    time_horizon = 65

    if sys.argv[1] == 'train':

        n_outer_iter = 3000  # 3000
        n_inner_iter = 1

        for random_data in range(3):
            states, actions = random_babbling(env, time_horizon)
            dmodel.train(torch.Tensor(states), torch.Tensor(actions))

        meta_train(policy, ml3_loss,dmodel,env, task_loss, goals, n_outer_iter, n_inner_iter, time_horizon, EXP_FOLDER)

    if sys.argv[1] == 'test':
        ml3_loss.load_state_dict(torch.load(f"{EXP_FOLDER}/ml3_loss_reacher.pt"))
        ml3_loss.eval()
        opt_iter = 2

        xy = [0.05534078, 0.150863741]

        test_goal = np.array(env.get_target_joint_configuration(np.array([xy[0], xy[1], 0.0])))
        test_goal = np.hstack([test_goal, np.zeros(2)])
        args = (torch.Tensor(test_goal),time_horizon,None,env,True)
        print('goal joint position:', test_goal[:2])
        states = test_ml3_loss(policy, ml3_loss,opt_iter,*args)
        print('achieved joint position',states[-1,:2])
