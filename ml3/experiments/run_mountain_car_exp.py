# Copyright (c) Facebook, Inc. and its affiliates.
import sys
import os
import numpy as np
import torch
import ml3
from ml3.ml3_train import meta_train_mountain_car as meta_train
from ml3.ml3_test import test_ml3_loss_mountain_car as test_ml3_loss
from ml3.learnable_losses import Ml3_loss_mountain_car as Ml3_loss
from ml3.optimizee import MC_Policy
from ml3.envs.mountain_car import MountainCar

EXP_FOLDER = os.path.join(ml3.__path__[0], "experiments/data/mountain_car")


class Task_loss(object):
    def __call__(self, a, s, goal, goal_exp,shaped_loss):

        loss = (torch.norm(s - goal)).mean()
        if shaped_loss:
            loss = (torch.norm(s[:15] - goal_exp)).mean() + (torch.norm(s[15:] - goal)).mean()

        return loss


if __name__ == '__main__':

    if not os.path.exists(EXP_FOLDER):
        os.makedirs(EXP_FOLDER)

    np.random.seed(0)
    torch.manual_seed(0)

    policy = MC_Policy(2,1)
    ml3_loss = Ml3_loss(4,1)

    task_loss = Task_loss()

    goal = [0.5000, 1.0375]
    goal_extra = [-0.9470, -0.0055]

    env = MountainCar()
    s_0 = env.reset()

    n_outer_iter = 300
    n_inner_iter = 1

    time_horizon = 35

    if sys.argv[1] == 'train':
        shaped_loss = sys.argv[2] == 'True'
        meta_train(policy, ml3_loss, task_loss, s_0, goal, goal_extra, n_outer_iter, n_inner_iter, time_horizon, shaped_loss)
        if shaped_loss:
            torch.save(ml3_loss.model.state_dict(), f"{EXP_FOLDER}/shaped_ml3_loss_mountain_car.pt")
        else:
            torch.save(ml3_loss.model.state_dict(), f"{EXP_FOLDER}/ml3_loss_mountain_car.pt")

    if sys.argv[1] == 'test':
        shaped_loss = sys.argv[2] == 'True'
        if shaped_loss:
            ml3_loss.model.load_state_dict(torch.load(f"{EXP_FOLDER}/shaped_ml3_loss_mountain_car.pt"))
        else:
            ml3_loss.model.load_state_dict(torch.load(f"{EXP_FOLDER}/ml3_loss_mountain_car.pt"))
        ml3_loss.model.eval()
        opt_iter = 2
        args = (torch.Tensor(s_0), torch.Tensor(goal), time_horizon)
        states = test_ml3_loss(policy,ml3_loss, opt_iter,*args)
        if shaped_loss:
            np.save(f"{EXP_FOLDER}/shaped_ml3_mc_states.npy",states)
        else:
            np.save(f"{EXP_FOLDER}/ml3_mc_states.npy", states)

        if shaped_loss:
            env.render(list(np.array(states)[:, 0]), file_path=f"{EXP_FOLDER}/shaped_ml3_mc.gif")
        else:
            env.render(list(np.array(states)[:, 0]), file_path=f"{EXP_FOLDER}/ml3_mc.gif")



