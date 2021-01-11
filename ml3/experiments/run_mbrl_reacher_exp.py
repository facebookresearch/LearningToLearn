# Copyright (c) Facebook, Inc. and its affiliates.
import sys
import os
import numpy as np
import torch
import ml3
from ml3.envs.reacher_sim import ReacherSimulation
from ml3.mbrl_utils import Dynamics
from ml3.learnable_losses import Ml3_loss_reacher as Ml3_loss
from ml3.optimizee import Reacher_Policy as Policy
from ml3.ml3_train import meta_train_reacher as meta_train
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
    env = ReacherSimulation(gui=True)

    # initialize policy and save initialization for training
    policy = Policy(8,2)
    torch.save(policy.model.state_dict(), f"{EXP_FOLDER}/init_policy.pt")
    policy.model.load_state_dict(torch.load(f"{EXP_FOLDER}/init_policy.pt"))
    policy.model.eval()

    # initialize learned loss
    ml3_loss = Ml3_loss(7,1)
    # initialize task loss for meta training
    task_loss = Task_loss()

    # initialize learned dynamics model
    dmodel = Dynamics(env)

    # randomly create 5 goals for training
    num_task = 5
    x = np.random.uniform(-0.2, 0.2, num_task)
    y = np.random.uniform(-0.2, 0.2, num_task)
    np.random.shuffle(x)
    np.random.shuffle(y)
    z = np.zeros([num_task, 1])
    xygoals = np.vstack([x.T, y.T]).reshape(num_task, 2)
    xygoals = np.hstack([xygoals, z])

    goals = [np.hstack([np.array(env.get_target_joint_configuration(np.array(xy))),np.zeros(2)]) for xy in xygoals]
    time_horizon = 65

    if sys.argv[1] == 'train':

        n_outer_iter = 3000
        n_inner_iter = 1

        for random_data in range(3):
            states, actions = random_babbling(env, time_horizon)
            dmodel.train(torch.Tensor(states), torch.Tensor(actions))

        meta_train(policy, ml3_loss,dmodel,env, task_loss, goals, n_outer_iter, n_inner_iter, time_horizon, EXP_FOLDER)

    if sys.argv[1] == 'test':
        ml3_loss.model.load_state_dict(torch.load(f"{EXP_FOLDER}/ml3_loss_reacher.pt"))
        ml3_loss.model.eval()
        opt_iter = 2
        test_goal = np.array(env.get_target_joint_configuration(np.array([0.02534078, 0.19863741, 0.0])))
        test_goal = np.hstack([test_goal, np.zeros(2)])
        args = (torch.Tensor(test_goal),time_horizon,None,env,True)
        states = test_ml3_loss(policy, ml3_loss,opt_iter,*args)


