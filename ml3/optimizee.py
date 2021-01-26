# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import torch
import torch.nn as nn
from ml3.envs.mountain_car import MountainCar


def weight_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight, gain=1.0)
        if module.bias is not None:
            module.bias.data.zero_()


def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()


class SineModel(nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim):
        super(SineModel, self).__init__()
        net_dim = [in_dim] + hidden_dim

        layers = []

        for i in range(1, len(net_dim)):
            layers.append(nn.Linear(net_dim[i-1], net_dim[i]))
            layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)
        self.mean_pred = nn.Linear(hidden_dim[-1], out_dim)

    def reset(self):
        self.layers.apply(weight_init)
        self.mean_pred.apply(weight_init)

    def forward(self, x):
        feat = self.layers(x)
        return self.mean_pred(feat)


class MC_Policy(nn.Module):

    def __init__(self, pi_in, pi_out):
        super(MC_Policy, self).__init__()

        num_neurons = 200
        self.model = nn.Sequential(nn.Linear(pi_in, num_neurons,bias=False),
                                   nn.Linear(num_neurons, pi_out,bias=False))
        self.learning_rate = 1e-3

    def reset_gradients(self):
        for i, param in enumerate(self.model.parameters()):
            param.detach()

    def roll_out(self,fpolicy,s_0,goal,time_horizon):
        env = MountainCar()
        state = torch.Tensor(env.reset_to(s_0))
        states = []
        actions = []
        states.append(state)
        for t in range(time_horizon):

            u = fpolicy.forward(state)
            u = u.clamp(env.min_action,env.max_action)
            state = env.sim_step_torch(state.squeeze(), u.squeeze()).clone()
            states.append(state.clone())
            actions.append(u.clone())

        running_reward = torch.norm(state-goal)
        rewards = [torch.Tensor([running_reward])]*time_horizon
        return torch.stack(states), torch.stack(actions), torch.stack(rewards)


class Reacher_Policy(nn.Module):

    def __init__(self, pi_in, pi_out):
        super(Reacher_Policy, self).__init__()

        num_neurons = 64
        self.activation = torch.nn.Tanh
        self.policy_params = torch.nn.Sequential(torch.nn.Linear(pi_in, num_neurons),
                                                 self.activation(),
                                                 torch.nn.Linear(num_neurons, num_neurons),
                                                 self.activation(),
                                                 torch.nn.Linear(num_neurons, pi_out))
        self.learning_rate = 1e-4
        self.norm_in = torch.Tensor(np.array([1.0,1.0,8.0,8.0,1.0,1.0,1.0,1.0]))

    def forward(self, x):
        return self.policy_params(x)

    def reset_gradients(self):
        for i, param in enumerate(self.policy_params.parameters()):
            param.detach()

    def roll_out(self, goal, time_horizon, dmodel, env, real_rollout=False):

        state = torch.Tensor(env.reset())
        states = []
        actions = []
        states.append(state.clone())
        for t in range(time_horizon):

            u = self.forward(torch.cat((state.detach(), goal[:]), dim=0) / self.norm_in)
            u = u.clamp(-1.0, 1.0)
            if not real_rollout:
                pred_next_state = dmodel.step_model(state.squeeze(), u.squeeze()).clone()
            else:
                pred_next_state = torch.Tensor(env.step_model(state.squeeze().detach().numpy(), u.squeeze().detach().numpy()).copy())
            states.append(pred_next_state.clone())
            actions.append(u.clone())
            state_cost = torch.norm(pred_next_state[:]-goal[:]).detach().unsqueeze(0)
            state = pred_next_state.clone()

        # rewards to pass to meta loss
        rewards = [state_cost]*time_horizon
        return torch.stack(states), torch.stack(actions), torch.stack(rewards).detach()


class ShapedSineModel(torch.nn.Module):

    def __init__(self,theta=None):
        super(ShapedSineModel, self).__init__()
        if theta is None:
            self.freq = torch.nn.Parameter(torch.Tensor([0.1]))
        else:
            self.freq = torch.nn.Parameter(torch.Tensor([theta]))
        self.learning_rate = 1.0

    def forward(self, x):
        return torch.sin(self.freq*x)