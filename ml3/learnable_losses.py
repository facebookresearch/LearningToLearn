# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import torch
import torch.nn as nn


def weight_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight, gain=1.0)
        if module.bias is not None:
            module.bias.data.zero_()


def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()


class ML3_SineRegressionLoss(nn.Module):

    def __init__(self, in_dim, hidden_dim):
        super(ML3_SineRegressionLoss, self).__init__()
        w = [50, 50]
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim[0], bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim[0], hidden_dim[1], bias=False),
            nn.ReLU(),
        )
        self.loss = nn.Sequential(nn.Linear(hidden_dim[1], 1, bias=False), nn.Softplus())
        self.reset()

    def forward(self, y_in, y_target):
        y = torch.cat((y_in, y_target), dim=1)
        yp = self.layers(y)
        return self.loss(yp).mean()

    def reset(self):
        self.layers.apply(weight_init)
        self.loss.apply(weight_init)


class Ml3_loss_mountain_car(nn.Module):

    def __init__(self, meta_in, meta_out):
        super(Ml3_loss_mountain_car, self).__init__()


        activation = torch.nn.ELU
        num_neurons = 400
        self.model = torch.nn.Sequential(torch.nn.Linear(meta_in, num_neurons),
                                         activation(),
                                         torch.nn.Linear(num_neurons, num_neurons),
                                         activation(),
                                         torch.nn.Linear(num_neurons, meta_out))
        self.learning_rate = 1e-3

    def forward(self, x):
        return self.model(x)


class Ml3_loss_reacher(nn.Module):

    def __init__(self, meta_in, meta_out):
        super(Ml3_loss_reacher, self).__init__()

        activation = torch.nn.ELU
        output_activation = torch.nn.Softplus
        num_neurons = 400
        self.loss_fun = torch.nn.Sequential(torch.nn.Linear(meta_in, num_neurons),
                                         activation(),
                                         torch.nn.Linear(num_neurons, num_neurons),
                                         activation(),
                                         torch.nn.Linear(num_neurons, meta_out),
                                         output_activation())
        self.learning_rate = 1e-2

        self.norm_in = torch.Tensor(np.expand_dims(np.array([1.0, 1.0, 8.0, 8.0, 1.0, 1.0,1.0]), axis=0))

    def forward(self, x):
        return self.loss_fun(x/self.norm_in)


class Ml3_loss_shaped_sine(object):

    def __init__(self, meta_in=3, meta_out=1):
        def init_weights(m):
            if type(m) == torch.nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        activation = torch.nn.ELU
        num_neurons = 10
        self.model = torch.nn.Sequential(torch.nn.Linear(meta_in, num_neurons),activation(),
                                         torch.nn.Linear(num_neurons, num_neurons), activation(),
                                         torch.nn.Linear(num_neurons, num_neurons), activation(),
                                         torch.nn.Linear(num_neurons, meta_out))

        self.model.apply(init_weights)

        self.learning_rate = 3e-3


