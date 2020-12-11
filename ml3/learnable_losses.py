# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import torch
import torch.nn as nn

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





class Ml3_loss_reacher(nn.Module):

    def __init__(self, meta_in, meta_out):
        super(Ml3_loss_reacher, self).__init__()

        activation = torch.nn.ELU
        output_activation = torch.nn.Softplus
        num_neurons = 400
        self.model = torch.nn.Sequential(torch.nn.Linear(meta_in, num_neurons),
                                         activation(),
                                         torch.nn.Linear(num_neurons, num_neurons),
                                         activation(),
                                         torch.nn.Linear(num_neurons, meta_out),
                                         output_activation())
        self.learning_rate = 1e-2

        self.norm_in = torch.Tensor(np.expand_dims(np.array([1.0, 1.0, 8.0, 8.0, 1.0, 1.0,1.0]), axis=0))


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


