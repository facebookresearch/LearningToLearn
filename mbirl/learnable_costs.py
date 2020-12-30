# Copyright (c) Facebook, Inc. and its affiliates.
import torch

# The learned weighted cost, with fixed weights ###
class LearnableWeightedCost(torch.nn.Module):
    def __init__(self, dim=9):
        super(LearnableWeightedCost, self).__init__()
        self.weights = torch.nn.Parameter(0.01 * torch.ones([dim,1]))
        self.clip = torch.nn.Softplus()
        self.meta_grads = [[] for _, _ in enumerate(self.parameters())]

    def forward(self, y_in, y_target):
        assert y_in.dim() == 2
        mse = ((y_in[:,-9:] - y_target[-9:]) ** 2).squeeze()

        # weighted mse
        wmse = torch.mm(mse,self.weights)
        return wmse.mean()


# The learned weighted cost, with time dependent weights ###
class LearnableTimeDepWeightedCost(torch.nn.Module):
    def __init__(self, dim=9):
        super(LearnableTimeDepWeightedCost, self).__init__()
        self.weights = torch.nn.Parameter(0.01 * torch.ones([25,dim]))
        self.clip = torch.nn.Softplus()
        self.meta_grads = [[] for _, _ in enumerate(self.parameters())]

    def forward(self, y_in, y_target):
        assert y_in.dim() == 2
        mse = ((y_in[1:,-9:] - y_target[-9:]) ** 2).squeeze()
        # weighted mse
        wmse = torch.matmul(mse,self.weights.T)
        return wmse.mean()