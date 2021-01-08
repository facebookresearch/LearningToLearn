# Copyright (c) Facebook, Inc. and its affiliates.
import torch


# The learned weighted cost, with fixed weights ###
class LearnableWeightedCost(torch.nn.Module):
    def __init__(self, dim=9, weights=None):
        super(LearnableWeightedCost, self).__init__()
        if weights is None:
            self.weights = torch.nn.Parameter(0.1 * torch.ones([dim, 1]))
        else:
            self.weights = weights
        self.clip = torch.nn.ReLU()
        self.meta_grads = [[] for _, _ in enumerate(self.parameters())]

    def forward(self, y_in, y_target):
        assert y_in.dim() == 2
        mse = ((y_in[:,-9:] - y_target[-9:]) ** 2).squeeze()

        # weighted mse
        wmse = torch.mm(mse, self.clip(self.weights))
        return wmse.mean()


# The learned weighted cost, with time dependent weights ###
class LearnableTimeDepWeightedCost(torch.nn.Module):
    def __init__(self, dim=9, weights=None):
        super(LearnableTimeDepWeightedCost, self).__init__()
        if weights is None:
            self.weights = torch.nn.Parameter(0.1 * torch.ones([10, dim]))
        else:
            self.weights = weights
        self.clip = torch.nn.ReLU()
        self.meta_grads = [[] for _, _ in enumerate(self.parameters())]

    def forward(self, y_in, y_target):
        assert y_in.dim() == 2
        mse = ((y_in[:,-9:] - y_target[-9:]) ** 2).squeeze()
        # weighted mse
        #wmse = torch.matmul(mse,self.weights.T)
        wmse = mse * self.clip(self.weights)
        return wmse.mean()


class RBFWeights(torch.nn.Module):

    def __init__(self, dim, width, weights=None):
        super(RBFWeights, self).__init__()
        k_list = torch.linspace(0, 5, 10)[1::2]
        if weights is None:
            self.weights = torch.nn.Parameter(0.1 * torch.ones(len(k_list), dim))
        else:
            self.weights = weights

        x = torch.arange(0, 10)
        self.K = torch.stack([torch.exp(-(int(k) - x) ** 2 / width) for k in k_list]).T
        print(f"\nRBFWEIGHTS: {k_list}")

        self.clip = torch.nn.ReLU()

    def forward(self):
        return self.K.matmul(self.clip(self.weights))


class LearnableRBFWeightedCost(torch.nn.Module):
    def __init__(self, dim=9, width=2.0, weights=None):
        super(LearnableRBFWeightedCost, self).__init__()
        self.dim = dim
        self.weights_fn = RBFWeights(dim=dim, width=width, weights=weights)

    def forward(self, y_in, y_target):
        assert y_in.dim() == 2
        mse = ((y_in[:, -9:] - y_target[-9:]) ** 2).squeeze()

        self.weights = self.weights_fn()
        wmse = self.weights * mse

        return wmse.mean()


class BaselineCost(object):
    def __init__(self, weights):
        self.weights = weights

    def __call__(self, y_in, y_target):
        assert y_in.dim() == 2
        mse = ((y_in[:, -9:] - y_target[-9:]) ** 2).squeeze()

        # weighted mse
        wmse = mse * self.weights
        return wmse.mean()