# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
from termcolor import colored
import logging
import torch.nn as nn
import torch.utils.data


log = logging.getLogger(__name__)

import torch
import numpy as np
import math


class Dataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.dataset = [
            (torch.FloatTensor(x[i]), torch.FloatTensor(y[i])) for i in range(len(x))
        ]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

class Dynamics(nn.Module):

    def __init__(self,env):
        super(Dynamics, self).__init__()

        self.env=env
        self.dt = env.dt

        self.model_cfg = {}
        self.model_cfg['device'] = 'cpu'
        self.model_cfg['hidden_size'] = [100, 30]
        self.model_cfg['batch_size'] = 128
        self.model_cfg['epochs'] = 500
        self.model_cfg['display_epoch'] = 50
        self.model_cfg['learning_rate'] = 0.001
        self.model_cfg['ensemble_size'] = 3
        self.model_cfg['state_dim'] = env.state_dim
        self.model_cfg['action_dim'] = env.action_dim
        self.model_cfg['output_dim'] = env.pos_dim

        self.ensemble = EnsembleProbabilisticModel(self.model_cfg)

        self.data_X = []
        self.data_Y = []
        self.norm_in = torch.Tensor(np.expand_dims(np.array([1.0,1.0,8.0,8.0,1.0,1.0]),axis=0))



    def train(self,states,actions):

        inputs = (torch.cat((states[:-1],actions),dim=1)/self.norm_in).detach().numpy()
        outputs = (states[1:,self.env.pos_dim:] - states[:-1,self.env.pos_dim:]).detach().numpy()

        self.data_X+=list(inputs)
        self.data_Y+=list(outputs)

        training_dataset = {}
        training_dataset['X'] = np.array(self.data_X)
        training_dataset['Y'] = np.array(self.data_Y)

        #self.ensemble = EnsembleProbabilisticModel(self.model_cfg)
        self.ensemble.train_model(training_dataset, training_dataset, 0.0)

    def step_model(self,state,action):
        input_x = torch.cat((state,action),dim=0)/self.norm_in
        pred_acc = self.ensemble.forward(input_x)[0].squeeze()

        #numerically integrate predicted acceleration to velocity and position

        pred_vel = state[self.env.pos_dim:]+pred_acc
        pred_pos = state[:self.env.pos_dim] + pred_vel*self.dt
        pred_pos = torch.clamp(pred_pos, min=-3.0, max=3.0)
        pred_vel = torch.clamp(pred_vel, min=-4.0, max=4.0)
        next_state = torch.cat((pred_pos.squeeze(),pred_vel.squeeze()),dim=0)
        return next_state.squeeze()



# I did not make this inherit from nn.Module, because our GP implementation is not torch based
class AbstractModel(object):

    # def forward(self, x):
    #    raise NotImplementedError("Subclass must implement")

    def train_model(self, training_dataset, testing_dataset, training_params):
        raise NotImplementedError("Subclass must implement")

    # function that (if necessary) converts between numpy input x and torch, and returns a prediction in numpy
    def predict_np(self, x):
        raise NotImplementedError("Subclass must implement")

    def get_input_size(self):
        raise NotImplementedError("Subclass must implement")

    def get_output_size(self):
        raise NotImplementedError("Subclass must implement")

    def get_hyperparameters(self):
        return None



class Dataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.dataset = [
            (torch.FloatTensor(x[i]), torch.FloatTensor(y[i])) for i in range(len(x))
        ]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


# creates K datasets out of X and Y
# if N is the total number of data points, then this function splits it in to K subsets. and each dataset contains K-1
# subsets.
# so let's say K=5. We create 5 subsets.
# Each datasets contains 4 out of the 5 datasets, by leaving out one of the K subsets.
def split_to_subsets(X, Y, K):
    if K == 1:
        # for 1 split, do not resshuffle dataset
        return [Dataset(X, Y)]

    n_data = len(X)
    chunk_sz = int(math.ceil(n_data / K))
    all_idx = np.random.permutation(n_data)

    datasets = []
    # each dataset contains
    for i in range(K):
        start_idx = i * (chunk_sz)
        end_idx = min(start_idx + chunk_sz, n_data)
        dataset_idx = np.delete(all_idx, range(start_idx, end_idx), axis=0)
        X_subset = [X[idx] for idx in dataset_idx]
        Y_subset = [Y[idx] for idx in dataset_idx]
        datasets.append(Dataset(X_subset, Y_subset))

    return datasets


class NLLLoss(torch.nn.modules.loss._Loss):
    """
    Specialized NLL loss used to predict both mean (the actual function) and the variance of the input data.
    """

    def __init__(self, size_average=None, reduce=None, reduction="mean"):
        super(NLLLoss, self).__init__(size_average, reduce, reduction)

    def forward(self, net_output, target):
        assert net_output.dim() == 3
        assert net_output.size(0) == 2
        mean = net_output[0]
        var = net_output[1]
        reduction = "mean"
        ret = 0.5 * torch.log(var) + 0.5 * ((mean - target) ** 2) / var
        # ret = 0.5 * ((mean - target) ** 2)

        if reduction != "none":
            ret = torch.mean(ret) if reduction == "mean" else torch.sum(ret)
        return ret

class EnsembleProbabilisticModel(AbstractModel):
    def __init__(self, model_cfg):
        super(EnsembleProbabilisticModel, self).__init__()

        self.input_dimension = model_cfg['state_dim'] + model_cfg['action_dim']
        # predicting velocity only (second half of state space)
        assert model_cfg['state_dim'] % 2 == 0
        self.output_dimension = model_cfg['state_dim'] // 2
        if model_cfg['device'] == "gpu":
            self.device = model_cfg['gpu_name']
        else:
            self.device = "cpu"
        self.ensemble_size = model_cfg['ensemble_size']
        self.model_cfg = model_cfg

        self.reset()

    def reset(self):
        self.models = [PModel(self.model_cfg) for _ in range(self.ensemble_size)]

    def forward(self, x):
        x = torch.Tensor(x)
        means = []
        variances = []
        for eid in range(self.ensemble_size):
            mean_and_var = self.models[eid](x)
            means.append(mean_and_var[0])
            variances.append(mean_and_var[1])

        mean = sum(means) / len(means)
        dum = torch.zeros_like(variances[0])
        for i in range(len(means)):
            dum_var2 = variances[i]
            dum_mean2 = means[i] * means[i]
            dum += dum_var2 + dum_mean2

        var = (dum / len(means)) - (mean * mean)
        # Clipping the variance to a minimum of 1e-3, we can interpret this as saying weexpect a minimum
        # level of noise
        # the clipping here is probably not necessary anymore because we're now clipping at the individual model level
        var = var.clamp_min(1e-3)
        return torch.stack((mean, var))

    def predict_np(self, x_np):
        x = torch.Tensor(x_np)
        pred = self.forward(x).detach().cpu().numpy()
        return pred[0].squeeze(), pred[1].squeeze()

    def train_model(self, training_dataset, testing_dataset, training_params):
        X = training_dataset["X"]
        Y = training_dataset["Y"]

        datasets = split_to_subsets(X, Y, self.ensemble_size)

        for m in range(self.ensemble_size):
            print(colored("training model={}".format(m), "green"))
            self.models[m].train_model(datasets[m])

    def get_gradient(self, x_np):

        x = torch.Tensor(x_np).requires_grad_()
        output_mean, _ = self.forward(x)
        gradients = []
        # get gradients of ENN with respect to x and u
        for output_dim in range(self.output_dimension):
            grads = torch.autograd.grad(
                output_mean[0, output_dim], x, create_graph=True
            )[0].data
            gradients.append(grads.detach().cpu().numpy()[0, :])

        return np.array(gradients).reshape(
            [self.output_dimension, self.input_dimension]
        )

    def get_input_size(self):
        return self.input_dimension

    def get_output_size(self):
        return self.output_dimension

    def get_hyper_params(self):
        return None


class PModel(nn.Module):
    """
    Probabilistic network
    Output a 3d tensor:
    d0 : always 2, first element is mean and second element is variance
    d1 : batch size
    d2 : output size (number of dimensions in the output of the modeled function)
    """

    def __init__(self, config):
        super(PModel, self).__init__()
        if config["device"] == "gpu":
            self.device = config["gpu_name"]
        else:
            self.device = "cpu"
        self.input_sz = config['state_dim'] + config['action_dim']
        self.output_sz = config['output_dim']

        self.learning_rate = config["learning_rate"]
        self.display_epoch = config["display_epoch"]
        self.epochs = config["epochs"]

        w = config["hidden_size"]

        self.layers = nn.Sequential(
            nn.Linear(self.input_sz, w[0]),
            nn.Tanh(),
            nn.Linear(w[0], w[1]),
            nn.Tanh(),
        )

        self.mean = nn.Linear(w[1], self.output_sz)
        self.var = nn.Sequential(nn.Linear(w[1], self.output_sz), nn.Softplus())
        self.to(self.device)

    def forward(self, x):
        x = x.to(device=self.device)
        assert x.dim() == 2, "Expected 2 dimensional input, got {}".format(x.dim())
        assert x.size(1) == self.input_sz
        y = self.layers(x)
        mean_p = self.mean(y)
        var_p = self.var(y)
        # Clipping the variance to a minimum of 1e-3, we can interpret this as saying weexpect a minimum
        # level of noise
        var_p = var_p.clamp_min(1e-3)
        return torch.stack((mean_p, var_p))

    def predict_np(self, x_np):
        x = torch.Tensor(x_np)
        pred = self.forward(x).detach().cpu().numpy()
        return pred[0].squeeze(), pred[1].squeeze()

    def train_model(self, training_data):
        train_loader = torch.utils.data.DataLoader(
            training_data, batch_size=64, num_workers=0
        )
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_fn = NLLLoss()
        for epoch in range(self.epochs):
            losses = []
            for batch, (data, target) in enumerate(
                train_loader, 1
            ):  # This is the training loader
                x = data.type(torch.FloatTensor).to(device=self.device)
                y = target.type(torch.FloatTensor).to(device=self.device)

                if x.dim() == 1:
                    x = x.unsqueeze(0).t()
                if y.dim() == 1:
                    y = y.unsqueeze(0).t()

                py = self.forward(x)
                loss = loss_fn(py, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            if epoch % self.display_epoch == 0:
                print(
                    colored(
                        "epoch={}, loss={}".format(epoch, np.mean(losses)), "yellow"
                    )
                )