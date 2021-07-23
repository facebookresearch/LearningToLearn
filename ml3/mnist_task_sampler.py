import numpy as np
import torch
from torchvision import datasets, transforms


class SineCosineTaskSampler(object):
    """
    adapted from Artem's code
    """
    def __init__(self, num_tasks_total, num_tasks_per_batch, num_data_points,
                 input_range=[-5.0, 5.0],
                 amp_range=[1.0, 1.0],
                 freq_range=[1.0, 1.0],
                 phase_range=[np.pi, np.pi],
                 fun_type="sine", device="cpu"):

        self.input_range = input_range

        self.amp_range = amp_range
        self.freq_range = freq_range
        self.phase_range = phase_range

        self.fun_type = fun_type

        self.observation_space = np.ones([1], dtype=np.float32)
        self.action_space = np.ones([1], dtype=np.float32)
        self.sample_space = np.ones([1], dtype=np.float32)

        self.num_tasks_total = num_tasks_total
        self.num_tasks_per_task = num_tasks_per_batch
        self.train_tasks = self._sample_tasks(num_tasks_total, num_data_points)
        self.valid_tasks = self._sample_tasks(num_tasks_total, num_data_points)

    def _sample_tasks(self, num_tasks, n_data_points):
        """
        Returns a list of task parameters
        """
        amp = np.random.uniform(self.amp_range[0], self.amp_range[1], [num_tasks]).astype(np.float32)
        freq = np.random.uniform(self.freq_range[0], self.freq_range[1], [num_tasks]).astype(np.float32)
        phase = np.random.uniform(self.phase_range[0], self.phase_range[1], [num_tasks]).astype(np.float32)
        inputs = np.random.uniform(self.input_range[0], self.input_range[1], [num_tasks, n_data_points, 1]).astype(np.float32)

        return [[amp[i], freq[i], phase[i], inputs[i]] for i in range(num_tasks)]

    def _sample_from_tasks(self, tasks):
        task_idx = np.random.permutation(self.num_tasks_total)[:self.num_tasks_per_task]
        inputs, targets = [], []
        for i in task_idx:
            task_params = tasks[i]
            inputs_np = task_params[3]
            targets_np = (task_params[0] * np.sin(task_params[1] * (inputs_np - task_params[2]))).astype(np.float32)
            inputs.append(torch.FloatTensor(inputs_np))
            targets.append(torch.FloatTensor(targets_np))
        return inputs, targets

    def sample(self):
        """
        Samples from a single task
        """
        ## [traj_len=1, batch_size, obs_shape=1]
        train_inputs, train_targets = self._sample_from_tasks(self.train_tasks)
        valid_inputs, valid_targets = self._sample_from_tasks(self.valid_tasks)

        return train_inputs, train_targets, valid_inputs, valid_targets

SHARD_NAME_TRAIN = 'train'
SHARD_NAME_VALIDATION = 'validation'
SHARD_NAME_TEST = 'test'


class Mnist():
    """Just the standard mnist example.
    """

    def __init__(self, batch_size, shard_name, shuffle, data_dir):
        self._data_dir = data_dir
        self._shard_name = shard_name
        self._batch_size = batch_size
        if shard_name == SHARD_NAME_TRAIN:
            self.dataset = torch.utils.data.DataLoader(
                datasets.MNIST(data_dir, train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                               ])),
                batch_size=batch_size, shuffle=shuffle)

        if shard_name == SHARD_NAME_TEST:
            self.dataset = torch.utils.data.DataLoader(
                datasets.MNIST(data_dir, train=False,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                               ])),
                batch_size=batch_size, shuffle=shuffle)

        self._shuffle = shuffle
        self._batch_size = batch_size

    @property
    def num_examples(self):
        if self._shard_name == SHARD_NAME_TRAIN:
            return len(self.dataset.data)
        if self._shard_name == SHARD_NAME_TEST:
            return len(self.dataset.data)

    def get_next_batch(self):
        if self._shard_name == SHARD_NAME_TRAIN:
            (images, labels) = self.dataset.__iter__().next()
            return {'x': images, 'y': labels}
        if self._shard_name == SHARD_NAME_TEST:
            (images, labels) = self.dataset.__iter__().next()
            return {'x': images, 'y': labels}


class MnistClassPair():

    def __init__(self, batch_size,
                 pos_class,
                 neg_class,
                 shard_name,
                 seed,
                 data_dir):

        shuffle = False
        if shard_name == SHARD_NAME_TRAIN:
            self.dataset = torch.utils.data.DataLoader(
                datasets.MNIST(data_dir, train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                               ])),
                batch_size=1, shuffle=shuffle)

        if shard_name == SHARD_NAME_TEST:
            self.dataset = torch.utils.data.DataLoader(
                datasets.MNIST(data_dir, train=False,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                               ])),
                batch_size=1, shuffle=shuffle)

        self._shard_name = shard_name
        self._shuffle = shuffle
        self._batch_size = batch_size
        self._pos_class = pos_class
        self._neg_class = neg_class

        if self._shard_name == SHARD_NAME_TRAIN:
            self._pos_idx = torch.nonzero(self.dataset.dataset.train_labels ==
                                          pos_class)
            self._neg_idx = torch.nonzero(self.dataset.dataset.train_labels ==
                                          neg_class)

        if shard_name == SHARD_NAME_TEST:
            self._pos_idx = torch.nonzero(self.dataset.dataset.test_labels ==
                                          pos_class)
            self._neg_idx = torch.nonzero(self.dataset.dataset.test_labels ==
                                          neg_class)

        idx = torch.cat([self._pos_idx, self._neg_idx], dim=0)
        torch.manual_seed(seed)

        rand_order = torch.randperm(len(idx))
        self._idx = idx[rand_order]
        self._step = 0

        if self._batch_size == -1:
            self._batch_size = len(self._idx)
            print("({}) Using all data per batch".format(shard_name))

    def num_examples(self):
        return len(self._idx)

    def get_next_batch(self):

        # idx = torch.randperm(len(self._idx))[:self._batch_size]
        if (self._step + 1) * self._batch_size > len(self._idx):
            # print "({}) cycled through data, restart".format(self._shard_name)
            self._step = 0

        start = self._step * self._batch_size
        end = (self._step + 1) * self._batch_size
        binary_idx = self._idx[start:end, 0]
        # print binary_idx
        # binary_idx = self._idx.index_select(dim=0, index=idx)#[:, 0]

        if self._shard_name == SHARD_NAME_TRAIN:
            x = self.dataset.dataset.train_data[binary_idx, :].float()
            yb = self.dataset.dataset.train_labels[binary_idx]
        if self._shard_name == SHARD_NAME_TEST:
            x = self.dataset.dataset.test_data[binary_idx, :].float()
            yb = self.dataset.dataset.test_labels[binary_idx]

        y_1hot = torch.FloatTensor(self._batch_size, 2).zero_()
        # transform y to 0 and 1, pos_class = 1, neg_class = 0
        y_1hot[yb == self._neg_class, 0] = 1.0
        y_1hot[yb == self._pos_class, 1] = 1.0

        y = torch.zeros_like(yb)
        y[yb == self._pos_class] = 1
        y[yb == self._neg_class] = 0

        x_shape = x.shape
        self._step += 1
        return {'x': x.view(x_shape[0], 1, x_shape[1], x_shape[2]), 'y_1hot': y_1hot, 'y': y}

    def reset(self):
        self._step = 0