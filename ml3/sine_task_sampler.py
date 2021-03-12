# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import torch


class SineTaskSampler(object):
    def __init__(self, num_tasks_total, num_tasks_per_batch, num_data_points,
                 input_range=[-5.0, 5.0],
                 amp_range=[1.0, 1.0],
                 freq_range=[1.0, 1.0],
                 phase_range=[np.pi, np.pi],
                 fun_type="sine"):

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

