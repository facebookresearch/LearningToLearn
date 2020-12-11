# Copyright (c) Facebook, Inc. and its affiliates.

import torch
import numpy as np
import sys
import os
from ml3.optimizee import ShapedSineModel
from ml3.ml3_train import meta_train_shaped_sine as meta_train
from ml3.learnable_losses import Ml3_loss_shaped_sine as Ml3_loss
from ml3.ml3_test import test_ml3_loss_shaped_sine as test_ml3_loss

class Task_loss(object):
    def __call__(self, input,outputs,labels,shaped,new_theta,label_thetas):
        if shaped:
            loss = (new_theta - label_thetas) ** 2
        else:
            loss = (outputs - labels) ** 2
        return loss



def generate_sinusoid_batch(num_tasks, num_examples_task, num_steps, random_steps=False,
                            freq_range=[-5.0, 5.0], input_range=[-5.0, 5.0]):
    """ Generate samples from random sine functions. """
    freq = np.random.uniform(freq_range[0], freq_range[1], [num_tasks])
    outputs = np.zeros([num_tasks, num_steps, num_examples_task, 1])
    thetas = np.zeros([num_tasks, num_steps, num_examples_task, 1])
    init_inputs = np.zeros([num_tasks, num_steps, num_examples_task, 1])

    for task in range(num_tasks):
        if random_steps:
            init_inputs[task] = np.random.uniform(input_range[0], input_range[1],
                                                  [num_steps, num_examples_task, 1])
        else:
            init_inputs[task] = np.repeat(np.random.uniform(input_range[0], input_range[1],
                                                            [1, num_examples_task, 1]), num_steps, -3)

        outputs[task] = np.sin(freq[task]*init_inputs[task])
        thetas[task] = np.zeros_like(outputs[task]) + freq[task]
    return init_inputs, outputs,thetas





if __name__ == '__main__':

    if not os.path.isdir("./data"):
        os.mkdir("./data")

    shaped = sys.argv[2]=='True'
    torch.manual_seed(0)
    np.random.seed(0)

    n_outer_iter=1000
    num_task = 4
    n_inner_iter = 10
    batch_size = 64

    ml3_loss = Ml3_loss()
    sine_model=ShapedSineModel()
    torch.save(sine_model.state_dict(), 'data/shaped_sine_init_policy.pt')
    sine_model.load_state_dict(torch.load('data/shaped_sine_init_policy.pt'))
    sine_model.eval()

    # initialize task loss for meta training
    task_loss_fn = Task_loss()

    if sys.argv[1] == 'train':

        #batch_inputs, batch_labels, batch_thetas = generate_sinusoid_batch(num_task, batch_size, n_inner_iter)
        meta_train(n_outer_iter, shaped, num_task, n_inner_iter, sine_model, ml3_loss,task_loss_fn)



    if sys.argv[1] == 'test':
        freq=0.7
        test_x = np.expand_dims(np.arange(-5.0,5.0,0.1),1)
        test_y = np.sin(freq*test_x)
        x = torch.Tensor(test_x)
        y = torch.Tensor(test_y)

        ml3_loss.model.load_state_dict(torch.load('data/ml3_loss_shaped_sine_'+str(shaped)+'.pt'))
        ml3_loss.model.eval()
        opt_iter = 1
        args = (torch.Tensor(test_x),torch.Tensor(test_y))
        test_ml3_loss(sine_model, ml3_loss,opt_iter,*args)








