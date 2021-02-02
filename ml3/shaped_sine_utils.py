# Copyright (c) Facebook, Inc. and its affiliates.
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from ml3.optimizee import ShapedSineModel
from ml3.learnable_losses import Ml3_loss_shaped_sine as MetaNetwork


'''GENERATE DATA'''
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

'''PLOTTING THE LOSS LANDSCAPES FOR ILLUSTRATION'''
def plot_loss(extra, exp_folder, freq=0.5):
    meta = MetaNetwork()
    meta.load_state_dict(torch.load(f'{exp_folder}/ml3_loss_shaped_sine_'+str(extra)+'.pt'))
    meta.eval()

    loss_landscape = []

    theta_ranges = np.arange(-7.0, 7.0, 0.1)
    test_x = np.expand_dims(np.arange(-5.0, 5.0, 0.1), 1)
    test_y = np.sin(freq * test_x)
    x = torch.Tensor(test_x)
    y = torch.Tensor(test_y)

    for theta in theta_ranges:
        pi = ShapedSineModel(theta)
        pi.learning_rate = 0.1
        pi_out = pi(x)
        loss = 0.5 * (pi_out - y) ** 2
        loss_landscape.append(loss.mean().detach().numpy())

    meta_loss_landscape = []
    for theta in theta_ranges:
        pi = ShapedSineModel(theta)
        policy_theta = torch.Tensor(np.zeros_like(test_y)) + pi.freq
        pi_out = pi(x)
        meta_input = torch.cat([x, pi_out, y], 1)
        loss = meta(meta_input).mean()
        meta_loss_landscape.append(loss.clone().mean().detach().numpy())

    return theta_ranges, np.array(meta_loss_landscape), np.array(loss_landscape)


def render(theta_ranges,loss,color,freq=0.5,file_path='./ml3_loss_sine.gif', mode='gif'):
    """ When the method is called it saves an animation
    of what happened until that point in the episode.
    Ideally it should be called at the end of the episode,
    and every k episodes.

    ATTENTION: It requires avconv and/or imagemagick installed.
    @param file_path: the name and path of the video file
    @param mode: the file can be saved as 'gif' or 'mp4'
    """

    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111,autoscale_on=False, xlim=(-7.0, 7.0), ylim=(0.0, 1.0))


    ax.axvline(x=freq, c='red')
    delta_t = 1.0/10.0
    dot, = ax.plot([], [],color=color)
    time_text = ax.text(0.25, 1.05, '', transform=ax.transAxes,fontsize=14)
    _theta_ranges = theta_ranges
    _loss = loss
    _delta_t = delta_t

    def _init():
        dot.set_data([], [])
        time_text.set_text('')
        return dot, time_text

    def _animate(i):
        x = _theta_ranges[i]
        y = _loss[i]
        dot.set_data(x, y)
        time_text.set_text("Iteration: "+str(i))
        return dot, time_text

    ani = animation.FuncAnimation(fig, _animate, np.arange(1, len(theta_ranges)),
                                  blit=True, init_func=_init, repeat=False)

    if mode == 'gif':
        ani.save(file_path, writer='imagemagick', fps=int(1 / delta_t))
    # Clear the figure
    fig.clear()
    plt.close(fig)