# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class MountainCar():
    def __init__(self):
        self.m = 0.2
        self.g = -9.8
        self.k = 0.3
        self.max_position = 0.5
        self.min_position = -1.2
        self.max_speed = 1.5
        self.min_speed = -1.5
        self.delta_t = 0.1
        self.min_action = -0.25
        self.max_action = 0.25
        self.cur_pos = 0.0
        self.cur_vel = 0.0

    def sim_step_torch(self, state, action):

        position = state[0]
        velocity = state[1]

        action = torch.clamp(action, min=self.min_action, max=self.max_action)

        velocity = velocity + (self.g * self.m * torch.cos(3.0 * position) + (action / self.m) - (
                    self.k * velocity)) * self.delta_t
        position = position + (velocity * self.delta_t)

        if (velocity.data > self.max_speed): velocity.data = torch.Tensor([self.max_speed])
        if (velocity.data < -self.max_speed): velocity.data = torch.Tensor([-self.max_speed])

        if (position.data >= self.max_position): position.data = torch.Tensor([self.max_position])
        if (position.data < self.min_position): position.data = torch.Tensor([self.min_position])
        if (position.data == self.min_position and velocity.data < 0): velocity.data = torch.Tensor([0.0])

        new_state = torch.stack([position.squeeze(), velocity.squeeze()])
        return new_state

    def sim_step(self, state, action):
        position = state[0]
        velocity = state[1]

        velocity = velocity + (self.g * self.m * np.cos(3 * position) + (action / self.m) - (self.k * velocity)) * self.delta_t
        position = position + (velocity * self.delta_t)

        if (velocity > self.max_speed): velocity = self.max_speed
        if (velocity < -self.max_speed): velocity = -self.max_speed

        if (position > self.max_position): position = self.max_position
        if (position < self.min_position): position = self.min_position
        if (position==self.min_position and velocity<0): velocity = 0

        new_state = np.array([position, velocity])
        return new_state.squeeze()

    def step(self, action):
        position = self.cur_pos
        velocity = self.cur_vel

        velocity = velocity + (
                    self.g * self.m * np.cos(3 * position) + (action / self.m) - (self.k * velocity)) * self.delta_t
        position = position + (velocity * self.delta_t)

        if (velocity > self.max_speed): velocity = self.max_speed
        if (velocity < -self.max_speed): velocity = -self.max_speed

        if (position > self.max_position): position = self.max_position
        if (position < self.min_position): position = self.min_position
        if (position == self.min_position and velocity < 0): velocity = 0

        new_state = np.array([position, velocity])
        self.cur_pos = position
        self.cur_vel = velocity
        reward = 0
        if new_state[0] >= 0.5:
            reward = 100
        return np.array([self.cur_pos, self.cur_vel]), reward

    def reset(self):
        self.cur_pos = -0.55
        self.cur_vel = 0
        return np.array([self.cur_pos, self.cur_vel])

    def reset_to(self, state):
        self.cur_pos = state[0]
        self.cur_vel = state[1]
        return np.array([self.cur_pos, self.cur_vel])

    def render(self, position_list, file_path='./mountain_car.gif', mode='gif'):
        """ When the method is called it saves an animation
        of what happened until that point in the episode.
        Ideally it should be called at the end of the episode,
        and every k episodes.

        ATTENTION: It requires avconv and/or imagemagick installed.
        @param file_path: the name and path of the video file
        @param mode: the file can be saved as 'gif' or 'mp4'
        """

        # Plot init
        fig = plt.figure(figsize=(4,4))
        ax = fig.add_subplot(111, autoscale_on=False, xlim=(-1.3, 0.6), ylim=(-1.2, 1.5))
        ax.grid(False)  # disable the grid
        x_sin = np.linspace(start=-1.2, stop=0.5, num=100)
        y_sin = np.sin(3 * x_sin)
        ax.plot(x_sin, y_sin,c='black',linewidth=3)  # plot the sine wave
        ax.plot(0.50, 1.16, marker="$\u2691$", markersize=25, color='green')

        dot, = ax.plot([], [],marker="$\u25A1$",markersize=15,color='red')
        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
        _position_list = position_list
        _delta_t = self.delta_t

        def _init():
            dot.set_data([], [])
            time_text.set_text('')
            return dot, time_text

        def _animate(i):
            x = _position_list[i]
            y = np.sin(3 * x)
            dot.set_data(x, y)
            time_text.set_text("")
            return dot, time_text

        ani = animation.FuncAnimation(fig, _animate, np.arange(1, len(position_list)),
                                      blit=True, init_func=_init, repeat=False)

        if mode == 'gif':
            ani.save(file_path, writer='imagemagick', fps=int(1 / self.delta_t))
        elif mode == 'mp4':
            ani.save(file_path, fps=int(1 / self.delta_t), writer='avconv', codec='libx264')
        # Clear the figure
        fig.clear()
        plt.close(fig)