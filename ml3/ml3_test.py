# Copyright (c) Facebook, Inc. and its affiliates.
import torch
import matplotlib.pyplot as plt


def test_ml3_loss_mountain_car(policy,ml3_loss,opt_iter,*args):

    opt = torch.optim.SGD(policy.parameters(), lr=policy.learning_rate)
    for i in range(opt_iter):
        s_tr, a_tr, g_tr = policy.roll_out(*args)
        pred_task_loss = ml3_loss(torch.cat((torch.cat((s_tr[:-1], a_tr), dim=1), g_tr), dim=1)).mean()
        opt.zero_grad()
        pred_task_loss.backward()
        opt.step()
        s_tr, a_tr, g_tr = policy.roll_out(*args)
        print('last state: ', s_tr[-1])
    return s_tr.detach().numpy()


def test_ml3_loss_reacher(policy, ml3_loss, opt_iter, *args):
    opt = torch.optim.SGD(policy.parameters(), lr=policy.learning_rate)
    for i in range(opt_iter):
        s_tr, a_tr, g_tr = policy.roll_out(*args)
        meta_input = torch.cat((torch.cat((s_tr[:-1], a_tr), dim=1), g_tr), dim=1)
        pred_task_loss = ml3_loss(meta_input).mean()
        opt.zero_grad()
        pred_task_loss.backward()
        opt.step()
    return s_tr.detach().numpy()


def test_ml3_loss_shaped_sine(sine_model,ml3_loss,opt_iter,test_x,test_y):
    opt = torch.optim.SGD(sine_model.parameters(), lr=sine_model.learning_rate)
    for i in range(opt_iter):
        yp = sine_model(test_x)
        meta_input = torch.cat((torch.cat((test_x, yp), 1),test_y), 1)
        pred_task_loss = ml3_loss.policy_params(meta_input).mean()
        opt.zero_grad()
        pred_task_loss.backward()
        opt.step()
        yp = sine_model(test_x)
        print('last state: ', yp[-1])
        print('label: ',test_y[-1])
        plt.plot(yp.detach().numpy())
        plt.plot(test_y.detach().numpy())
        plt.show()