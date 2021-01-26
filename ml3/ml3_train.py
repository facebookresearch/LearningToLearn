# Copyright (c) Facebook, Inc. and its affiliates.
import torch
import higher
import numpy as np
from ml3.shaped_sine_utils import plot_loss, generate_sinusoid_batch
from ml3.optimizee import ShapedSineModel


def meta_train_mountain_car(policy,ml3_loss,task_loss_fn,s_0,goal,goal_extra,n_outer_iter,n_inner_iter,time_horizon,shaped_loss):
    s_0 = torch.Tensor(s_0)
    goal = torch.Tensor(goal)
    goal_extra = torch.Tensor(goal_extra)

    inner_opt = torch.optim.SGD(policy.policy_params.parameters(), lr=policy.learning_rate)
    meta_opt = torch.optim.Adam(ml3_loss.policy_params.parameters(), lr=ml3_loss.learning_rate)

    for outer_i in range(n_outer_iter):
        # set gradient with respect to meta loss parameters to 0
        meta_opt.zero_grad()

        with higher.innerloop_ctx(
                policy.policy_params, inner_opt, copy_initial_weights=False) as (fpolicy, diffopt):
            for _ in range(n_inner_iter):
                # use current meta loss to update model
                s_tr, a_tr, g_tr  = policy.roll_out(fpolicy,s_0,goal,time_horizon)

                pred_task_loss = ml3_loss.policy_params(torch.cat((torch.cat((s_tr[:-1], a_tr), dim=1), g_tr), dim=1)).mean()
                diffopt.step(pred_task_loss)

            # compute task loss
            s, a, g = policy.roll_out(fpolicy,s_0,goal,time_horizon)
            task_loss = task_loss_fn(a,s[:], goal,goal_extra,shaped_loss)
            # backprop grad wrt to task loss
            task_loss.backward()

        meta_opt.step()

        if outer_i%100==0:
            print("meta iter: {} loss: {}".format(outer_i,task_loss.item()))
            print('last state',s[-1])


def meta_train_mbrl_reacher(policy, ml3_loss, dmodel, env, task_loss_fn, goals, n_outer_iter, n_inner_iter, time_horizon, exp_folder):
    goals = torch.Tensor(goals)

    meta_opt = torch.optim.Adam(ml3_loss.parameters(), lr=ml3_loss.learning_rate)

    for outer_i in range(n_outer_iter):
        # set gradient with respect to meta loss parameters to 0
        meta_opt.zero_grad()
        all_loss = 0
        for goal in goals:
            goal = torch.Tensor(goal)

            # todo: replace with reset function, instead of loading initial policy
            policy.load_state_dict(torch.load(f'{exp_folder}/init_policy.pt'))
            inner_opt = torch.optim.SGD(policy.parameters(), lr=policy.learning_rate)
            for _ in range(n_inner_iter):
                inner_opt.zero_grad()
                # todo: there used to be a policy reset (of the gradients) here, is that necessary?
                with higher.innerloop_ctx(policy, inner_opt, copy_initial_weights=False) as (fpolicy, diffopt):
                    # use current meta loss to update model
                    s_tr, a_tr, g_tr = fpolicy.roll_out(goal, time_horizon, dmodel, env)
                    # todo: the normalization should happen in the loss fun
                    meta_input = torch.cat((torch.cat((s_tr[:-1].detach(),a_tr),dim=1), g_tr.detach()),dim=1)/ml3_loss.norm_in
                    pred_task_loss = ml3_loss(meta_input).mean()
                    diffopt.step(pred_task_loss)
                    # compute task loss
                    s, a, g = fpolicy.roll_out(goal, time_horizon, dmodel, env)
                    task_loss = task_loss_fn(a, s[:], goal).mean()

            # collect losses for logging
            all_loss += task_loss
            # backprop grad wrt to task loss
            task_loss.backward()

            if outer_i % 100 == 0:
                # roll out in real environment, to monitor training and tp collect data for dynamics model update
                states, actions, _ = fpolicy.roll_out(goal, time_horizon, dmodel, env, real_rollout=True)
                print("meta iter: {} loss: {}".format(outer_i, (torch.mean((states[-1,:2]-goal[:2])**2))))
                if outer_i % 300 == 0 and outer_i < 3001:
                    # update dynamics model under current optimal policy
                    dmodel.train(torch.Tensor(states), torch.Tensor(actions))

        # step optimizer to update meta loss network
        meta_opt.step()
        torch.save(ml3_loss.state_dict(), f'{exp_folder}/ml3_loss_reacher.pt')


def meta_train_shaped_sine(n_outer_iter,shaped,num_task,n_inner_iter,sine_model,ml3_loss,task_loss_fn, exp_folder):
    theta_ranges = []
    landscape_with_extra = []
    landscape_mse = []

    meta_opt = torch.optim.Adam(ml3_loss.policy_params.parameters(), lr=ml3_loss.learning_rate)

    for outer_i in range(n_outer_iter):
        # set gradient with respect to meta loss parameters to 0
        batch_inputs, batch_labels, batch_thetas = generate_sinusoid_batch(num_task, 64, n_inner_iter)
        for task in range(num_task):
            sine_model = ShapedSineModel()
            inner_opt = torch.optim.SGD([sine_model.freq], lr=sine_model.learning_rate)
            for step in range(n_inner_iter):
                inputs = torch.Tensor(batch_inputs[task, step, :])
                labels = torch.Tensor(batch_labels[task, step, :])
                label_thetas = torch.Tensor(batch_thetas[task, step, :])

                ''' Updating the frequency parameters, taking gradien of theta wrt meta loss '''

                with higher.innerloop_ctx(sine_model, inner_opt) as (fmodel, diffopt):
                    # use current meta loss to update model
                    yp = fmodel(inputs)
                    meta_input = torch.cat((torch.cat((inputs, yp), 1), labels), 1)

                    meta_out = ml3_loss.policy_params(meta_input)
                    loss = meta_out.mean()
                    diffopt.step(loss)

                    yp = fmodel(inputs)
                    task_loss = task_loss_fn(inputs, yp, labels, shaped, fmodel.freq, label_thetas)

                sine_model.freq = torch.nn.Parameter(fmodel.freq.clone().detach())
                inner_opt = torch.optim.SGD([sine_model.freq], lr=sine_model.learning_rate)

                ''' updating the meta network '''
                ml3_loss.policy_params.zero_grad()
                meta_opt.zero_grad()
                task_loss.mean().backward()
                meta_opt.step()

        if outer_i%100==0:
            print("task loss: {}".format(task_loss.mean().item()))

        torch.save(ml3_loss.policy_params.state_dict(), f'{exp_folder}/ml3_loss_shaped_sine_' + str(shaped) + '.pt')

        if outer_i%10==0:
            t_range, l_with_extra, l_mse = plot_loss(shaped, exp_folder)
            theta_ranges.append(t_range)
            landscape_with_extra.append(l_with_extra)
            landscape_mse.append(l_mse)
    np.save(f'{exp_folder}/theta_ranges_'+str(shaped)+'_.npy', theta_ranges)
    np.save(f'{exp_folder}/landscape_with_extra_'+str(shaped)+'_.npy',landscape_with_extra)
    np.save(f'{exp_folder}/landscape_mse_'+str(shaped)+'_.npy',landscape_mse)

