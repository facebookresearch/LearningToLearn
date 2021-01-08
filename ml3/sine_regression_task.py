
import dill as pickle
import os

import numpy as np
import torch.nn as nn
import torch
import higher

from ml3.optimizee import SineModel
from ml3.sine_task_sampler import SineTaskSampler
from ml3.learnable_losses import ML3_SineRegressionLoss


def regular_train(loss_fn, eval_loss_fn, task_model, x_tr, y_tr, exp_cfg):
    n_iter = exp_cfg['n_gradient_steps_at_test']
    lr = exp_cfg['inner_lr']

    loss_trace = []

    optimizer = torch.optim.SGD(task_model.parameters(), lr=lr)
    for i in range(n_iter):
        optimizer.zero_grad()
        y_pred = task_model(x_tr)
        loss = loss_fn(y_pred, y_tr)

        loss.backward()
        optimizer.step()

        eval_loss = eval_loss_fn(y_pred, y_tr)
        loss_trace.append(eval_loss.item())

    return loss_trace


def meta_train(meta_loss_model, meta_optimizer, meta_objective, task_sampler_train, task_sampler_test, exp_cfg):

    num_tasks = exp_cfg['num_train_tasks']
    n_outer_iter= exp_cfg['n_outer_iter']
    inner_lr = exp_cfg['inner_lr']

    results = []

    task_models = []
    task_opts = []
    for i in range(num_tasks):
        task_models.append(SineModel(in_dim=exp_cfg['model']['in_dim'],
                                     hidden_dim=exp_cfg['model']['hidden_dim'],
                                     out_dim=1))
        task_opts.append(torch.optim.SGD(task_models[i].parameters(), lr=inner_lr))

    for outer_i in range(n_outer_iter):
        # Sample a batch of support and query images and labels.

        x_spt, y_spt, x_qry, y_qry = task_sampler_train.sample()

        for i in range(num_tasks):
            task_models[i].reset()

        qry_losses = []
        for _ in range(1):
            pred_losses = []
            meta_optimizer.zero_grad()

            for i in range(num_tasks):
                # zero gradients wrt to meta loss parameters
                with higher.innerloop_ctx(task_models[i], task_opts[i],
                                          copy_initial_weights=False) as (fmodel, diffopt):

                    # update model parameters via meta loss
                    yp = fmodel(x_spt[i])
                    pred_loss = meta_loss_model(yp, y_spt[i])
                    diffopt.step(pred_loss)

                    # compute task loss with new model
                    yp = fmodel(x_spt[i])
                    task_loss = meta_objective(yp, y_spt[i])

                    # this accumulates gradients wrt to meta parameters
                    task_loss.backward()
                    qry_losses.append(task_loss.item())

            meta_optimizer.step()
            # stepping the models forward so that meta optimizer sees different stages
            # for i in range(num_tasks):
            #     task_opts[i].zero_grad()
            #     yp, feat = task_models[i](x_spt[i])
            #     loss = meta_objective(yp, y_spt[i])
            #     loss.backward()
            #     task_opts[i].step()

        avg_qry_loss = sum(qry_losses) / num_tasks
        if outer_i % 10 == 0:
            res_train_eval_reg = eval(task_sampler=task_sampler_train, exp_cfg=exp_cfg,
                                      train_loss_fn=nn.MSELoss(), eval_loss_fn=nn.MSELoss())

            res_train_eval_ml3 = eval(task_sampler=task_sampler_train, exp_cfg=exp_cfg,
                                      train_loss_fn=meta_loss_model, eval_loss_fn=nn.MSELoss())

            res_test_eval_reg = eval(task_sampler=task_sampler_test, exp_cfg=exp_cfg,
                                     train_loss_fn=nn.MSELoss(), eval_loss_fn=nn.MSELoss())

            res_test_eval_ml3 = eval(task_sampler=task_sampler_test, exp_cfg=exp_cfg,
                                     train_loss_fn=meta_loss_model, eval_loss_fn=nn.MSELoss())

            res = {}
            res['train_reg'] = res_train_eval_reg
            res['train_ml3'] = res_train_eval_ml3
            res['test_reg'] = res_test_eval_reg
            res['test_ml3'] = res_test_eval_ml3
            res['task_loss'] = {}
            res['task_loss']['mse'] = qry_losses
            results.append(res)
            test_loss_ml3 = np.mean(res_test_eval_ml3['mse'])
            test_loss_reg = np.mean(res_test_eval_reg['mse'])
            print(
                f'[Epoch {outer_i:.2f}] Train Loss: {avg_qry_loss:.2f}]| Test Loss ML3: {test_loss_ml3:.2f} | TestLoss REG: {test_loss_reg:.2f}'
            )

    return results


def eval(task_sampler, exp_cfg, train_loss_fn, eval_loss_fn):
    seed = exp_cfg['seed']
    num_tasks = task_sampler.num_tasks_total

    np.random.seed(seed)
    torch.manual_seed(seed)

    mse = []
    nmse = []
    loss_trace = []
    x, y, _, _ = task_sampler.sample()
    for i in range(num_tasks):
        task_model_test = SineModel(in_dim=exp_cfg['model']['in_dim'],
                                    hidden_dim=exp_cfg['model']['hidden_dim'],
                                    out_dim=1)
        loss = regular_train(loss_fn=train_loss_fn, eval_loss_fn=eval_loss_fn, task_model=task_model_test,
                             x_tr=x[i], y_tr=y[i], exp_cfg=exp_cfg)
        yp = task_model_test(x[i])
        l = eval_loss_fn(yp, y[i])

        mse.append(l.item())
        nmse.append(l.item()/y[i].var())
        loss_trace.append(loss)

    res = {'nmse': nmse, 'mse': mse, 'loss_trace': loss_trace}
    return res


def main(exp_cfg):
    seed = exp_cfg['seed']
    num_train_tasks = exp_cfg['num_train_tasks']
    num_test_tasks = exp_cfg['num_test_tasks']
    outer_lr = exp_cfg['outer_lr']

    np.random.seed(seed)
    torch.manual_seed(seed)

    meta_loss_model = ML3_SineRegressionLoss(in_dim=exp_cfg['metaloss']['in_dim'],
                                             hidden_dim=exp_cfg['metaloss']['hidden_dim'])

    meta_optimizer = torch.optim.Adam(meta_loss_model.parameters(), lr=outer_lr)

    meta_objective = nn.MSELoss()

    task_sampler_train = SineTaskSampler(num_tasks_total=num_train_tasks, num_tasks_per_batch=num_train_tasks, num_data_points=100,
                                               amp_range=[1.0, 1.0],
                                               input_range=[-2.0, 2.0],
    )

    task_sampler_test = SineTaskSampler(num_tasks_total=num_test_tasks, num_tasks_per_batch=num_test_tasks, num_data_points=100,
                                              input_range=[-5.0, 5.0],
                                              amp_range=[0.2, 5.0],
                                              phase_range=[-np.pi, np.pi]
                                              )
#
    res = meta_train(meta_loss_model=meta_loss_model, meta_optimizer=meta_optimizer, meta_objective=meta_objective,
                     task_sampler_train=task_sampler_train, task_sampler_test=task_sampler_test,
                     exp_cfg=exp_cfg)

    pkl_file = os.path.join(exp_cfg['log_dir'], exp_cfg['exp_log_file_name'])

    pkl_dir = os.path.dirname(pkl_file)
    if pkl_dir is not '' and not os.path.exists(pkl_dir):  # Create directory if it doesn't exist.
        os.makedirs(pkl_dir)
    with open(pkl_file, 'wb') as fp:
        pickle.dump(res, fp, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    exp_cfg = {}
    exp_cfg['seed'] = 0
    exp_cfg['num_train_tasks'] = 1
    exp_cfg['num_test_tasks'] = 10
    exp_cfg['n_outer_iter'] = 500
    exp_cfg['n_gradient_steps_at_test'] = 100
    exp_cfg['inner_lr'] = 0.001
    exp_cfg['outer_lr'] = 0.001

    exp_cfg['model'] = {}
    exp_cfg['model']['in_dim'] = 1
    exp_cfg['model']['hidden_dim'] = [100, 10]

    exp_cfg['metaloss'] = {}
    exp_cfg['metaloss']['in_dim'] = 2
    exp_cfg['metaloss']['hidden_dim'] = [50, 50]

    model_arch_str = str(exp_cfg['model']['hidden_dim'])
    meta_arch_str = "{}".format(exp_cfg['metaloss']['hidden_dim'])
    exp_cfg['log_dir'] = "sin_cos_exp"
    exp_file = "sine_regression_seed_{}.pkl".format(exp_cfg['seed'])
    exp_cfg['exp_log_file_name'] = exp_file
    main(exp_cfg)
