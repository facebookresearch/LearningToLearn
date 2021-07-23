
import dill as pickle
import os

import numpy as np
import torch.nn as nn
import torch
import higher

from ml3.learnable_losses import MnistLearnedLoss
from ml3.optimizee import LeNet
from ml3.mnist_task_sampler import MnistClassPair


def round2dec(arr, n_dec=2):
    return torch.round(arr * 10 ** n_dec) / (10 ** n_dec)


def regular_train(loss_fn, eval_loss_fn, task_model, task_sampler, exp_cfg):
    n_iter = exp_cfg['n_inner_iter']
    lr = exp_cfg['inner_lr']

    loss_trace = []
    acc = []

    optimizer = torch.optim.SGD(task_model.parameters(), lr=lr)
    for i in range(n_iter):
        batch_spt = task_sampler.get_next_batch()
        x_tr = batch_spt['x']
        y_tr = batch_spt['y']
        y_1hot_tr = batch_spt['y_1hot']

        optimizer.zero_grad()
        y_pred = task_model(x_tr)
        loss = loss_fn(y_pred, y_1hot_tr)

        loss.backward()
        optimizer.step()

        acc.append((y_pred.argmax(dim=1) == y_tr).sum().item() / len(y_tr))

        eval_loss = eval_loss_fn(y_pred, y_tr)
        loss_trace.append(eval_loss.item())

    return loss_trace, acc


def meta_train(meta_loss_model, meta_optimizer, meta_objective,
               task_sampler_train, task_sampler_test_eval, task_sampler_train_eval, exp_cfg):
    num_tasks = exp_cfg['num_tasks']
    n_outer_iter= exp_cfg['n_outer_iter']
    n_inner_iter=exp_cfg['n_inner_iter']
    inner_lr = exp_cfg['inner_lr']

    results = []
    train_loss_fn = nn.NLLLoss() #nn.BCEWithLogitsLoss() #nn.BCELoss() #nn.CrossEntropyLoss()

    task_model = LeNet(n_classes=2)
    task_opt = torch.optim.SGD(task_model.parameters(), lr=0.001)#inner_lr)

    for outer_i in range(n_outer_iter):
        # Sample a batch of support and query images and labels.

        qry_losses = []
        for _ in range(1):

            meta_optimizer.zero_grad()
            for i in range(num_tasks):
                batch_spt = task_sampler_train.get_next_batch()
                x_spt = batch_spt['x']
                y_1hot_spt = batch_spt['y_1hot']
                y_spt = batch_spt['y'] #.float().reshape(exp_cfg['batch_size'], 1)

                # zero gradients wrt to meta loss parameters
                with higher.innerloop_ctx(task_model, task_opt,
                                          copy_initial_weights=False) as (fmodel, diffopt):

                    # update model parameters via meta loss
                    yp = fmodel(x_spt)
                    pred_loss, pred_lr = meta_loss_model(yp, y_1hot_spt)#y_spt)
                    diffopt.step(pred_loss)

                    # compute task loss with new model
                    yp = fmodel(x_spt)
                    task_loss = meta_objective(yp, y_spt)

                    # this accumulates gradients wrt to meta parameters
                    task_loss.backward()
                    qry_losses.append((yp.argmax(dim=1) == y_spt).sum().item() / len(y_spt))

            meta_optimizer.step()

        avg_qry_loss = sum(qry_losses) / num_tasks
        if outer_i % 10 == 0:
            res_train_eval_reg = eval(task_sampler_lst=task_sampler_train_eval, exp_cfg=exp_cfg,
                                      train_loss_fn=train_loss_fn, eval_loss_fn=train_loss_fn)

            res_train_eval_ml3 = eval(task_sampler_lst=task_sampler_train_eval, exp_cfg=exp_cfg,
                                      train_loss_fn=meta_loss_model, eval_loss_fn=train_loss_fn)

            res_test_eval_reg = eval(task_sampler_lst=task_sampler_test_eval, exp_cfg=exp_cfg,
                                     train_loss_fn=train_loss_fn, eval_loss_fn=train_loss_fn)

            res_test_eval_ml3 = eval(task_sampler_lst=task_sampler_test_eval, exp_cfg=exp_cfg,
                                     train_loss_fn=meta_loss_model, eval_loss_fn=train_loss_fn)

            res = {}
            res['train_reg'] = res_train_eval_reg
            res['train_ml3'] = res_train_eval_ml3
            res['test_reg'] = res_test_eval_reg
            res['test_ml3'] = res_test_eval_ml3
            res['task_loss'] = {}
            res['task_loss']['acc'] = qry_losses
            results.append(res)
            test_loss_ml3 = np.mean(res_test_eval_ml3['acc'], axis=0)[-1]
            test_loss_reg = np.mean(res_test_eval_reg['acc'], axis=0)[-1]
            train_loss_reg = np.mean(res_train_eval_reg['acc'], axis=0)[-1]
            train_loss_ml3 = np.mean(res_train_eval_ml3['acc'], axis=0)[-1]
            print(
                f'[Epoch {outer_i:.2f}] Train ACC REG: {train_loss_reg:.2f}]| Train ACC ML3: {train_loss_ml3:.2f}]| ...'
                f'Test Acc ML3: {test_loss_ml3:.2f} | Test Acc REG: {test_loss_reg:.2f}'
            )

    return results


def eval(task_sampler_lst, exp_cfg, train_loss_fn, eval_loss_fn):
    seed = exp_cfg['seed']

    np.random.seed(seed)
    torch.manual_seed(seed)
    accs = []
    loss_traces = []
    for task_sampler in task_sampler_lst:
        task_sampler.reset()

        task_model_test = LeNet(n_classes=2)
        loss, acc = regular_train(loss_fn=train_loss_fn, eval_loss_fn=eval_loss_fn, task_model=task_model_test,
                                  task_sampler=task_sampler, exp_cfg=exp_cfg)
        accs.append(acc)
        loss_traces.append(loss)

    res = {'acc': accs, 'loss_trace': loss_traces}
    return res


def main(exp_cfg):
    seed = exp_cfg['seed']
    outer_lr = exp_cfg['outer_lr']
    batch_size = exp_cfg['batch_size']
    posc_tr = exp_cfg['train_pos_class']
    negc_tr = exp_cfg['train_neg_class']

    np.random.seed(seed)
    torch.manual_seed(seed)

    meta_loss_model = MnistLearnedLoss(in_dim=exp_cfg['metaloss']['in_dim'],
                                       hidden_dim=exp_cfg['metaloss']['hidden_dim'])

    meta_optimizer = torch.optim.Adam(meta_loss_model.parameters(), lr=outer_lr)

    meta_objective = nn.NLLLoss()

    task_sampler_train = MnistClassPair(pos_class=posc_tr, neg_class=negc_tr, batch_size=1, shard_name='train',
                                        seed=seed, data_dir='mnist_data')
    task_sampler_train_eval = []
    task_sampler_train_eval.append(MnistClassPair(pos_class=posc_tr, neg_class=negc_tr, batch_size=64,
                                                  shard_name='train',seed=seed, data_dir='mnist_data'))

    task_sampler_test_eval = []
    for i in range(4):
        posc_te = exp_cfg['test_pos_classes'][i]
        negc_te = exp_cfg['test_neg_classes'][i]
        test_task = MnistClassPair(pos_class=posc_te, neg_class=negc_te, batch_size=batch_size,
                                   shard_name='train',seed=seed, data_dir='mnist_data')

        task_sampler_test_eval.append(test_task)

    res = meta_train(meta_loss_model=meta_loss_model, meta_optimizer=meta_optimizer, meta_objective=meta_objective,
                     task_sampler_train=task_sampler_train,
                     task_sampler_test_eval=task_sampler_test_eval, task_sampler_train_eval=task_sampler_train_eval,
                     exp_cfg=exp_cfg)

    exp ={}
    exp['res'] = res
    exp['config'] = exp_cfg

    pkl_file = os.path.join(exp_cfg['log_dir'], exp_cfg['exp_log_file_name'])

    pkl_dir = os.path.dirname(pkl_file)
    if pkl_dir is not '' and not os.path.exists(pkl_dir):  # Create directory if it doesn't exist.
        os.makedirs(pkl_dir)
    with open(pkl_file, 'wb') as fp:
        pickle.dump(exp, fp, protocol=pickle.HIGHEST_PROTOCOL)

