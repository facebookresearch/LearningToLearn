import time
import os
import numpy as np
from ml3.mnist_task import main

if __name__ == '__main__':
    EXP_TITLE = 'binary_mnist_25_random_trials'
    experiment_name = os.path.join('mnist_experiments', EXP_TITLE, str(int(time.time())))

    np.random.seed(0)
    seeds = [2]

    exp_cfg = {
        'outer_lr': 0.001,
        'n_outer_iter': 2000,
        'num_test_tasks': 10,
    }

    exp_cfg['model'] = {}
    exp_cfg['model']['in_dim'] = 1

    exp_cfg['metaloss'] = {}

    exp_str_template = "seed_{}_meta_arch_{}_batch_size_{}_num_tasks_{}_n_inner_iter_{}_inner_lr_{}_outer_lr_{}.pkl"

    for seed in seeds:
        exp_cfg['seed'] = seed
        exp_cfg['n_inner_iter'] = 100
        exp_cfg['inner_lr'] = 0.001
        exp_cfg['batch_size'] = 1
        exp_cfg['num_tasks'] = 100

        randc = np.random.permutation(10)
        exp_cfg['train_pos_class'] = randc[0]
        exp_cfg['train_neg_class'] = randc[1]

        exp_cfg['test_pos_classes'] = [randc[2], randc[3], randc[4], randc[5]]
        exp_cfg['test_neg_classes'] = [randc[6], randc[7], randc[8], randc[9]]

        exp_cfg['metaloss']['in_dim'] = 4
        exp_cfg['metaloss']['hidden_dim'] = [40, 40]

        exp_cfg['log_dir'] = "experiments/{}".format(EXP_TITLE)
        exp_file = exp_str_template.format(exp_cfg['seed'],
                                           str(exp_cfg['metaloss']['hidden_dim']),
                                           exp_cfg['batch_size'],
                                           exp_cfg['num_tasks'],
                                           exp_cfg['n_inner_iter'],
                                           exp_cfg['inner_lr'],
                                           exp_cfg['outer_lr']
                                           )
        exp_cfg['exp_log_file_name'] = exp_file

        main(exp_cfg)
