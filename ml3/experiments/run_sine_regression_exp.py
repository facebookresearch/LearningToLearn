import os
import ml3
from ml3.sine_regression_task import main as meta_train

EXP_FOLDER = os.path.join(ml3.__path__[0], "experiments/data/sine_exp")


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
    exp_cfg['log_dir'] = f"{EXP_FOLDER}"


    for seed in range(5):
        exp_cfg['seed'] = seed
        exp_file = "sine_regression_seed_{}.pkl".format(exp_cfg['seed'])
        exp_cfg['exp_log_file_name'] = exp_file
        meta_train(exp_cfg)