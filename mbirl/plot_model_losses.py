import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from os.path import dirname, abspath
import os

_ROOT_DIR = dirname(abspath(__file__))
sys.path.append(_ROOT_DIR)

model_data_dir = os.path.join(_ROOT_DIR, 'model_data')

for data_type in ['placing', 'reaching']:

    baseline = torch.load(f"{model_data_dir}/{data_type}_Abbeel")
    timedep = torch.load(f"{model_data_dir}/{data_type}_TimeDep")
    weighted = torch.load(f"{model_data_dir}/{data_type}_Weighted")

    # IRL Cost

    plt.figure()

    plt.plot(baseline['irl_cost_tr'].detach(), color='red', alpha=0.5, label="Baseline")
    plt.plot(weighted['irl_cost_tr'].detach(), color='orange', label="Weighted Ours")
    plt.plot(timedep['irl_cost_tr'].detach(), color='green', label="Time Dep Weighted Ours")
    plt.xlabel("iterations")
    plt.ylabel("IRL Cost")
    plt.legend()

    plt.savefig(f"{model_data_dir}/{data_type}_IRL_cost.png")

    # Eval
    plt.figure()
    baseline_trace = baseline['irl_cost_eval'].detach()
    b_mean = baseline_trace.mean(dim=-1)
    b_std = baseline_trace.std(dim=-1)
    weighted_trace = weighted['irl_cost_eval'].detach()
    w_mean = weighted_trace.mean(dim=-1)
    w_std = weighted_trace.std(dim=-1)
    timedep_trace = timedep['irl_cost_eval'].detach()
    t_mean = timedep_trace.mean(dim=-1)
    t_std = timedep_trace.std(dim=-1)
    plt.plot(b_mean, color='red', alpha=0.5, label="Baseline")
    plt.fill_between(np.arange(len(b_mean)), b_mean - b_std, b_mean + b_std, color='red', alpha=0.1)
    plt.plot(w_mean, color='orange', label="Weighted Ours")
    plt.fill_between(np.arange(len(w_mean)), w_mean - w_std, w_mean + w_std, color='orange', alpha=0.1)
    plt.plot(t_mean, color='green', label="Time Dep Weighted Ours")
    plt.fill_between(np.arange(len(t_mean)), t_mean - t_std, t_mean + t_std, color='green', alpha=0.1)
    plt.xlabel("iterations")
    plt.ylabel("IRL Cost")
    plt.legend()

    plt.savefig(f"{model_data_dir}/{data_type}_Eval.png")
