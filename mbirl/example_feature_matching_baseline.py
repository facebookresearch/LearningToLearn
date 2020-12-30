import os
import random
import numpy as np
import torch
from structured_kinematics.diff_kinematics import RobotModelTorch
import logging; logging.disable(logging.CRITICAL);

import cvxpy as cp

##Code adapted from:  https://github.com/reinforcement-learning-kr/lets-do-irl

class EnvWrapper(torch.nn.Module):
    def __init__(self, model,lr=1.0):
        super().__init__()
        self.action = torch.nn.Parameter(torch.Tensor(np.zeros([25, 7])))
        self.robot_model = model
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, x, u=0):
        xdesired = x + u
        tl = torch.Tensor(joint_limits)
        xdesired = torch.where(xdesired > tl, tl, xdesired)
        xdesired = torch.where(xdesired < -tl, -tl, xdesired)
        keypoints = []
        for link in [1,2,3]:
            kp_pos, _ = self.robot_model.forward_kinematics(xdesired, 'kp_link_'+str(link))
            keypoints+=kp_pos
        return torch.stack(keypoints).squeeze()

    def roll_out(self, joint_state):
        qs = []
        key_pos = []
        qs.append(joint_state.clone())
        key_pos.append(self.forward(joint_state.detach()))
        for t in range(25):
            ac = self.action[t]
            ee_pred = self.forward(joint_state, ac)
            joint_state = (joint_state+ac).clone()
            tl = torch.Tensor(joint_limits)
            joint_state = torch.where(joint_state > tl, tl, joint_state)
            joint_state = torch.where(joint_state < -tl, -tl, joint_state)
            qs.append(joint_state.clone())
            key_pos.append(ee_pred.clone())
        return torch.cat((torch.stack(qs), torch.stack(key_pos)),dim=1)

    def reset_gradients(self):
        self.action.detach()

class QPoptimizer(object):
    def __call__(self, feature_num, learner, expert, loss, nathan=False):
        w = cp.Variable(feature_num)
        obj_func = cp.Minimize(cp.norm(w))
        if not nathan:
            constraints = [(expert-learner) @ w >= 1]
        else:
            constraints = [(expert-learner) @ w - loss >= 0]

        prob = cp.Problem(obj_func, constraints)
        prob.solve()

        if prob.status == "optimal":
            weights = np.squeeze(np.asarray(w.value))
            return weights, prob.value
        else:
            weights = np.zeros(feature_num)
            return weights, prob.status

class IRL_Cost(object):
    def __call__(self, pred_traj, target_traj):
        loss = ((pred_traj[:,-9:] - target_traj[:,-9:])**2).sum(dim=0)
        return loss.mean()

class WeightedLoss(object):
    def __call__(self, phi, weight):
        loss = torch.stack([torch.dot(weight,l) for l in phi])
        return loss.mean()

def irl_train(action_seq,expert_traj, current_traj, expert_features, learned_features, feats, goals, loss_fn, task_loss_fn, target_loss_fn, params):
    count = 0
    costs = []
    all_iter = 0
    while True:
        all_iter+=1
        if all_iter>=200:
            break

        if not torch.is_tensor(current_traj):
            current_traj = torch.FloatTensor(current_traj[:,-9:])
        if not torch.is_tensor(expert_traj):
            expert_traj = torch.FloatTensor(expert_traj)

        if params['nathan']:
            loss = torch.sum((expert_traj - current_traj)**2, dim=1).detach().numpy()
        else:
            loss = torch.zeros(1)

        # compute optimal weights with convex optimization (Abbeel et al.)
        W, _ = task_loss_fn(len(expert_features), expert_features, feats, loss, params['nathan'])
        if params['features'] == 'pred-goals':
            temp_features = np.zeros_like(goals[0][:-1, -9:])
        elif params['features'] == 'pred-targ':
            temp_features = np.zeros_like(goals[0][:, -9:])
        else:
            raise NotImplementedError
        costs.append([])
        iter_res = {}
        for y in range(len(goals)):
            # start over
            for idx in range(params['n_inner_iter']):
                # reset gradients
                action_seq.optimizer.zero_grad()
                action_seq.reset_gradients()
                # useful variables
                start_pose = goals[y][0,:7]
                target_pose = goals[y]
                joint_state = torch.Tensor(start_pose)
                target_traj = torch.Tensor(target_pose)
                weight = torch.DoubleTensor(W)
                # unroll and update
                pred_traj = action_seq.roll_out(joint_state.clone())
                if params['features'] == 'pred-goals':
                    phi = ((pred_traj[:-1,-9:]-target_traj[1:, -9:])**2)
                elif params['features'] == 'pred-targ':
                    phi = ((pred_traj[:,-9:]-target_traj[:, -9:])**2)
                else:
                    raise NotImplementedError

                #update the actions, given the current cost function
                cost = loss_fn(phi, weight)
                current_traj = pred_traj[:, -9:]
                cost.backward(retain_graph=True)
                action_seq.optimizer.step()
                # unroll again after inner loop completes
                pred_traj = action_seq.roll_out(joint_state.clone())

            print("irl cost training iter: {} loss: {}".format(all_iter, target_loss_fn(pred_traj,target_traj).item()))





            # compute features extracted using current weights
            if params['features'] == 'pred-goals':
                temp_features[:] += ((pred_traj[:-1, -9:]-target_traj[1:, -9:])**2).detach().numpy().astype(np.double)
            elif params['features'] == 'pred-targ':
                temp_features[:] += ((pred_traj[:, -9:]-target_traj[:, -9:])**2).detach().numpy().astype(np.double)
            else:
                raise NotImplementedError

        temp_features[:] /= float(len(goals))
        r = np.array([params['gamma']**i * temp_features[i] for i in range(len(temp_features))])
        temp_features = r[::-1].cumsum(axis=0)[::-1][0]
        feats = temp_features
        hyper_distance = np.abs(np.dot(W, expert_features-temp_features)) #hyperdistance = t
        learned_features[hyper_distance] = temp_features
        if hyper_distance <= params['epsilon']: # terminate if the point reached close enough
            break
        count += 1


if __name__ == '__main__':
    random.seed(10)
    np.random.seed(10)
    torch.manual_seed(0)
    rest_pose = [0.0, 0.0, 0.0, 1.57079633, 0.0, 1.03672558, 0.0]
    joint_limits = [2.967,2.094,2.967,2.094,2.967,2.094,3.054]

    robot_model = RobotModelTorch(rel_urdf_path='env/kuka_iiwa/urdf/iiwa7_ft_with_obj_keypts.urdf')
    learned_u_seq = EnvWrapper(robot_model)


    params = {
        'gamma': 0.9,
        'epsilon': 0.001,
        'features': 'pred-targ', # 'pred-goals', #
        'nathan': False, # use l(t,y) as an additional constraint
        'n_inner_iter': 10,
        'time_horizon': 25
    }

    expert_demo = torch.Tensor(np.load('expert_demo.npy'))
    no_demos = 1

    avg_features = []
    if params['features'] == 'pred-goals':
        avg_local_features = np.zeros((no_demos,25,9))
    elif params['features'] == 'pred-targ':
        avg_local_features = np.zeros((no_demos,26,9))
    else:
        raise NotImplementedError
    # for idx, traj in enumerate(trajs[:no_demos]):

    expert_traj = expert_demo[:,-9:]
    if params['features'] == 'pred-goals':
        avg_local_features[0] = (expert_demo[:-1,-9:]-expert_demo[1:,-9:])**2
    elif params['features'] == 'pred-targ':
        avg_local_features[0] = (expert_demo[:,-9:]-expert_demo[:,-9:])**2
    else:
        raise NotImplementedError
    r = np.array([params['gamma']**i * avg_local_features[0][i] for i in range(len(avg_local_features[0]))])
    avg_local_features[0] = r[::-1].cumsum(axis=0)[::-1]

    avg_local_features = avg_local_features[:, 0]
    expert_features = np.mean(avg_local_features, axis=0)
    random_features = np.random.uniform(low=[-1 for _ in range(expert_features.shape[0])],high=[1 for _ in range(expert_features.shape[0])])

    random_target = np.linalg.norm(np.asarray(expert_features)-np.asarray(random_features)) #norm of the diff in expert and random
    policies_features = {random_target:random_features} # storing the policies and their respective t values in a dictionary
    random_traj = np.zeros_like(expert_traj)

    task_loss_fn = QPoptimizer()
    target_loss_fn = IRL_Cost()
    loss_fn = WeightedLoss()


    irl_train(learned_u_seq, expert_traj, random_traj, expert_features, policies_features, random_features, [expert_demo], loss_fn, task_loss_fn, target_loss_fn, params)
