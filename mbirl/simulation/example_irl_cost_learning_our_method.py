import random
import torch
import numpy as np
import higher
from structured_kinematics.diff_kinematics import RobotModelTorch
import logging; logging.disable(logging.CRITICAL);


### The learned weighted cost, with fixed weights ###
class IRLCost(torch.nn.Module):
    def __init__(self, dim=9):
        super(IRLCost, self).__init__()
        self.weights = torch.nn.Parameter(0.01 * torch.ones([dim,1]))
        self.clip = torch.nn.Softplus()
        self.meta_grads = [[] for _, _ in enumerate(self.parameters())]

    def forward(self, y_in, y_target):
        assert y_in.dim() == 2
        mse = ((y_in[:,-9:] - y_target[-9:]) ** 2).squeeze()

        # weighted mse
        wmse = torch.mm(mse,self.weights)
        return wmse.mean()

### The learned weighted cost, with time dependent weights ###
class IRLCostSeq(torch.nn.Module):
    def __init__(self, dim=9):
        super(IRLCostSeq, self).__init__()
        self.weights = torch.nn.Parameter(0.01 * torch.ones([25,dim]))
        self.clip = torch.nn.Softplus()
        self.meta_grads = [[] for _, _ in enumerate(self.parameters())]

    def forward(self, y_in, y_target):
        assert y_in.dim() == 2
        mse = ((y_in[1:,-9:] - y_target[-9:]) ** 2).squeeze()
        # weighted mse
        wmse = torch.matmul(mse,self.weights.T)
        return wmse.mean()


### A wrapper class for the robot model and the action parameters to be optimized ###
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

### The actual IRL cost, the learning objective for the learned cost functions ###
class IRL_Cost(object):
    def __call__(self, pred_traj, target_traj):
        loss = ((pred_traj[:,-9:] - target_traj[:,-9:])**2).sum(dim=0)
        return loss.mean()



### Helper function for the meta learning loop ###
def meta_train(learned_cost, robot_model, irl_cost_fn, expert_demo, n_outer_iter, n_inner_iter):

    learned_cost_opt = torch.optim.Adam(learned_cost.parameters(), lr=1e-2)

    for outer_i in range(n_outer_iter):

        learned_cost_opt.zero_grad()
        # re-initialize action parameters for each outer iteration
        action_seq = EnvWrapper(robot_model)

        for _ in range(n_inner_iter):
            action_seq.optimizer.zero_grad()
            action_seq.reset_gradients()
            with higher.innerloop_ctx(action_seq, action_seq.optimizer) as (fpolicy, diffopt):
                start_pose = expert_demo[0,:7]
                joint_state = torch.Tensor(start_pose)
                pred_traj = fpolicy.roll_out(joint_state.clone())
                #use the learned loss to update the action sequence
                loss = learned_cost(pred_traj, expert_demo[-1])
                diffopt.step(loss)

                joint_state = torch.Tensor(start_pose)
                pred_traj = fpolicy.roll_out(joint_state)
                # compute task loss
                irl_cost = irl_cost_fn(pred_traj, expert_demo).mean()



        if outer_i % 10 == 0:
            print("irl cost training iter: {} loss: {}".format(outer_i, irl_cost.item()))

        # backprop gradient of learned loss parameters wrt task loss
        learned_cost_opt.zero_grad()
        learned_cost.zero_grad()
        irl_cost.backward(retain_graph=True)

        learned_cost_opt.step()


if __name__ == '__main__':
    random.seed(10)
    np.random.seed(10)
    torch.manual_seed(0)

    rest_pose = [0.0, 0.0, 0.0, 1.57079633, 0.0, 1.03672558, 0.0]
    joint_limits = [2.967,2.094,2.967,2.094,2.967,2.094,3.054]

    # Initialize the differentiable kinematics model of the Kuka arm
    robot_model = RobotModelTorch(rel_urdf_path='env/kuka_iiwa/urdf/iiwa7_ft_peg.urdf')


    #type of cost, either seq=time dependent or fixed
    cost_type = 'seq' #'fixed'


    expert_demo = torch.Tensor(np.load('expert_demo.npy'))
    learned_cost = None

    if cost_type=='fixed':
        learned_cost = IRLCost()
    elif cost_type == 'seq':
        learned_cost = IRLCostSeq()
    else:
        print('Loss not implemented')

    irl_cost_fn = IRL_Cost()

    n_outer_iter = 200
    n_inner_iter = 1
    time_horizon = 25
    meta_train(learned_cost, robot_model, irl_cost_fn, expert_demo, n_outer_iter, n_inner_iter)