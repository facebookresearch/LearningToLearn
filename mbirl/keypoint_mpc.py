# Copyright (c) Facebook, Inc. and its affiliates.
import torch
import numpy as np


joint_limits = [2.967, 2.094, 2.967, 2.094, 2.967, 2.094, 3.054]


# A wrapper class keypoint MPC with action parameters to be optimized
# This implementation assumes object keypoints are known and part of the robot model
# this means the  keypoint dynamics model can be implemented through a forward kinematics call
class KeypointMPCWrapper(torch.nn.Module):

    def __init__(self, model):
        super().__init__()
        self.action_seq = torch.nn.Parameter(torch.Tensor(np.zeros([25, 7])))
        self.robot_model = model

    def forward(self, x, u=0):
        xdesired = x + u
        tl = torch.Tensor(joint_limits)
        xdesired = torch.where(xdesired > tl, tl, xdesired)
        xdesired = torch.where(xdesired < -tl, -tl, xdesired)
        keypoints = []
        for link in [1,2,3]:
            kp_pos, _ = self.robot_model.compute_forward_kinematics(xdesired.reshape(1, 7), 'kp_link_'+str(link))
            keypoints += kp_pos[0]
        return torch.stack(keypoints).squeeze()

    def roll_out(self, joint_state):
        qs = []
        key_pos = []
        qs.append(joint_state.clone())
        key_pos.append(self.forward(joint_state.detach()))
        for t in range(25):
            ac = self.action_seq[t]
            # TODO: this seems wrong - shouldn't it detach the joint_state?
            keypts_pred = self.forward(joint_state, ac)
            joint_state = (joint_state+ac).clone()
            tl = torch.Tensor(joint_limits)
            joint_state = torch.where(joint_state > tl, tl, joint_state)
            joint_state = torch.where(joint_state < -tl, -tl, joint_state)
            qs.append(joint_state.clone())
            key_pos.append(keypts_pred.clone())
        return torch.cat((torch.stack(qs), torch.stack(key_pos)),dim=1)

    def reset_actions(self):
        # TODO: this seems wrong - shouldn't it be setting the actions to 0 ?
        self.action_seq.detach()
