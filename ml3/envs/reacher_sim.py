# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
from ml3.envs.bullet_sim import BulletSimulationFromMJCF


class ReacherSimulation(BulletSimulationFromMJCF):
    def __init__(self,  gui, file_name = 'mujoco_robots/reacher.xml',controlled_joints=None, ee_idx=None, torque_limits=None):
        rel_xml_path = file_name

        #fingertip
        if ee_idx is None:
            self.ee_idx = 4
        if controlled_joints is None:
            controlled_joints = [0, 2]
        if torque_limits is None:
            torque_limits = np.asarray([1, 1])

        self.action_dim=2
        self.state_dim=4
        self.pos_dim=2

        super(ReacherSimulation, self).__init__(rel_mjcf_path=rel_xml_path,
                                                gui=gui,
                                                controlled_joints=controlled_joints,
                                                ee_idx=self.ee_idx,
                                                torque_limits=torque_limits)

        if gui:
            self.sim.resetDebugVisualizerCamera(cameraDistance=0.5, cameraYaw=-50, cameraPitch=-50,
                                                cameraTargetPosition=[0, 0, 0])

        n_dofs_total = self.sim.getNumJoints(self.robot_id)
        print("n dofs total (including fixed joints): {}".format(n_dofs_total))

        for i in range(n_dofs_total):
            print(self.sim.getJointInfo(self.robot_id, i))
        return

