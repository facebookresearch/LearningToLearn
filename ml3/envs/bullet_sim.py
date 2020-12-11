# Copyright (c) Facebook, Inc. and its affiliates.
import os

import numpy as np

import pybullet_utils.bullet_client as bc
import pybullet_data
import pybullet

class BulletSimulation(object):

    def __init__(self, gui, controlled_joints, ee_idx, torque_limits,target_pos):

        if gui:
            self.sim = bc.BulletClient(connection_mode=pybullet.GUI)
        else:
            self.sim = bc.BulletClient(connection_mode=pybullet.DIRECT)
        self.sim.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.ee_idx = ee_idx

        self.cur_joint_pos = None
        self.cur_joint_vel = None
        self.curr_ee = None
        self.babbling_torque_limits = None
        self.logger = None

        self.controlled_joints = controlled_joints
        self.torque_limits = torque_limits

        self.n_dofs = len(controlled_joints)

        #pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        # TODO: the following should be extracted into some world model that is loaded independently of the robot
        #self.planeId = pybullet.loadURDF("plane.urdf",[0,0,0])
        if target_pos is not None:
            self.cubeId = pybullet.loadURDF("sphere_small.urdf",target_pos)

    def disconnect(self):
        self.sim.disconnect()
        #pybullet.disconnect()

    def get_random_torques(self, time_horizon, bounds_low, bounds_high):
        trajectory= []
        for t in range(time_horizon):
            torque_limits_babbling = bounds_high
            torques = np.random.uniform(-torque_limits_babbling, torque_limits_babbling)
            trajectory.append(np.array(torques))
        return np.array(trajectory)

    def get_random_torques_uinit(self, time_horizon):
        trajectory= []
        for t in range(time_horizon):
            torques = np.random.uniform(-0.3*self.torque_limits, 0.3*self.torque_limits)
            trajectory.append(np.array(torques))
        return np.array(trajectory)

    def get_target_joint_configuration(self, target_position):
        print(target_position)
        self.reset()
        des_joint_state = self.sim.calculateInverseKinematics(self.robot_id,
                                                              self.ee_idx,
                                                              np.array(target_position), jointDamping = [0.1 for i in range(self.n_dofs)])
        return np.asarray(des_joint_state)

    def reset(self, joint_pos=None, joint_vel=None):
        if joint_vel is None:
            joint_vel = list(np.zeros(self.n_dofs))

        if joint_pos is None:
            joint_pos = list(np.zeros(self.n_dofs))

        for i in range(self.n_dofs):
            self.sim.resetJointState(bodyUniqueId=self.robot_id,
                                     jointIndex=self.controlled_joints[i],
                                     targetValue=joint_pos[i],
                                     targetVelocity=joint_vel[i])

        self.sim.stepSimulation()
        self.cur_joint_pos = self.get_current_joint_pos()
        self.cur_joint_vel = self.get_current_joint_vel()
        self.curr_ee = self.get_current_ee_state()
        return np.hstack([self.get_current_joint_pos(),self.get_current_joint_vel()])

    def move_to_joint_positions(self, joint_pos, joint_vel=None):
        if joint_vel is None:
            joint_vel = [0]*len(joint_pos)

        for i in range(self.n_dofs):
            self.sim.resetJointState(bodyUniqueId=self.robot_id,
                                     jointIndex=self.controlled_joints[i],
                                     targetValue=joint_pos[i],
                                     targetVelocity=joint_vel[i])

        self.sim.stepSimulation()

        self.cur_joint_pos = self.get_current_joint_pos()
        self.cur_joint_vel = self.get_current_joint_vel()
        self.curr_ee = self.get_current_ee_state()
        return np.hstack([self.cur_joint_pos,self.cur_joint_vel])

    def get_MassM(self,angles):
        for link_idx in self.controlled_joints:
            self.sim.changeDynamics(self.robot_id, link_idx, linearDamping=0.0, angularDamping=0.0, jointDamping=0.0)
        cur_joint_angles = list(angles)
        mass_m =  self.sim.calculateMassMatrix(bodyUniqueId=self.robot_id,
                                            objPositions = cur_joint_angles)

        return np.array(mass_m)

    def get_F(self,angles,vel):
        for link_idx in self.controlled_joints:
            self.sim.changeDynamics(self.robot_id, link_idx, linearDamping=0.0, angularDamping=0.0, jointDamping=0.0)
        cur_joint_angles = list(angles)
        cur_joint_vel = list(vel)
        torques = self.sim.calculateInverseDynamics(self.robot_id,
                                                    cur_joint_angles,
                                                    cur_joint_vel,
                                                    [0]*self.action_dim)
        return np.asarray(torques)



    def joint_angles(self):
        return self.cur_joint_pos

    def joint_velocities(self):
        return self.cur_joint_vel

    def forwad_kin(self,state):
        return self.endeffector_pos()

    def endeffector_pos(self):
        return self.curr_ee

    def get_target_ee(self, state):

        for i in range(self.n_dofs):
            self.sim.resetJointState(bodyUniqueId=self.robot_id,
                                     jointIndex=self.controlled_joints[i],
                                     targetValue=state[i],
                                     targetVelocity=0.0)
        self.sim.stepSimulation()

        ls = self.sim.getLinkState(self.robot_id, self.ee_idx)[0]
        return ls

    def reset_then_step(self, des_joint_state, torque):
        # for link_idx in self.controlled_joints:
        #     self.sim.changeDynamics(self.robot_id, link_idx, linearDamping=0.0, angularDamping=0.0, jointDamping=0.0)
        for i in range(self.n_dofs):
            self.sim.resetJointState(bodyUniqueId=self.robot_id,
                                     jointIndex=self.controlled_joints[i],
                                     targetValue=des_joint_state[i],
                                     targetVelocity=des_joint_state[(i+self.n_dofs)])

        return self.apply_joint_torque(torque)[0]

    def step_model(self,state,torque):
        return self.sim_step(state,torque)

    def sim_step(self,state,torque):
        for link_idx in self.controlled_joints:
            self.sim.changeDynamics(self.robot_id, link_idx, linearDamping=0.0, angularDamping=0.0, jointDamping=0.0)
        if str(state.dtype).startswith('torch'):
            state = state.clone().detach().numpy()
        if str(torque.dtype).startswith('torch'):
            torque = torque.clone().detach().numpy()
        return self.reset_then_step(state,torque)

    def step(self,state,torque):
        for link_idx in self.controlled_joints:
            self.sim.changeDynamics(self.robot_id, link_idx, linearDamping=0.0, angularDamping=0.0, jointDamping=0.0)
        if str(state.dtype).startswith('torch'):
            state = state.clone().detach().numpy()
        if str(torque.dtype).startswith('torch'):
            torque = torque.clone().detach().numpy()
        return self.reset_then_step(state,torque)

    def apply_joint_torque(self, torque):


        self.grav_comp = self.inverse_dynamics([0] * self.action_dim)
        torque = torque + self.grav_comp
        full_torque = torque.copy()


        #torque = torque.clip(-self.torque_limits, self.torque_limits)

        self.sim.setJointMotorControlArray(bodyIndex=self.robot_id,
                                           jointIndices=self.controlled_joints,
                                           controlMode=pybullet.TORQUE_CONTROL,
                                           forces=torque)
        self.sim.stepSimulation()

        cur_joint_states = self.sim.getJointStates(self.robot_id, self.controlled_joints)
        cur_joint_angles = [cur_joint_states[i][0] for i in range(self.n_dofs)]
        cur_joint_vel = [cur_joint_states[i][1] for i in range(self.n_dofs)]

        next_state = cur_joint_angles + cur_joint_vel

        ls = list(self.sim.getLinkState(self.robot_id, self.ee_idx)[0])
        self.cur_joint_pos = self.get_current_joint_pos()
        self.cur_joint_vel = self.get_current_joint_vel()
        self.curr_ee = self.get_current_ee_state()
        return np.hstack([self.cur_joint_pos,self.cur_joint_vel]),self.curr_ee

    def get_current_ee_state(self):
        ee_state = self.sim.getLinkState(self.robot_id, self.ee_idx)
        return np.array(ee_state[0])

    def get_current_joint_pos(self):
        cur_joint_states = self.sim.getJointStates(self.robot_id, self.controlled_joints)
        cur_joint_angles = [cur_joint_states[i][0] for i in range(self.n_dofs)]
        return np.array(cur_joint_angles)

    def get_current_joint_vel(self):
        cur_joint_states = self.sim.getJointStates(self.robot_id, self.controlled_joints)
        cur_joint_vel = [cur_joint_states[i][1] for i in range(self.n_dofs)]
        return np.array(cur_joint_vel)

    def get_current_joint_state(self):
        cur_joint_states = self.sim.getJointStates(self.robot_id, self.controlled_joints)
        cur_joint_angles = [cur_joint_states[i][0] for i in range(self.n_dofs)]
        cur_joint_vel = [cur_joint_states[i][1] for i in range(self.n_dofs)]
        return np.hstack([cur_joint_angles, cur_joint_vel])

    def get_ee_jacobian(self):
        cur_joint_states = self.sim.getJointStates(self.robot_id, self.controlled_joints)
        cur_joint_angles = [cur_joint_states[i][0] for i in range(self.n_dofs)]
        cur_joint_vel = [cur_joint_states[i][1] for i in range(self.n_dofs)]
        bullet_jac_lin, bullet_jac_ang = self.sim.calculateJacobian(
            bodyUniqueId=self.robot_id,
            linkIndex=self.ee_idx,
            localPosition=[0, 0, 0],
            objPositions=cur_joint_angles,
            objVelocities=cur_joint_vel,
            objAccelerations=[0] * self.n_dofs,
        )
        return np.asarray(bullet_jac_lin), np.asarray(bullet_jac_ang)

    def inverse_dynamics(self, des_acc):
        for link_idx in self.controlled_joints:
            self.sim.changeDynamics(self.robot_id, link_idx, linearDamping=0.0, angularDamping=0.0, jointDamping=0.0)
        cur_joint_states = self.sim.getJointStates(self.robot_id, self.controlled_joints)
        cur_joint_angles = [cur_joint_states[i][0] for i in range(self.n_dofs)]
        cur_joint_vel = [cur_joint_states[i][1] for i in range(self.n_dofs)]
        torques = self.sim.calculateInverseDynamics(self.robot_id,
                                                    cur_joint_angles,
                                                    cur_joint_vel,
                                                    des_acc)
        return np.asarray(torques)


    def detect_collision(self):
        return False

    def return_grav_comp_torques(self):
        return 0.0

    def get_pred_error(self,x,u):
        return np.zeros(len(u))

    def sim_step_un(self,x,u):
        return np.zeros(len(u)),np.zeros(len(u))

    def get_gravity_comp(self):
        return 0.0


class BulletSimulationFromURDF(BulletSimulation):
    def __init__(self, rel_urdf_path, gui, controlled_joints, ee_idx, torque_limits, target_pos):
        super(BulletSimulationFromURDF, self).__init__(gui, controlled_joints, ee_idx, torque_limits, target_pos)
        urdf_path = os.getcwd()+'/envs/'+rel_urdf_path
        print("loading urdf file: {}".format(urdf_path))

        self.robot_id = self.sim.loadURDF(urdf_path, basePosition=[-0.5, 0, 0.0], useFixedBase=True)
        self.n_dofs = len(controlled_joints)

        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        #self.planeId = pybullet.loadURDF("plane.urdf")

        self.sim.resetBasePositionAndOrientation(self.robot_id,[-0.5,0,0.0],[0,0,0,1])
        self.sim.setGravity(0, 0, -9.81)
        dt = 1.0/240.0
        self.dt = dt
        self.sim.setTimeStep(dt)
        self.sim.setRealTimeSimulation(0)
        self.sim.setJointMotorControlArray(self.robot_id,
                                            self.controlled_joints,
                                            pybullet.VELOCITY_CONTROL,
                                            forces=np.zeros(self.n_dofs))


class BulletSimulationFromMJCF(BulletSimulation):

    def __init__(self, rel_mjcf_path, gui, controlled_joints, ee_idx, torque_limits):
        super(BulletSimulationFromMJCF, self).__init__(gui, controlled_joints, ee_idx, torque_limits, None)
        print('hierhierhierhier')

        xml_path = os.getcwd()+'/envs/'+rel_mjcf_path
        if rel_mjcf_path[0] != os.sep: xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), rel_mjcf_path)
        else: xml_path = rel_mjcf_path

        #xml_path = '/Users/sarah/Documents/GitHub/LearningToLearn/ml3/envs/mujoco_robots/reacher.xml'
        print("loading this mjcf file: {}".format(xml_path))

        self.world_id, self.robot_id = self.sim.loadMJCF(xml_path)

        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        print(pybullet_data.getDataPath())
        self.planeId = pybullet.loadURDF("plane.urdf")
        #self.cubeId = pybullet.loadURDF("sphere_small.urdf", [0.02534078, -0.19863741, 0.01]) #0.02534078, -0.19863741 0.10534078, 0.1663741

        self.n_dofs = len(controlled_joints)
        self.sim.setGravity(0, 0, -9.81)
        dt = 1.0/100.0
        self.dt = dt
        self.sim.setTimeStep(dt)
        self.sim.setRealTimeSimulation(0)
        self.sim.setJointMotorControlArray(self.robot_id,
                                           self.controlled_joints,
                                           pybullet.VELOCITY_CONTROL,
                                           forces=np.zeros(self.n_dofs))
