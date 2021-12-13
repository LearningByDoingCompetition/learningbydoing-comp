import numpy as np
import scipy as sp

from dgp.systems import RobotControllerInterface


class RobotRandomController(RobotControllerInterface):
    """ Implements a random controller """

    def __init__(self, seed, dim_output, **kwargs):

        # Set seed
        np.random.seed(seed)
        self._seed = seed

        super().__init__(dim_output, **kwargs)

    def get_input(self, observation=None, reference=None):
        inp = np.random.normal(0, 1, self._dim_output)
        return inp


class RobotZeroController(RobotControllerInterface):
    """ Implements a zero control controller """

    def __init__(self, seed, dim_output, **kwargs):
        super().__init__(dim_output, **kwargs)

    def get_input(self, observation=None, reference=None):
        return np.zeros(self._dim_output)


class RobotLQRController(RobotControllerInterface):
    """ Implements a LQR controller that uses the exact robot parameters
    Useful for debugging and a baseline with using the robot directly """

    def __init__(self, seed, robot_sys, **kwargs):

        # Set seed
        np.random.seed(seed)
        self._seed = seed

        super().__init__(robot_sys.d_U, **kwargs)

        # Save object for later use
        self._robot_sys = robot_sys

        self.A = np.zeros((4, 4))
        self.B = np.zeros((4, 2))
        self.A[:2, 2:] = np.eye(2)
        self.B[2:, :] = np.eye(2)
        self.Q = np.diag([10., 10., 1., 1.])
        self.R = 0.01*np.eye(2)
        self.F = self.get_LQR()

    def get_LQR(self):
        P = np.matrix(sp.linalg.solve_continuous_are(
            self.A, self.B, self.Q, self.R))
        # F = sp.linalg.inv(self.R)@self.B.T@P
        # numerically more stable alternatives to explicit inversion
        # F = np.linalg.lstsq(self.R, self.B.T.dot(P), rcond=None)[0]
        F = np.linalg.solve(self.R, self.B.T.dot(P))
        return -F

    def get_input(self, observation, reference):
        # Normally, get_input provides ee pos
        # but this uses the robot joint information
        # to act as a ground truth controller
        # curr_state = self._robot_sys.current_state
        curr_pos_joints, curr_vel_joints = self._robot_sys.current_joint_posvel

        M, C, N = self._robot_sys.compute_dynamics(
            curr_pos_joints, curr_vel_joints)
        J = self._robot_sys.current_jacobian
        F = self.get_LQR()
        ee_cart_pos, ee_cart_vel = self._robot_sys.current_ee_cart_posvel
        ee_cart_posvel = np.hstack((ee_cart_pos, ee_cart_vel))
        wM_des = F@((ee_cart_posvel - reference).reshape(-1, 1))
        torque = N + C@curr_vel_joints
        torque += np.asarray(M@J.T@wM_des).flatten()

        # If a control interface exists, use it
        if 'i_G' in self._robot_sys.parameters:
            i_G = self._robot_sys.parameters['i_G']
            # torque = np.linalg.pinv(i_G).dot(torque)
            # numerically more stable alternatives to explicit pinversion
            torque = np.linalg.lstsq(i_G, torque, rcond=None)[0]

        return torque / 1000
