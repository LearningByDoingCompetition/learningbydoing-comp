import numpy as np
from .robot_arm import RobotOpenChainArm

from matplotlib.patches import Rectangle

# ~~~~~~~~~~~ 2 link robots ~~~~~~~~~~~


class RobotPrismatic2Link(RobotOpenChainArm):
    """ 2-link prismatic """

    _num_links = 2
    d_U = 2

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _calculate_workspace(self):
        # Assumes that length links has already been set
        link1_len, link2_len = self._length_links
        self._ws_x_bounds = np.array(
            [-2.*link2_len, 2.*link2_len]
        )
        self._ws_y_bounds = np.array(
            [-2.*link1_len, 2.*link1_len]
        )

        # workspace limits for visualization
        self._ws_xlim_viz = self._ws_x_bounds + np.array([-1., 1.])
        self._ws_ylim_viz = self._ws_y_bounds + np.array([-1., 1.])

    def _draw_workspace(self, fig, ax):
        rec_xy = [self._ws_x_bounds[0], self._ws_y_bounds[0]]
        rec_width = self._ws_x_bounds[1] - self._ws_x_bounds[0]
        rec_height = self._ws_y_bounds[1] - self._ws_y_bounds[0]
        patch = Rectangle(
            rec_xy,
            rec_width,
            rec_height,
            alpha=0.2,
            color='black'
        )
        ax.add_patch(patch)

    def compute_dynamics(self, link_pos_vector, link_vel_vector):

        link1_mass, link2_mass = self._mass_links

        # Intertia matrix
        M = np.array([[link1_mass + link2_mass, 0],
                      [0, link2_mass]])

        # No rotational movement: coriolis matrix zero
        # TODO: double check, this should fix a problem if d_U
        #       is different due to a control interface
        # C = np.zeros((self.d_U, self.d_U))
        C = np.zeros((self._num_links, self._num_links))

        # Gravity terms
        N = np.array([self.g*(link1_mass + link2_mass), 0])

        return M, C, N

    def compute_joints_from_ee_cart_pos(self, ee_cart_pos):

        link1_len, link2_len = self._length_links
        ee_x, ee_y = ee_cart_pos
        joint1_pos = ee_y - link1_len
        joint2_pos = ee_x - link2_len

        joint_pos_vec = np.array([joint1_pos, joint2_pos])

        # Turn off assertion below, but was helpful for debugging
        # test_pts = self.compute_links_cart_pos(joint_pos_vec)
        # assert np.linalg.norm(test_pts[-1] - ee_cart_pos) < 1.e-8

        return joint_pos_vec

    def compute_links_cart_pos(self, joint_pos_vec):
        link1_len, link2_len = self._length_links
        x1 = 0.
        y1 = link1_len + joint_pos_vec[0]
        x2 = x1 + link2_len + joint_pos_vec[1]
        y2 = y1

        link_cart_pos = np.array([
            [x1, y1],
            [x2, y2],
        ])
        return link_cart_pos

    def get_Jacobian(self, link_pos_vector):
        J = np.zeros((2, self.num_links))
        J[0, 0] = 0.
        J[0, 1] = 1.
        J[1, 0] = 1.
        J[1, 1] = 0.
        return J

    def get_Jacobians(self, link_pos_vector):
        J2 = self.get_Jacobian(link_pos_vector)
        J1 = np.zeros((2, self.num_links - 1))
        J1[0, 0] = 0.
        J1[1, 0] = 1.
        return [J1, J2]

    def sample_cart_point_from_ws(self):
        np.random.seed(self.seed)
        x_pt = np.random.uniform(
            low=self._ws_x_bounds[0],
            high=self._ws_x_bounds[1]
        )
        y_pt = np.random.uniform(
            low=self._ws_y_bounds[0],
            high=self._ws_y_bounds[1]
        )
        ee_point = np.array([x_pt, y_pt])
        return ee_point

    def is_cart_point_in_ws(self, point_to_check):
        x_pt, y_pt = point_to_check
        is_in_ws = \
            self._ws_x_bounds[0] <= x_pt and \
            x_pt <= self._ws_x_bounds[1] and \
            self._ws_y_bounds[0] <= y_pt and \
            y_pt <= self._ws_y_bounds[1]
        return is_in_ws


class RobotPrismatic2LinkILinear(RobotPrismatic2Link):

    @property
    def d_U(self):
        return self.parameters['i_G'].shape[1]

    def __init__(self, **kwargs):
        # default linear interface parameters
        defaults = {'i_G': np.eye(2)}
        if hasattr(self, 'parameters'):
            self.parameters.update(defaults)
        else:
            self.parameters = defaults
        super().__init__(**kwargs)

    def control_interface(self, u=None):
        if u is not None:
            u = self.parameters['i_G'].dot(u)
        return super().control_interface(u)
