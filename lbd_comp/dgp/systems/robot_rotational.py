from itertools import product
import numpy as np
from .robot_arm import RobotOpenChainArm

# ~~~~~~~~~~~ move to separate utils ~~~~~~~~~~~


def points_of_intersecting_circles(
        circle1_xy,
        circle1_radius,
        circle2_xy,
        circle2_radius,
):

    # Input argument parsing
    cir1_xy = np.array(circle1_xy)
    cir2_xy = np.array(circle2_xy)
    assert circle1_radius > 0., \
        "Expected 'circle1_radius' to be positive, but it is not."
    assert circle2_radius > 0., \
        "Expected 'circle2_radius' to be positive, but it is not."
    cir1_r = circle1_radius
    cir2_r = circle2_radius

    cir_vec = cir2_xy - cir1_xy
    dist_between_circle_centers = np.linalg.norm(cir_vec)

    if dist_between_circle_centers == 0.:
        # Infinite solutions (if radii are same)
        # or, zero solutions (if radii are different)
        # Either way, return None
        sol = None
    elif dist_between_circle_centers > (cir1_r + cir2_r):
        # No solutions - circles are too far apart
        sol = None
    elif cir2_r > (dist_between_circle_centers + cir1_r):
        # No solutions - circle 2 fully envelops circle 1
        sol = None
    elif cir1_r > (dist_between_circle_centers + cir2_r):
        # No solutions - circle 1 fully envelops circle 2
        sol = None
    else:
        cir1_x, cir1_y = cir1_xy

        # vector between circles
        cir_vec_x, cir_vec_y = cir_vec

        # angle of vector between circles
        cir_dir_angle = np.arctan2(cir_vec_y, cir_vec_x)

        # determine where projection of circle 1 exists
        dist_proj_cir1_ray = (
            dist_between_circle_centers**2. + cir1_r**2. - cir2_r**2.
        )/(2.*dist_between_circle_centers)
        proj_cir1_ray_x = dist_proj_cir1_ray*np.cos(cir_dir_angle) + cir1_x
        proj_cir1_ray_y = dist_proj_cir1_ray*np.sin(cir_dir_angle) + cir1_y

        # Vector from projection of circle 1 to an intersection point
        dist_proj_cir1_to_int = np.sqrt(cir1_r**2. - dist_proj_cir1_ray**2.)
        dir_from_proj_cir1_to_int = np.array(
            [-cir_vec_y, cir_vec_x])/dist_between_circle_centers

        # Two solutions are possible
        vec = dist_proj_cir1_to_int*dir_from_proj_cir1_to_int
        int1_possol = np.array([proj_cir1_ray_x, proj_cir1_ray_y]) + vec
        int2_negsol = np.array([proj_cir1_ray_x, proj_cir1_ray_y]) - vec

        intersecting_points = np.vstack((int1_possol, int2_negsol))

        # This case should not return a nan
        # But if it does, we have a geometry implementation bug
        # Throw assert so we can detect it
        assert not np.any(np.isnan(intersecting_points))

        sol = intersecting_points
    return sol

# # ~~~~~~~~~~~ 3 link robots ~~~~~~~~~~~


class RobotRotational3Link(RobotOpenChainArm):
    """ 3-link rotational """

    _num_links = 3
    d_U = 3

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _calculate_workspace(self):
        # Assumes that length links has already been set
        self._ws_radius_max = np.sum(self._length_links)

        # Triangle geometry
        x_largest_link = np.argmax(self._length_links)
        x_other_links = np.delete(range(self._num_links), x_largest_link)
        largest_link = self._length_links[x_largest_link]
        sum_other_links = np.sum(self._length_links[x_other_links])
        if largest_link <= sum_other_links:
            # We can form a triangle with the center as a vertex
            self._ws_radius_min = 0.
        else:
            #  We cannot reach the center
            self._ws_radius_min = largest_link - sum_other_links

        # workspace limits for visualization
        self._ws_xlim_viz = np.array([
            -self._ws_radius_max - 1.,
            self._ws_radius_max + 1.
        ])
        self._ws_ylim_viz = np.array([
            -self._ws_radius_max - 1.,
            self._ws_radius_max + 1.
        ])

    def _draw_workspace(self, fig, ax):
        #  courtesy of https://stackoverflow.com/questions/22789356/plot-a-donut-with-fill-or-fill-between-use-pyplot-in-matplotlib # noqa: E501
        n, radii = 50, [self._ws_radius_min, self._ws_radius_max]
        theta_ws = np.linspace(0, 2*np.pi, n, endpoint=True)
        x_ws = np.outer(radii, np.cos(theta_ws))
        y_ws = np.outer(radii, np.sin(theta_ws))
        x_ws[1, :] = x_ws[1, ::-1]
        y_ws[1, :] = y_ws[1, ::-1]
        ax.fill(np.ravel(x_ws), np.ravel(y_ws), alpha=0.2, color='black')

    def compute_dynamics(self, link_pos_vector, link_vel_vector):

        link1_pos, link2_pos, link3_pos = link_pos_vector
        link1_vel, link2_vel, link3_vel = link_vel_vector

        link1_len, link2_len, link3_len = self._length_links
        link1_r, link2_r, link3_r = self._massloc_links
        link1_inertia, link2_inertia, link3_inertia = self._inertia_links
        link1_mass, link2_mass, link3_mass = self._mass_links

        # Implemented from:
        # "Dynamic Model and Motion Control Analysis
        # of Three-link Gymnastic Robot on Horizontal Bar"
        # X. Jian and L. Zushu
        # 2003 Int'l Conf on Robotics, Intel. Sys. and Signal Proc.
        # https://www.cs.cmu.edu/~akalamda/kdc/hw3/3linkmanipulator.pdf
        # Note: the motor masses aren't implemented
        # Note: the c1 c2 (motor friction) terms are accounted elsewhere
        # Note: The angles here are defined
        # DIFFERENTLY than for the 2link robot case
        # For this formulation, all angles are defined w/r/t vertical axis.

        A_11 = link1_mass*link1_r**2. + \
            link1_inertia + \
            (link2_mass + link3_mass)*link1_len**2.
        A_12 = (link2_mass*link2_r
                + link3_mass*link2_len)*link1_len
        A_13 = link3_mass*link3_r*link1_len
        A_22 = link2_mass*link2_r**2. + \
            link2_inertia + \
            link3_mass*link2_len**2.
        A_23 = link3_mass*link3_r*link2_len
        A_33 = link3_mass*link3_r**2. + link3_inertia

        B_11 = 0.
        B_12 = (link2_mass*link2_r
                + link3_mass*link2_len)*link1_len
        B_13 = link3_mass*link3_r*link1_len
        B_21 = -(link2_mass*link2_r
                 + link3_mass*link2_len)*link1_len
        B_22 = 0.
        B_23 = link3_mass*link3_r*link2_len
        B_32 = -link3_mass*link3_r*link2_len
        B_33 = 0.

        C_1 = (link1_mass*link1_r
               + (link2_mass + link3_mass)*link1_len)*self.g
        C_2 = (link2_mass*link2_r
               + link3_mass*link2_len)*self.g
        C_3 = (link3_mass*link3_r)*self.g

        s1 = np.sin(link1_pos)
        # c1 = np.cos(link1_pos)
        s2 = np.sin(link2_pos)
        # c2 = np.cos(link2_pos)
        s3 = np.sin(link3_pos)
        # c3 = np.cos(link3_pos)
        s21 = np.sin(link2_pos - link1_pos)
        s31 = np.sin(link3_pos - link1_pos)
        s32 = np.sin(link3_pos - link2_pos)
        c21 = np.cos(link2_pos - link1_pos)
        c31 = np.cos(link3_pos - link1_pos)
        c32 = np.cos(link3_pos - link2_pos)

        A = np.array([
            [A_11, A_12*c21, A_13*c31],
            [A_12*c21, A_22, A_23*c32],
            [A_13*c31, A_23*c32, A_33],
        ])
        B = np.array([
            [B_11, B_12*s21*link2_vel, B_13*s31*link3_vel],
            [B_21*s21*link1_vel, B_22, B_23*s32*link3_vel],
            [-B_13*s31*link1_vel, B_32*s32*link2_vel, B_33],
        ])

        # external forces
        C_ext = np.array([C_1*s1, C_2*s2, C_3*s3])

        # add rotational friction
        C_ext -= 2*np.array(link_vel_vector)

        # Convert to expected sign
        M = A
        C = -B
        N = -C_ext

        return M, C, N

    def compute_joints_from_ee_cart_pos(self, ee_cart_pos):
        np.random.seed(self.seed)

        # think about what to do instead of asserting
        # maybe closest point in ws
        assert self.is_cart_point_in_ws(ee_cart_pos)

        link1_len, link2_len, link3_len = self._length_links
        ee_x, ee_y = ee_cart_pos
        # arctan2 backwards b/c of this robot's joint convention
        ee_angle = np.arctan2(ee_x, ee_y)
        ee_len = np.linalg.norm(ee_cart_pos)

        if ee_len == self._ws_radius_max:
            # one solution
            joint1_pos = ee_angle
            joint2_pos = ee_angle
            joint3_pos = ee_angle

        elif ee_len == self._ws_radius_min and self._ws_radius_min > 0.:
            # on minimum radius, but it's not a triangle
            combs = list(product([-1, 1], repeat=self.num_links))
            radii_combs = np.array(combs).dot(self._length_links)
            x_valid_config = np.flatnonzero(radii_combs == ee_len)[0]
            config_min_rad = combs[x_valid_config]

            if config_min_rad[0] == 1:
                # keep direction
                joint1_pos = ee_angle
            else:
                joint1_pos = self.wrap_angle(ee_angle + np.pi)

            if config_min_rad[1] == 1:
                # keep direction
                joint2_pos = ee_angle
            else:
                # reverse direction
                joint2_pos = self.wrap_angle(ee_angle + np.pi)

            if config_min_rad[2] == 1:
                # keep direction
                joint3_pos = ee_angle
            else:
                # reverse direction
                joint3_pos = self.wrap_angle(ee_angle + np.pi)

        else:
            # see if the point is always reachable with the last two links
            sum_link23_len = link2_len + link3_len
            abs_diff_link23_len = np.abs(link3_len - link2_len)

            ws_radius_link23_max = sum_link23_len
            ws_radius_link23_min = abs_diff_link23_len

            if ws_radius_link23_min > np.abs(ee_len - link1_len):
                # feasible limits due to minimum reach of link23 workspace
                # check limits when links 1 and 2 are acting as one link
                sum_link12_len = link1_len + link2_len
                sol = points_of_intersecting_circles(
                    [0., 0.],
                    sum_link12_len,
                    ee_cart_pos,
                    link3_len,
                )
                # just need one solution to calculate angle diff
                sol_x, sol_y = sol[np.random.randint(2)]
                joint2_pos_sol = np.arctan2(sol_x, sol_y)
                joint1_pos_sol = joint2_pos_sol
                # Add np.pi here to preserve the correct range to sample
                ee_angle_opp = self.wrap_angle(ee_angle + np.pi)
                joint_diff = np.abs(
                    self.wrap_angle(joint1_pos_sol - ee_angle_opp)
                )
                # don't wrap these, since we need to preserve the range
                joint1_pos_low = ee_angle_opp - joint_diff
                joint1_pos_high = ee_angle_opp + joint_diff

            elif ws_radius_link23_max < (ee_len + link1_len):
                # feasible limits due to maximum reach of link23 workspace
                # check limits when links 2 and 3 are acting as one link
                sol = points_of_intersecting_circles(
                    [0., 0.],
                    link1_len,
                    ee_cart_pos,
                    sum_link23_len
                )
                # just need one solution to calculate angle diff
                sol_x, sol_y = sol[np.random.randint(2)]
                joint1_pos_sol = np.arctan2(sol_x, sol_y)
                joint_diff = np.abs(self.wrap_angle(joint1_pos_sol - ee_angle))
                # don't wrap these, since we need to preserve the range
                joint1_pos_low = ee_angle - joint_diff
                joint1_pos_high = ee_angle + joint_diff

            else:
                # ee is reachable with any joint1 pos, so pick randomly
                joint1_pos_low = -np.pi
                joint1_pos_high = np.pi

            joint1_pos = self.wrap_angle(np.random.uniform(
                low=joint1_pos_low,
                high=joint1_pos_high
            ))
            link1_x, link1_y = self.compute_links_cart_pos(
                [joint1_pos, 0., 0.]
            )[0]

            # Pick one of two possible solutions
            sol = points_of_intersecting_circles(
                [link1_x, link1_y],
                link2_len,
                ee_cart_pos,
                link3_len
            )
            link2_x, link2_y = sol[np.random.randint(2)]
            # arctan2 backwards b/c of this robot's joint convention
            joint2_pos = np.arctan2(link2_x - link1_x, link2_y - link1_y)
            joint3_pos = np.arctan2(ee_x - link2_x, ee_y - link2_y)

        joint_pos_vec = np.array([joint1_pos, joint2_pos, joint3_pos])

        # Turn off assertion below, but was helpful for debugging
        # test_pts = self.compute_links_cart_pos(joint_pos_vec)
        # assert np.linalg.norm(test_pts[-1] - ee_cart_pos) < 1.e-8

        return joint_pos_vec

    def compute_links_cart_pos(self, joint_pos_vec):
        link1_len, link2_len, link3_len = self._length_links
        x1 = link1_len*np.sin(joint_pos_vec[0])
        y1 = link1_len*np.cos(joint_pos_vec[0])
        x2 = x1 + link2_len*np.sin(joint_pos_vec[1])
        y2 = y1 + link2_len*np.cos(joint_pos_vec[1])
        x3 = x2 + link3_len*np.sin(joint_pos_vec[2])
        y3 = y2 + link3_len*np.cos(joint_pos_vec[2])
        link_cart_pos = np.array([
            [x1, y1],
            [x2, y2],
            [x3, y3],
        ])

        return link_cart_pos

    def get_Jacobian(self, link_pos_vector):
        theta_1, theta_2, theta_3 = link_pos_vector
        link1_len, link2_len, link3_len = self._length_links

        J = np.zeros((2, self.num_links))
        J[0, 0] = link1_len*np.cos(theta_1)
        J[0, 1] = link2_len*np.cos(theta_2)
        J[0, 2] = link3_len*np.cos(theta_3)
        J[1, 0] = -link1_len*np.sin(theta_1)
        J[1, 1] = -link2_len*np.sin(theta_2)
        J[1, 2] = -link3_len*np.sin(theta_3)

        return J

    def get_Jacobians(self, link_pos_vector):
        J3 = self.get_Jacobian(link_pos_vector)
        J2 = J3[:, 0:self.num_links - 1]
        J1 = J3[:, 0:self.num_links - 2]
        return [J1, J2, J3]

    def sample_cart_point_from_ws(self):
        np.random.seed(self.seed)
        # Sample along radius, weighted to de-bias points
        radius_ratio_sq = (self._ws_radius_min / self._ws_radius_max)**2.
        radius_pt = self._ws_radius_max * np.sqrt(
            np.random.uniform(low=radius_ratio_sq, high=1.0)
        )

        rot_pt = 2. * np.pi * np.random.random()

        x_pt = radius_pt*np.cos(rot_pt)
        y_pt = radius_pt*np.sin(rot_pt)
        ee_point = np.array([x_pt, y_pt])
        return ee_point

    def is_cart_point_in_ws(self, point_to_check):
        radius_pt = np.linalg.norm(point_to_check)
        is_in_ws = \
            self._ws_radius_min <= radius_pt and \
            radius_pt <= self._ws_radius_max
        return is_in_ws


class RobotRotational3LinkILinear(RobotRotational3Link):
    @property
    def d_U(self):
        return self.parameters['i_G'].shape[1]

    def __init__(self, **kwargs):
        # default linear interface parameters
        defaults = {'i_G': np.eye(3)}
        if hasattr(self, 'parameters'):
            self.parameters.update(defaults)
        else:
            self.parameters = defaults
        super().__init__(**kwargs)

    def control_interface(self, u=None):
        if u is not None:
            u = self.parameters['i_G'].dot(u)
        return super().control_interface(u)


# ~~~~~~~~~~~ 2 link robots ~~~~~~~~~~~

class RobotRotational2Link(RobotOpenChainArm):
    """ 2-link rotational """

    _num_links = 2
    d_U = 2

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _calculate_workspace(self):
        # Assumes that length links has already been set
        self._ws_radius_max = np.sum(self._length_links)
        self._ws_radius_min = np.abs(
            self._length_links[1] - self._length_links[0]
        )

        # workspace limits for visualization
        self._ws_xlim_viz = np.array([
            -self._ws_radius_max - 1.,
            self._ws_radius_max + 1.
        ])
        self._ws_ylim_viz = np.array([
            -self._ws_radius_max - 1.,
            self._ws_radius_max + 1.
        ])

    def _draw_workspace(self, fig, ax):
        # courtesy of https://stackoverflow.com/questions/22789356/plot-a-donut-with-fill-or-fill-between-use-pyplot-in-matplotlib # noqa: E501
        n, radii = 50, [self._ws_radius_min, self._ws_radius_max]
        theta_ws = np.linspace(0, 2*np.pi, n, endpoint=True)
        x_ws = np.outer(radii, np.cos(theta_ws))
        y_ws = np.outer(radii, np.sin(theta_ws))
        x_ws[1, :] = x_ws[1, ::-1]
        y_ws[1, :] = y_ws[1, ::-1]
        ax.fill(np.ravel(x_ws), np.ravel(y_ws), alpha=0.2, color='black')

    def compute_dynamics(self, link_pos_vector, link_vel_vector):

        link1_pos, link2_pos = link_pos_vector
        link1_vel, link2_vel = link_vel_vector

        link1_len, link2_len = self._length_links
        link1_r, link2_r = self._massloc_links
        link1_inertia, link2_inertia = self._inertia_links
        link1_mass, link2_mass = self._mass_links

        # Implemented from:
        # "A Mathematical Introduction to Robotic Manipulation"
        # R. M. Murray, Z. Li, S. S. Sastry
        # Defined on pages 164-165

        alpha = link1_inertia + \
            link2_inertia + \
            link1_mass*link1_r**2. + \
            link2_mass*(link1_len**2. + link2_r**2.)
        beta = link2_mass*link1_len*link2_r
        delta = link2_inertia + \
            link2_mass*link2_r**2.

        # s1 = np.sin(link1_pos)
        c1 = np.cos(link1_pos)
        s2 = np.sin(link2_pos)
        c2 = np.cos(link2_pos)
        # s12 = np.sin(link1_pos + link2_pos)
        c12 = np.cos(link1_pos + link2_pos)

        # inertia matrix
        M = np.array([
            [alpha + 2.*beta*c2, delta + beta*c2],
            [delta + beta*c2, delta]
        ])

        # coriolis matrix
        C = np.array([
            [-beta*s2*link2_vel, -beta*s2*(link1_vel + link2_vel)],
            [beta*s2*link1_vel, 0.]
        ])

        # external force vector
        N = np.zeros(2)

        # add gravity terms - from asada textbook
        N += np.array([
            link1_mass*self.g*link1_r*c1
            + link2_mass*self.g*(link2_r*c12 + link1_len*c1),
            link2_mass*self.g*link2_r*c12,
        ])

        # add rotational friction
        N += 2*np.array(link_vel_vector)

        return M, C, N

    def compute_joints_from_ee_cart_pos(self, ee_cart_pos):
        np.random.seed(self.seed)

        # think about what to do instead of asserting
        # maybe closest point in ws
        assert self.is_cart_point_in_ws(ee_cart_pos)

        link1_len, link2_len = self._length_links
        ee_x, ee_y = ee_cart_pos
        ee_len = np.linalg.norm(ee_cart_pos)

        if ee_len == 0.:
            # Degenerate case with a family of solutions
            # sample joint1 uniformly
            joint1_pos = np.random.uniform(low=-np.pi, high=np.pi)

            # just get the cartesian position of the first link
            # could probably use a method to do forward kinematics
            # without giving the entire joint positions
            link1_x, link1_y = self.compute_links_cart_pos([joint1_pos, 0.])[0]
        else:
            # Two solutions
            sol = points_of_intersecting_circles(
                [0., 0.],
                link1_len,
                ee_cart_pos,
                link2_len,
            )
            link1_x, link1_y = sol[np.random.randint(2)]
            joint1_pos = np.arctan2(link1_y, link1_x)

        joint12_sum = np.arctan2(ee_y - link1_y, ee_x - link1_x)
        joint2_pos = self.wrap_angle(joint12_sum - joint1_pos)

        joint_pos_vec = np.array([joint1_pos, joint2_pos])

        # Turn off assertion below, but was helpful for debugging
        # test_pts = self.compute_links_cart_pos(joint_pos_vec)
        # assert np.linalg.norm(test_pts[-1] - ee_cart_pos) < 1.e-8

        return joint_pos_vec

    def compute_links_cart_pos(self, joint_pos_vec):
        link1_len, link2_len = self._length_links
        x1 = link1_len*np.cos(joint_pos_vec[0])
        y1 = link1_len*np.sin(joint_pos_vec[0])
        x2 = x1 + link2_len*np.cos(joint_pos_vec[0] + joint_pos_vec[1])
        y2 = y1 + link2_len*np.sin(joint_pos_vec[0] + joint_pos_vec[1])

        link_cart_pos = np.array([
            [x1, y1],
            [x2, y2],
        ])
        return link_cart_pos

    def get_Jacobian(self, link_pos_vector):
        theta_1, theta_2 = link_pos_vector
        link1_len, link2_len = self._length_links

        J = np.zeros((2, self.num_links))
        J[0, 0] = -link1_len*np.sin(theta_1) + \
                  -link2_len*np.sin(theta_1)*np.cos(theta_2) + \
                  -link2_len*np.cos(theta_1)*np.sin(theta_2)
        J[0, 1] = -link2_len*np.cos(theta_1)*np.sin(theta_2) + \
                  -link2_len*np.sin(theta_1)*np.cos(theta_2)
        J[1, 0] = link1_len*np.cos(theta_1) + \
            link2_len*np.cos(theta_1)*np.cos(theta_2) + \
            -link2_len*np.sin(theta_1)*np.sin(theta_2)
        J[1, 1] = -link2_len*np.sin(theta_1)*np.sin(theta_2) + \
            link2_len*np.cos(theta_1)*np.cos(theta_2)

        return J

    def get_Jacobians(self, link_pos_vector):
        J2 = self.get_Jacobian(link_pos_vector)
        theta_1, _ = link_pos_vector
        link1_len, _ = self._length_links
        J1 = np.zeros((2, self.num_links - 1))
        J1[0, 0] = -link1_len*np.sin(theta_1)
        J1[1, 0] = link1_len*np.cos(theta_1)
        return [J1, J2]

    def sample_cart_point_from_ws(self):
        np.random.seed(self.seed)
        # Sample along radius, weighted to de-bias points
        radius_ratio_sq = (self._ws_radius_min / self._ws_radius_max)**2.
        radius_pt = self._ws_radius_max * np.sqrt(
            np.random.uniform(low=radius_ratio_sq, high=1.0)
        )

        rot_pt = 2. * np.pi * np.random.random()

        x_pt = radius_pt*np.cos(rot_pt)
        y_pt = radius_pt*np.sin(rot_pt)
        ee_point = np.array([x_pt, y_pt])
        return ee_point

    def is_cart_point_in_ws(self, point_to_check):
        radius_pt = np.linalg.norm(point_to_check)
        is_in_ws = \
            self._ws_radius_min <= radius_pt and \
            radius_pt <= self._ws_radius_max
        return is_in_ws


class RobotRotational2LinkILinear(RobotRotational2Link):

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
