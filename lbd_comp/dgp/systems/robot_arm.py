import abc
from collections import deque
import numpy as np
from .template import System

from matplotlib import pyplot as plt
import matplotlib.animation as animation

# we need to stay only within dgp (for codalab bundling)
# copied the robot_high_level_controller to bottom of this file
# from baselines.robot_control_baselines import robot_high_level_controller


class RobotOpenChainArm(System):
    """
    Implements open-chain planar robot arms.
    Open-chain means each link is only attached to (at most) one other link.
    """

    # if none, defaults to 4 * numlinks
    # that is the shape of the default participant observations
    d = None

    # class variables
    timegrid_min = 0.
    timegrid_max = 2.
    timegrid_steps = 201
    timegrid = np.linspace(
        timegrid_min,
        timegrid_max,
        timegrid_steps,
    )
    timegrid_dt = (timegrid_max - timegrid_min)/(timegrid_steps - 1)

    g = 9.81

    @property
    def num_links(self):
        return self._num_links

    @property
    def d_X(self):
        """ State per default will be the joint positions and velocities """
        return self.num_links * 2

    @property
    def mask(self):
        # Default to fully observable for now
        mask = tuple(range(self.d_X))
        return mask

    @property
    def current_state(self):
        """ Provides the current joint state at this timestep """
        curr_state = self._X[:, self._timestep * self._resolution]
        return curr_state

    @property
    def current_joint_posvel(self):
        """ Convenience method that separates the joint positions from vels """
        curr_state = self.current_state
        link_positions = curr_state[:(self.num_links)]
        link_vels = curr_state[(self.num_links):]
        return link_positions, link_vels

    @property
    def current_jacobian(self):
        """ Returns the Jacobian for the robot at this timestep """
        link_positions, _ = self.current_joint_posvel
        jacobian = self.get_Jacobian(link_positions)
        return jacobian

    @property
    def current_ee_cart_posvel(self):
        """ Returns the end effector tip position and vel
        in cartesian frame at this timestep """
        curr_joint_pos, curr_joint_vel = self.current_joint_posvel
        ee_cart_pos, ee_cart_vel = self.compute_ee_cart_posvel(
            curr_joint_pos, curr_joint_vel)
        return ee_cart_pos, ee_cart_vel

    @property
    def current_noisefree_participant_observation(self):
        """
        Convenience function that simply returns the current observation
        (coarse timestep).
        This bypasses noise, which can be accessed via indexing """
        curr_obsv = self._observations[:, self._timestep]
        return curr_obsv

    def get_cart_pos(self):
        """ Same as current_ee_cart_posvel,
        but returns as one vector (for backward compatibility) """
        ee_cart_pos, ee_cart_vel = self.current_ee_cart_posvel
        ee_cart_posvel_as_vec = np.hstack((ee_cart_pos, ee_cart_vel))
        return ee_cart_posvel_as_vec

    @abc.abstractmethod
    def compute_dynamics(self, link_pos_vector, link_vel_vector):
        """ Must be specified based on the link configuration """
        """ should return M, C, N """
        """ M is a (num_links, num_links) matrix """
        """ C is a (num_links, num_links) matrix """
        """ N is a (num_links, 1) vector """

    @abc.abstractmethod
    def compute_links_cart_pos(self, joint_pos_vec):
        """ Converts a joint pos representation to link cartesian pos """
        """ returns a (num_links, 2) matrix """

    @abc.abstractmethod
    def get_Jacobian(self, link_pos_vector):
        """ Returns the Jacobian for the robot at this timestep """
        """ returns a (2, num_links) matrix """

    @abc.abstractmethod
    def compute_joints_from_ee_cart_pos(self, ee_cart_pos):
        """ Returns a joint position vector """
        """ that yields the given ee cartesian position """
        """ May not always return a unique joint position vector """
        """ If multiple solutions are possible, they are sampled uniformly """
        """ Returns a vector of length num_links"""

    @abc.abstractmethod
    def _calculate_workspace(self):
        """ Sets robot-specific workspace variables. """

    # not an abstract method since it's strictly not required for running
    # child classes should implement this
    def _draw_workspace(self, fig, ax):
        """ Draws the workspace on a figure. """
        pass

    @abc.abstractmethod
    def sample_cart_point_from_ws(self):
        """ Returns a list/vector of length 2 (cartesian point) """
        """ sampled uniformly from robot workspace """

    @abc.abstractmethod
    def is_cart_point_in_ws(self, point_to_check):
        """ Given cartesian point, check whether it is in the workspace """
        """ points_to_check should be a list/vector of length 2 """
        """ returns an Boolean (True if ee can reach it) """

    def compute_ee_cart_posvel(self, joint_pos_vec, joint_vel_vec):
        """ Returns the end effector tip position
        and vel in cartesian frame """
        ee_cart_pos = self.compute_links_cart_pos(joint_pos_vec)[-1]

        J = self.get_Jacobian(joint_pos_vec)
        vel = J@joint_vel_vec.reshape(-1, 1)
        ee_cart_vel = vel.flatten()

        return ee_cart_pos, ee_cart_vel

    def step(self, u=None):
        """
        Keep track of cart_pos here.
        """
        super().step(u=u)
        self.log_noisefree_participant_observation()

    def log_noisefree_participant_observation(self):
        """
        This is called after each step,
        to log an observation.
        Defaults to joints positions and speeds.
        Can be overwritten by robots to make other stuff observable too.
        Observations are stored non-noisy,
        while accessing the system object via indexing will
        take care of observation noise.
        """
        curr_joint_pos, curr_joint_vel = self.current_joint_posvel
        # check if we want joint observations in joint (True)
        # or cartesian space (False)
        if self._is_joint_obsv_in_joint_space:
            # end effector x, y pos and x, y vel
            ee_cart_pos, ee_cart_vel = self.current_ee_cart_posvel

            # ordering is as follows:
            # ee X position (in cartesian space)
            # ee Y position (in cartesian space)
            # ee X velocity (in cartesian space)
            # ee Y velocity (in cartesian space)
            # joint 1 position (in joint space)
            # joint ...
            # joint N position (in joint space)
            # joint 1 velocity (in joint space)
            # joint ...
            # joint N velocity (in joint space)

            obsv = np.hstack(
                (ee_cart_pos, ee_cart_vel, curr_joint_pos, curr_joint_vel))
        else:
            positions = self.compute_links_cart_pos(curr_joint_pos)
            # for convenience, the first two dimensions are always ee X,Y
            # X, Y ee
            # X1, Y1 joint before that
            # ...
            pos = positions[::-1, :].flatten()
            Js = self.get_Jacobians(curr_joint_pos)
            vel = []
            for k in range(self.num_links - 1, -1, -1):
                vel += [
                    Js[k].dot(
                        curr_joint_vel.reshape(-1, 1)[:k + 1, :]).flatten()]
            obsv = np.hstack([pos] + vel)

        self._observations[:, self._timestep] = obsv

    def __getitem__(self, indices):
        """
        Safeguard indexing to only return coarse-grained
        cart_pos data to participants.
        It is logged as we apply step to the system.
        """
        np.random.seed(self.seed)
        return self.measurement_noise(self._observations)[indices]

    @property
    def shape(self):
        """
        Return the shape of the coarse-grained participant observations.
        """
        return self._observations.shape

    @staticmethod
    def wrap_angle(angle):
        """
        Wraps an angle to the range -np.pi to +np.pi
        """
        # Courtesy of https://stackoverflow.com/a/15927914
        wrapped_angle = (angle + np.pi) % (2. * np.pi) - np.pi
        return wrapped_angle

    auto_init_modes = [
        'zeros',
        'random_ws_no_vel',
        'from_ws_point_no_vel'
    ]

    def __init__(self,
                 X0,
                 **kwargs):

        # Set things here
        defaults = {
            'length_links': None,
            'massloc_links': None,
            'mass_links': None,
            'inertia_links': None,
            'is_joint_obsv_in_joint_space': False,
            'joint_limits': None,
        }
        if hasattr(self, 'parameters'):
            self.parameters.update(defaults)
        else:
            self.parameters = defaults
        defaults.update(kwargs['parameters'])
        kwargs['parameters'] = defaults

        self.seed = kwargs.get('seed', 42)

        # unpack parameters
        # not yet in self – only once super ctor is called further below
        length_links = kwargs['parameters']['length_links']
        massloc_links = kwargs['parameters']['massloc_links']
        mass_links = kwargs['parameters']['mass_links']
        inertia_links = kwargs['parameters']['inertia_links']
        joint_limits = kwargs['parameters']['joint_limits']
        self._is_joint_obsv_in_joint_space = kwargs['parameters'][
            'is_joint_obsv_in_joint_space']

        # Create link properties
        # length of each link
        if length_links is not None:
            # verify correct
            assert len(length_links) == self.num_links
            self._length_links = np.array(length_links)
        else:
            self._length_links = np.ones(self.num_links)

        # location of center-of-mass for each link
        if massloc_links is not None:
            # verify correct
            assert len(massloc_links) == self.num_links
            self._massloc_links = np.array(massloc_links)
        else:
            # default to halfway along link
            self._massloc_links = self._length_links / 2.

        # mass of each link
        if mass_links is not None:
            # verify correct
            assert len(mass_links) == self.num_links
            self._mass_links = np.array(mass_links)
        else:
            self._mass_links = np.ones(self.num_links)

        # rotational inertia for each link
        # note -- inertia here is w/r/t z-axis since these are all planar arms
        if inertia_links is not None:
            # verify correct
            assert len(inertia_links) == self.num_links
            self._inertia_links = np.array(inertia_links)
        else:
            self._inertia_links = np.ones(self.num_links)

        # set workspace config, after parsing robot geometry
        # needed for some of the auto init modes
        self._calculate_workspace()

        # Process inputs for auto init modes
        if isinstance(X0, str):
            # convenience wrapper
            X0 = {'mode': X0}

        if isinstance(X0, dict):
            # auto init modes
            mode = X0['mode']
            assert mode in self.auto_init_modes, \
                f"Auto init mode '{mode}' is not recognized."

            if mode == 'zeros':
                X0 = np.zeros(self.d_X)

            elif mode == 'random_ws_no_vel':
                ws_pt = self.sample_cart_point_from_ws()
                joint_pos = self.compute_joints_from_ee_cart_pos(ws_pt)
                joint_vel = np.zeros(self.num_links)
                X0 = np.hstack((joint_pos, joint_vel))

            elif mode == 'from_ws_point_no_vel':
                ws_pt = X0['ws_point']
                joint_pos = self.compute_joints_from_ee_cart_pos(ws_pt)
                joint_vel = np.zeros(self.num_links)
                X0 = np.hstack((joint_pos, joint_vel))

            else:
                raise NotImplementedError

        # Ensure X0 is consistent with the system
        # We'll assume that the first half are the joint positions,
        # and the second half are the joint velocities
        exp_len_X0 = self.num_links * 2
        assert len(X0) == exp_len_X0, (
            f"Expected 'X0' to have length of {exp_len_X0}, "
            f"but it has length {len(X0)}.")

        if joint_limits is not None:
            # verify correct
            assert len(joint_limits) == self.num_links
            self._joint_limits = joint_limits
        else:
            self._joint_limits = None

        # Hand off to parent ctor
        super().__init__(X0, **kwargs)

        # robot observation defaults to all joints posvel
        if self._is_joint_obsv_in_joint_space:
            self.d = 4 + self.num_links * 2
        else:
            self.d = 4 * self.num_links
        self._observations = np.empty(
            (self.d, self._timesteps))
        self._observations.fill(np.nan)
        self.log_noisefree_participant_observation()

        # Use a different ODE solver
        self._ode_solver_method = "LSODA"

    def check_joint_limits(self, link_pos_vec, link_vel_vec, link_pos_dot):
        link_pos_vec = link_pos_vec + self._length_links
        for idx, el in enumerate(link_pos_vec):
            # Reached positive joint limit
            pos = ((el >= self._joint_limits[idx])
                   and link_vel_vec[idx] > 0)
            # Reached negative joint limit
            neg = ((el <= -self._joint_limits[idx])
                   and link_vel_vec[idx] < 0)
            # If any joint limit reached: set vel to 0
            if pos or neg:
                link_pos_dot[idx] = 0.
        return link_pos_dot

    def rhs(self, t, x, u):
        """ u is the joint torques """
        """ u is after the control_interface is applied """

        # Note: this is expensive to do when integrating,
        # as this is tested over and over again,
        # we now do it as part of the robot arm's control_interface
        # if u is None:
        #     u = [0.]*self.d_U
        # Note: this is expensive to do when integrating,
        # as it creates a copy on each call;
        # we now do it as part of the robot arm's control_interface
        # joint_torques = np.nan_to_num(u)
        joint_torques = u

        # Unpack state vector
        # - assumes joint angles are first, then joint velocities
        link_pos_vec = x[:(self.num_links)]
        link_vel_vec = x[(self.num_links):]
        M, C, N = self.compute_dynamics(link_pos_vec, link_vel_vec)

        # Construct derivative of state
        link_pos_dot = link_vel_vec
        # both alternatives below are better unless we really need the inverse
        # re speed and numerical precision/stability
        # link_vel_dot = np.matmul(
        # np.linalg.inv(M),
        # joint_torques - np.matmul(C, link_vel_vec) - N)
        # link_vel_dot = np.linalg.lstsq(
        # M,
        # joint_torques - np.matmul(C, link_vel_vec) - N,
        # rcond=None)[0]
        link_vel_dot = np.linalg.solve(
            M,
            joint_torques - np.matmul(C, link_vel_vec) - N)

        if self._joint_limits is not None:
            link_pos_dot = self.check_joint_limits(link_pos_vec,
                                                   link_vel_vec, link_pos_dot)

        x_dot = np.concatenate((link_pos_dot, link_vel_dot))
        return x_dot

    def control_interface(self, u=None):
        if u is None:
            u = [0.]*self.d_U
        u = np.nan_to_num(u, copy=False)
        np.clip(1000 * u, -1e4, 1e4, out=u)
        return u

    # Visualization methods -- consider moving to a separate module
    # in order to separate visualization from control
    def show_traj_animation(self, *args, **kwargs):
        fig, ax, ani = self._traj_animation(*args, **kwargs)  # noqa: F841
        plt.show()

    def save_traj_animation(self, animation_path, *args, **kwargs):
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=30, bitrate=1800)
        fig, ax, ani = self._traj_animation(*args, **kwargs)
        ani.save(animation_path, writer=writer)
        plt.close(fig)

    def _traj_animation(self,
                        ref_traj_cart_pos=None,
                        annot_text=None,
                        hidden=False):
        """ Low-level animator """
        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(
            autoscale_on=False,
            xlim=self._ws_xlim_viz,
            ylim=self._ws_xlim_viz,
        )
        if not hidden:
            ax.grid()
        else:
            ax.axis('off')
        ax.set_aspect('equal')

        joint_pos_hist = np.array([
            self._X[
                n,
                :(self._timestep*self._resolution + 1):self._resolution
            ] for n in range(self.num_links)
        ]).T
        link_cart_pos_hist = np.array(
            [self.compute_links_cart_pos(i) for i in joint_pos_hist]
        )

        # Draw initial config
        plt.plot(
            np.r_[0., link_cart_pos_hist[0][:, 0]],
            np.r_[0., link_cart_pos_hist[0][:, 1]],
            c='black',
            lw=2,
            marker='o',
            ls=':',
        )

        # Draw reference trajectory (including if it's in the future)
        draw_traj_trace = False
        desired_traj_color = 'green'
        if ref_traj_cart_pos is not None:
            # convert to np array, in case it's passed as a list
            ref_traj_cart_pos = np.array(ref_traj_cart_pos)
            if ref_traj_cart_pos.ndim == 1:
                assert len(ref_traj_cart_pos) == 2
                desired_ee_x_end, desired_ee_y_end = ref_traj_cart_pos
            else:
                desired_ee_x_end, desired_ee_y_end = ref_traj_cart_pos[-1]
                if np.unique(ref_traj_cart_pos, axis=0).shape[0] > 1:
                    # entire trajectory given, plot the trace
                    draw_traj_trace = True
                    desired_ee_x, desired_ee_y = ref_traj_cart_pos.T
                    # plot trace
                    plt.plot(
                        desired_ee_x,
                        desired_ee_y,
                        lw=1.5,
                        ls=':',
                        c=desired_traj_color
                    )

            # plot endpoint
            plt.scatter(
                desired_ee_x_end,
                desired_ee_y_end,
                marker='*',
                s=64,
                c=desired_traj_color,
            )

        # draw workspace
        if not hidden:
            self._draw_workspace(fig, ax)

        # draw annotation text
        if annot_text is not None:
            ax.text(0.5, 0.9, annot_text, transform=ax.transAxes)

        line, = ax.plot([], [], 'o-', lw=2, c='#1f77b4')
        trace, = ax.plot([], [], ',-', lw=1, c='#ff7f0e')
        if draw_traj_trace:
            traj_trace, = ax.plot(
                [],
                [],
                markersize=10,
                marker='o',
                markerfacecolor='none',
                c=desired_traj_color
            )
        time_template = 'time = %.1fs'
        time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
        history_len = 500
        history_x = deque(maxlen=history_len)
        history_y = deque(maxlen=history_len)
        if not hidden:
            ax.set_xlabel('X position (m)')
            ax.set_ylabel('Y position (m)')

        dt = self.timegrid_dt
        if not hidden:
            playback_rate = 0.25  # 0.25 -> 1/4 slower than real time
        else:
            playback_rate = 0.42
        ani_interval_ms = dt*1000/playback_rate

        skip_frames = True
        if skip_frames:
            link_cart_pos_hist_ani = link_cart_pos_hist[::2]
            frame_count = self._timestep//2 + 1
            dt_ani = dt*2.
            if draw_traj_trace:
                desired_ee_x_ani = desired_ee_x[::2]
                desired_ee_y_ani = desired_ee_y[::2]
        else:
            link_cart_pos_hist_ani = link_cart_pos_hist
            frame_count = self._timestep + 1
            dt_ani = dt
            if draw_traj_trace:
                desired_ee_x_ani = desired_ee_x
                desired_ee_y_ani = desired_ee_y

        def animate(i):
            thisx = np.r_[0., link_cart_pos_hist_ani[i][:, 0]]
            thisy = np.r_[0., link_cart_pos_hist_ani[i][:, 1]]

            if i == 0:
                history_x.clear()
                history_y.clear()

            history_x.appendleft(thisx[-1])
            history_y.appendleft(thisy[-1])

            line.set_data(thisx, thisy)
            trace.set_data(history_x, history_y)
            if not hidden:
                time_text.set_text(time_template % (i*dt_ani))
            ani_out = line, trace, time_text

            if draw_traj_trace:
                traj_trace_x = desired_ee_x_ani[i]
                traj_trace_y = desired_ee_y_ani[i]
                traj_trace.set_data(traj_trace_x, traj_trace_y)

                ani_out = ani_out + (traj_trace,)

            return ani_out

        ani = animation.FuncAnimation(  # noqa: F841
            fig,
            animate,
            frame_count,
            interval=ani_interval_ms,
            blit=True
        )

        return fig, ax, ani
