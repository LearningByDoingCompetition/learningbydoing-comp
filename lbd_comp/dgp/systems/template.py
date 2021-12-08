import abc
from hashlib import blake2b
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from warnings import warn


class System(metaclass=abc.ABCMeta):
    """
    Template for system classes.
    """

    _resolution = 100
    _timestep = 0

    # TODO: sanity check noisy version reproducible
    # TODO: is initial value noisy?

    @property
    def mask(self):
        raise NotImplementedError

    @property
    def target(self):
        raise NotImplementedError

    @property
    def timegrid(self):
        raise NotImplementedError

    @property
    def d_U(self):
        raise NotImplementedError

    @property
    def d_X(self):
        raise NotImplementedError

    @property
    def ode_solver_method(self):
        return self._ode_solver_method

    def __init__(self,
                 X0,
                 name='',
                 mask=None,
                 parameters=None,
                 seed=42,
                 target=None):
        if mask:
            self.mask = mask
        if parameters:
            self.parameters.update(parameters)
        if target is not None:
            self.target = target
        self.name = name

        self.seed = seed

        # timegrid includes start and endpoint
        self._timesteps = len(self.timegrid)
        self._X = np.empty((self.d_X,
                            self._resolution * (self._timesteps - 1) + 1))
        self._X.fill(np.nan)

        self._U = np.empty((self.d_U, self._timesteps - 1))
        self._U.fill(np.nan)
        self._U.flags.writeable = False

        self.d = len(self.mask)

        self._X[:, 0] = X0

        # Default ODE solver
        self._ode_solver_method = "RK45"

    def __repr__(self):
        return self[:].__repr__()

    def __str__(self):
        return str(self.__class__)

    def __hash__(self):
        def hashprep(x):
            if isinstance(x, list) or isinstance(x, tuple):
                return str(tuple(hashprep(xx) for xx in x))
            if isinstance(x, dict):
                return hashprep(
                    tuple(x[k] for k in sorted(x))
                    + tuple(x))
            if isinstance(x, np.ndarray):
                return x.flatten().data.tobytes()
            return str(x)
        # TODO: check if it can be reverse engineered or a problem for
        # revealing something to participant unintendedly
        # IDEA: for participant version / starter kit use different hash
        # function to not reveal anything
        try:
            # if self.target is set to a value or np.nan, we are in track 1
            # else we are in track 2
            ids = (type(self).__name__,
                   self.mask,
                   self.parameters,
                   self.target,
                   self._X[:, 0],
                   )
        except NotImplementedError:
            ids = (type(self).__name__,
                   self.mask,
                   self.parameters,
                   self._X[:, 0],
                   )
        return int(
            blake2b(hashprep(ids).encode('utf8')).hexdigest(),
            16)

    @property
    def t(self):
        return self.timegrid[self._timestep]

    @property
    def t_next(self):
        return self.timegrid[self._timestep + 1]

    def impulsecontrol(self, u=None):
        """
        Submit an impulse control signal and forward simulate the system.
        Track 1.
        """
        if self._timestep > 0:
            warn('Impulse control cannot be applied. System already at t > 0.')
            return
        # impulse for 3 "participant" steps
        for _ in range(3):
            self.step(u=u)
        # simulate remainder
        while self._timestep < self._timesteps - 1:
            self.step()

    def control_interface(self, u=None):
        return u

    def step(self, u=None):
        if u is not None:
            self._U.flags.writeable = True
            self._U[:, self._timestep] = u
            self._U.flags.writeable = False
        u = self.control_interface(u)

        sol = solve_ivp(self.rhs,
                        [self.t, self.t_next],
                        self._X[:, self._timestep * self._resolution],
                        t_eval=np.linspace(self.t,
                                           self.t_next,
                                           self._resolution + 1),
                        args=(u,),
                        method=self.ode_solver_method)
        start = self._timestep * self._resolution + 1
        end = start + self._resolution
        try:
            self._X[:, start:end] = sol.y[:, 1:]
        except Exception:
            raise Exception(
                'Robot broken. Among others, this could be due to an extended '
                'period of setting extreme control inputs. – '
                '"I suggest another strategy, Artoo: let the Wookie win."')

        self._timestep += 1

    def measurement_noise(self, X):
        return X

    def __getitem__(self, indices):
        """
        Safeguard indexing to only return coarse-grained data to participants.
        """
        np.random.seed(self.seed)
        return self.measurement_noise(
            self._X[self.mask, ::self._resolution])[indices]

    @property
    def shape(self):
        """
        Return the shape of the coarse-grained data.
        """
        return self._X[self.mask, ::self._resolution].shape

    @property
    def U(self):
        self._U.flags.writeable = False
        return self._U

    @abc.abstractmethod
    def rhs(self, t, x, u):
        """
        RHS equations
        """

    def getDF(self):
        trackone = True
        try:
            # is set to a target value or np.nan in track 1
            # else we are in track 2
            self.target
        except NotImplementedError:
            trackone = False
        # !-suffix to ensure is being read as string
        sysid = pd.DataFrame(data=(f"{hash(self)}!", )*self._timesteps,
                             columns=('ID', ))
        sysname = pd.DataFrame(data=(self.name, )*self._timesteps,
                               columns=('System', ))
        t = pd.DataFrame(data=self.timegrid,
                         columns=('t', )).round(decimals=6)
        U = pd.DataFrame(data=self.U.T,
                         columns=[f'U{k + 1}' for k in range(self.U.shape[0])]
                         ).round(decimals=6)
        if trackone:
            target = pd.DataFrame(data=(self.target, )*self._timesteps,
                                  columns=('target', )).round(decimals=6)
            Y = pd.DataFrame(data=self[0, :], columns=('Y', )
                             ).round(decimals=6)
            X = pd.DataFrame(data=self[1:, :].T,
                             columns=[f'X{k}' for k in range(1, self.d)]
                             ).round(decimals=6)
            df = sysid.join(sysname).join(t).join(Y).join(
                target).join(X).join(U)
        else:
            # by convention
            labels_pos = ['X', 'Y'] + [f"{i}{j}"
                                       for j in range(1, self.num_links)
                                       for i in ["X", "Y"]]
            labels_vel = [f"d{k}" for k in labels_pos]
            X = pd.DataFrame(data=self[:].T,
                             columns=labels_pos + labels_vel
                             ).round(decimals=6)
            df = sysid.join(sysname).join(t).join(X).join(U)
        return df[:self._timestep + 1]
