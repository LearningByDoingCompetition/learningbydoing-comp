"""
Controller template.
"""

from contextlib import contextmanager
import joblib
import numpy as np
from pathlib import Path
import signal
import pickle
import json

class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException()
    signal.signal(signal.SIGALRM, signal_handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def to_polar(x, v):
    tt = np.arctan2(x[0], -x[1])
    v2 = np.sum(v**2)
    w = v2/(x[0]*v[1] - x[1]*v[0]) if v2 > 0 else 0.0
    return tt, w

def to_angles(x1, x2):
    x21 = x2 - x1
    tt1 = np.arctan2(x1[0], -x1[1])
    tt2 = np.arctan2(x21[0], -x21[1])
    return np.array([tt1, tt2])

def get_x1(x, l1, l2):
    d = np.sqrt(np.sum(x**2))
    l = (l1**2 - l2**2 + d**2)/2/d
    l = np.clip(l, -l1, l1)
    h = np.sqrt(l1**2 - l**2)
    
    center = x*l/d
    side = np.array([x[1], -x[0]])*h/d
    return np.stack((center + side, center - side), axis=0)

class controller(object):

    def __init__(self, system, d_control):
        """
        Entry point, called once when starting the controller for a newly
        initialized system.

        Input:
            system - holds the identifying system name;
                     all evaluation systems have corresponding training data;
                     you may use this name to instantiate system specific
                     controllers

            d_control  - indicates the control input dimension
        """
        self.system = system
        
        param_variation, input_variation, self.robot_type = system.split('-')

        if self.robot_type == "bumblebee":
            K = np.loadtxt(
                Path(__file__).parent / "{}.txt".format(system))
            self.controller_params = (K[:,:-1], K[:,-1])
        elif self.robot_type == "beetle":
            if param_variation == "great":
                self.params = (1.5, 1.5, 3.09)
            elif param_variation == "rebel":
                self.params = (2.0, 1.0, 5.35)
            else:
                raise ValueError("Invalid param variation:", param_variation)
            
            K = np.loadtxt(     # input matrices
                Path(__file__).parent / "{}.txt".format(system))
            with open(Path(__file__).parent / "beetle-pd.txt", "r") as file:
                pd_params = json.load(file)["{}-{}".format(param_variation, input_variation)]
            
            self.controller_params = (K[:,2:].T, K[:,:2], *pd_params)
            self.controller_state = None

            num_inputs = self.controller_params[0].shape[0]
            assert num_inputs == d_control, "Incorrect number of inputs: Expected {}, got {}".format(d_control, num_inputs)
        elif self.robot_type == "butterfly":
            if param_variation == "talented":
                self.params = (1.0, 1.0, 1.0, 0.614, 0.215)
            elif param_variation == "thoughtful":
                self.params = (0.5, 0.5, 2.0, 0.539, 0.531)
            else:
                raise ValueError("Invalid param variation:", param_variation)
            
            K = np.loadtxt(     # input matrices
                Path(__file__).parent / "{}.txt".format(system))
            with open(Path(__file__).parent / "butterfly-pd.txt", "r") as file:
                pd_params = json.load(file)["{}-{}".format(param_variation, input_variation)]
            
            self.controller_params = (K[:,3:].T, K[:,:3], *pd_params)
            self.controller_state = None

            num_inputs = self.controller_params[0].shape[0]
            assert num_inputs == d_control, "Incorrect number of inputs: Expected {}, got {}".format(d_control, num_inputs)
        else:
            raise ValueError("Invalid robot type:", self.robot_type)
        self.d_control = d_control


    def get_input(self, state, position, target):
        """
        This function is called at each time step and expects the next
        control input to apply to the system as return value.

        Input: (all column vectors, if default wrapcontroller.py is used)
            state - vector representing the current state of the system;
                    by convention the first two entries always correspond
                    to the end effectors X and Y coordinate;
                    the state variables are in the same order as in the
                    corresponding training data for the current system
                    with name self.system
            position - vector of length two representing the X and Y
                       coordinates of the current position
            target - vector of length two representing the X and Y
                     coordinates of the next steps target position
        """

        if self.robot_type == "bumblebee":
            K, k = self.controller_params
            x = state.flatten()[[0, 1, 4, 5]]
            x[:2] -= target.flatten()
            u = - k - np.dot(K, x)
            return u
        elif self.robot_type == "beetle":
            # Get Controller Params
            l1, l2, mu1 = self.params
            Rt, Kg, Kp, Kd = self.controller_params

            # Get State
            x = state.flatten()
            x2, x1, v2, v1 = x[:2], x[2:4], x[4:6], x[6:8]
            
            tt1, w1 = to_polar(x1, v1)
            tt2, w2 = to_polar(x2 - x1, v2 - v1)
            tt, w = np.array([tt1, tt2]), np.array([w1, w2])
            tt21 = tt2 - tt1

            s1, c1 = np.sin(tt1), np.cos(tt1)
            s2, c2 = np.sin(tt2), np.cos(tt2)

            # Get Reference
            x2ref = target.flatten()
            x2ref2 = np.sum(x2ref**2)
            if x2ref2 < 1e-5:
                x1ref = x1 # + 0.01*v1
                self.controller_state = None
            else:
                x1ref = get_x1(x2ref, l1, l2)
                if self.controller_state is None:
                    x1err = np.sum((x1ref - x1)**2, axis=1)
                else:
                    x1prev = self.controller_state
                    x1err = np.sum((x1ref - x1prev)**2, axis=1)
                
                idx = np.argmin(x1err, axis=0)
                x1ref = x1ref[idx]
                self.controller_state = x1ref

            ttref = to_angles(x1ref, x2ref)

            # Controller
            tterr = np.mod(tt - ttref + np.pi, 2*np.pi) - np.pi
            
            tg = 9.81/l1*np.array([mu1*s1, s2])  # gravity comp
            ut = np.dot(Kg, tg - Kp*tterr - Kd*w)
            return np.dot(Rt, ut)

        elif self.robot_type == "butterfly":
            # Get Controller Params
            l1, l2, l3, j12, j13 = self.params
            Rt, Kg, Kp, Kd, Km = self.controller_params

            # Get State
            x = state.flatten()
            x3, x2, x1, v3, v2, v1 = x[:2], x[2:4], x[4:6], x[6:8], x[8:10], x[10:12]
            
            tt1, w1 = to_polar(x1, v1)
            tt2, w2 = to_polar(x2 - x1, v2 - v1)
            tt3, w3 = to_polar(x3 - x2, v3 - v2)
            tt, w = np.array([tt1, tt2, tt3]), np.array([w1, w2, w3])
            tt21 = tt2 - tt1
            tt31 = tt3 - tt1
            tt32 = tt3 - tt2

            s1, c1 = np.sin(tt1), np.cos(tt1)
            s2, c2 = np.sin(tt2), np.cos(tt2)
            s3, c3 = np.sin(tt3), np.cos(tt3)
            s21, c21 = np.sin(tt21), np.cos(tt21)
            s32, c32 = np.sin(tt32), np.cos(tt32)

            J = np.array([
                [l1*c1, l2*c2, l3*c3],
                [l1*s1, l2*s2, l3*s3],
                ])

            x3ref = target.flatten()

            tc = np.dot(J.T, - Kd*v3 - Kp*(x3 - x3ref))
            tg = 9.81*np.array([s1, j12*s2, j13*s3]) # gravity comp
            tm = 2*Km*(c21*c32)*np.array([  # maneuverability "potential"
                -c32*s21,
                c32*s21 - c21*s32,
                c21*s32
                ])

            ut = np.dot(Kg, tg + tc + tm)
            return np.dot(Rt, ut)

        return np.zeros(self.d_control)

