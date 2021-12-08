"""
Controller template.
"""

from contextlib import contextmanager
import joblib
import onnxruntime as rt
import numpy as np
from pathlib import Path
import signal


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
        """load scaler"""
        self.scaler = joblib.load(Path(__file__).parent / f"{self.system}_scaler.joblib")

        """load in onnx model"""
        self.all_sess = []
        for i in range(10):
            sess = rt.InferenceSession(f"{system}_{i}.onnx")
            self.input_name = sess.get_inputs()[0].name
            self.label_name = sess.get_outputs()[0].name
            self.all_sess.append(sess)
        self.d_control = d_control

        # Initialize empty matrices used in PID controller
        self.position_mat = np.zeros((200, 2))
        self.target_mat = np.zeros((200, 2))
        self.prev_fea = None
        # Initialize variable that contains current timestep
        self.step = 0

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
        if self.step == 0:
            # Initial step is set to zero
            self.position_mat[0, :] = position.flatten()
            self.target_mat[0, :] = target.flatten()
            self.step += 1
            return np.zeros(self.d_control)
        else:
            # Compute P/I/D-terms and apply fitted PID controller
            self.position_mat[self.step, :] = position.flatten()
            self.target_mat[self.step, :] = target.flatten()

            PP = (self.target_mat[self.step, :]
                  - self.position_mat[self.step, :])
            II = np.sum((self.target_mat[:(self.step+1), :] -
                         self.position_mat[:(self.step+1), :]),
                        axis=0)/200
            DD = (PP - (self.target_mat[self.step-1, :]
                        - self.position_mat[self.step-1, :]))*200

            features = np.r_[PP, II, DD, state.flatten(), target.flatten()].reshape(1, -1).astype(np.float32)
            if self.step == 1:
                self.prev_fea = np.zeros_like(features)
            if "butterfly" in self.system:
                comb_fea = features
            else:
                comb_fea = np.hstack([features, self.prev_fea])

            """onnx model"""
            uopt = np.zeros(self.d_control)
            for sess in self.all_sess:
                pred = sess.run([self.label_name], {self.input_name: comb_fea})[0]
                uopt += self.scaler.inverse_transform(pred).flatten()
            uopt /= 10
            self.step += 1
            self.prev_fea = features
            return uopt
