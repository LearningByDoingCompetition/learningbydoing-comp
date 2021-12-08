from .template import System
import numpy as np


class Switch_2x2(System):

    # default mask:
    # participants get to observe X1, X3, X4, X5, X6, X7, X8, X9, X11, X13
    # in that order and Y is always the first component
    # mask = (0, 1, 3, 4, 5, 6, 7, 8, 9, 11, 13)
    # mask = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)
    mask = (0, 12, 6, 3, 7, 8, 10, 2, 9, 14, 13, 4, 11, 1, 5)
    # default target
    target = 0

    # system info
    timegrid = np.exp(np.linspace(0, np.log(81), 20))-1
    d_U = 8
    d_X = 15

    def __init__(self, **kwargs):
        # default parameters
        self.parameters = {'rates': (.05, ) * 10, 'noise_sigma': 0,
                           'B': np.array([[1, 0, 0, 0, 1, 0, 0, 0],
                                          [0, 1, 0, 0, 0, 1, 0, 0],
                                          [1, 0, 0, 0, 0, 0, 1, 0],
                                          [0, 1, 0, 0, 0, 0, 0, 1],
                                          [0, 0, 1, 0, 0, 0, 1, 0],
                                          [0, 0, 0, 1, 0, 0, 0, 1],
                                          [0, 0, 1, 0, 1, 0, 0, 0],
                                          [0, 0, 0, 1, 0, 1, 0, 0]])}

        super().__init__(**kwargs)

    def rhs(self, t, x, u):
        k = self.parameters['rates']
        B = self.parameters['B']

        if u is None:
            u = [0]*8
        u = np.nan_to_num(u)

        Y = k[4]*x[10] + k[5]*x[11] - k[6]*x[9]*x[0] - k[7]*x[12]*x[0]
        X1 = -k[0]*x[1]*x[2] + k[8]*x[13] + B[0, :].dot(u)
        X2 = -k[0]*x[1]*x[2] + k[8]*x[13] + B[1, :].dot(u)
        X3 = -k[1]*x[3]*x[4] + B[2, :].dot(u)
        X4 = -k[1]*x[3]*x[4] + B[3, :].dot(u)
        X5 = -k[2]*x[5]*x[6] + k[9]*x[14] + B[4, :].dot(u)
        X6 = -k[2]*x[5]*x[6] + k[9]*x[14] + B[5, :].dot(u)
        X7 = -k[3]*x[7]*x[8] + B[6, :].dot(u)
        X8 = -k[3]*x[7]*x[8] + B[7, :].dot(u)
        X9 = k[0]*x[1]*x[2] - k[6]*x[9]*x[0]
        X10 = k[1]*x[3]*x[4] - k[4]*x[10]
        X11 = k[2]*x[5]*x[6] - k[5]*x[11]
        X12 = k[3]*x[7]*x[8] - k[7]*x[12]*x[0]
        X13 = k[6]*x[9]*x[0] - k[8]*x[13]
        X14 = k[7]*x[12]*x[0] - k[9]*x[14]

        # allow for negativ controls by never allowing process to go below 0
        res = np.array([Y, X1, X2, X3, X4, X5, X6, X7,
                       X8, X9, X10, X11, X12, X13, X14])
        res[(x <= 0) & (res < 0)] = 0

        return(res)

    def measurement_noise(self, X):
        sigma = self.parameters['noise_sigma']
        # noise level depends on variation of trajectory
        if sigma <= 0 or np.all(np.isnan(X[:, 1:])):
            return X
        sigma = sigma * (np.nanmax(X[:, 1:], axis=1) -
                         np.nanmin(X[:, 1:], axis=1)) + 0.01
        sigma = np.repeat(sigma.reshape(-1, 1), X.shape[1], axis=1)
        # initial values are always noise-free
        return np.c_[
            X[:, :1],
            X[:, 1:] + sigma[:, 1:] * np.random.randn(*X[:, 1:].shape)]

    def control_interface(self, u=None):
        if u is not None:
            np.clip(u, -10., 10., out=u)
        return u
