"""
Controller template.
"""
import numpy as np
import pickle
import signal
# ~~~~~~~~~~~~
# NOTE: LBD commented out this line that imported tensorflow.
# NOTE: The original submission contained the tensorflow library.
# import tensorflow as tf
# ~~~~~~~~~~~~
import joblib
from sklearn.preprocessing import PolynomialFeatures

from contextlib import contextmanager


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
    self.d_control = d_control

    # TODO it should be ./ but for test I had to use ../

    if system.endswith('beetle'):  # if polynomial cliping
      model_save_path = f'./models_all_data/model_scikitquadratic_clipping_{system}_all_data'
      r_save_path = f'./models_all_data/model_scikitquadratic_clipping_R_{system}_all_data'

      with open(model_save_path, 'rb') as f:
        self.model = pickle.load(f)

      with open(r_save_path, 'rb') as f:
        self.Rinv = pickle.load(f)

    elif system.endswith('bumblebee'):
      self.all_controllers_params = joblib.load('linear_controllers_params.joblib')
      self.controller_params = self.all_controllers_params[self.system]
      self.n_states = self.controller_params['n_states']
      self.n_inputs = self.controller_params['n_inputs']
      self.scaler = self.controller_params['scaler']
      self.R = self.controller_params['R']
      self.A = self.controller_params['A']
      self.B = self.controller_params['B']
      self.d = self.controller_params['d']
      self.Binv = self.controller_params['Binv']

    elif system.endswith('butterfly'):
      model_save_path = f'./models_all_data/model_scikitquadratic_clipping_{system}_all_data'
      r_save_path = f'./models_all_data/model_scikitquadratic_clipping_R_{system}_all_data'

      with open(model_save_path, 'rb') as f:
        self.model = pickle.load(f)

      with open(r_save_path, 'rb') as f:
        self.Rinv = pickle.load(f)

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
    # TODO:  assumed (n,1) because of wrapcontroller.py

    # model input is (1, n_variables)
    #model_input = np.vstack([state, target]).transpose()
    if self.system.endswith('beetle'):
      model_input = np.vstack([state, target,
                               target[:2] - state[:2]]).transpose()
      # Outputs is 1,n_us
      outputs = np.clip(self.model.predict(model_input),
                        self.model.transformer_.data_min_,
                        self.model.transformer_.data_max_)

      outputs = np.matmul(outputs, self.Rinv)
      # output of the model is going to be (1,n_u) turn it into (n_u, 1)
      return outputs.transpose()

    elif self.system.endswith('bumblebee'):
      try:
        with time_limit(.9 * 16 / 200):
          state = state.flatten()
          target = target.flatten()
          transformed_u = self.Binv @ (
              target - self.A[:2] @ state - self.d[:2]
          )
          if self.system.endswith('bumblebee'):
            transformed_u = np.clip(transformed_u, -1, 1)
          else:
            transformed_u = np.clip(transformed_u, -0.7, 0.7)
          u = self.R @ self.scaler.inverse_transform([transformed_u])[0]
          return u
      except TimeoutException:
            return np.zeros(self.n_inputs)

    elif self.system.endswith('butterfly'):
      model_input = np.vstack([state, target,
                               target[:2] - state[:2]]).transpose()
      # Outputs is 1,n_us
      outputs = np.clip(self.model.predict(model_input),
                        self.model.transformer_.data_min_,
                        self.model.transformer_.data_max_)

      outputs = np.matmul(outputs, self.Rinv)
      # output of the model is going to be (1,n_u) turn it into (n_u, 1)
      return outputs.transpose()
