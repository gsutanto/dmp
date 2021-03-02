#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Created on Sun Jun 16 23:00:00 2019

@author: gsutanto
"""

import numpy as np
import numpy.linalg as npla
import os
import sys
import copy
import matplotlib.pyplot as plt
sys.path.append(
    os.path.join(os.path.dirname(__file__), "../../reinforcement_learning/"))
from pi2 import *


def pi2_test():
  # simple target/goal-tracking task using PI2 reinforcement learning
  D = 2
  goal = np.ones(D)
  start = np.zeros(D)
  init_std = 0.3
  N_iter = 1000
  N_sample = 15
  N_timestep = 200

  pi2_opt = Pi2(
      kl_threshold=1.0,
      covariance_damping=2.0,
      is_computing_eta_per_timestep=True)

  # initialization:
  position_mean = start
  position_cov = np.diag(np.ones(D) * (init_std * init_std))

  timestep_multipliers = np.power(0.5 * np.ones(N_timestep), range(N_timestep))

  for it in range(0, N_iter):
    position_mean_cost = npla.norm(goal - position_mean, ord=2)
    print("iter %d: position_mean = " % it + str(position_mean) +
          ", cost = %f" % position_mean_cost)

    position_samples = np.random.multivariate_normal(position_mean,
                                                     position_cov, N_sample)
    position_costs_max = npla.norm(
        goal[np.newaxis, :] - position_samples, axis=1, ord=2)
    position_costs = position_costs_max[:, np.newaxis] * timestep_multipliers[
        np.newaxis, :]
    [position_mean, position_cov, _,
     _] = pi2_opt.update(position_samples, position_costs, position_mean,
                         position_cov)

  return None


if __name__ == "__main__":
  pi2_test()
