#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Created on Mon Oct 30 19:00:00 2017

@author: gsutanto
"""

import numpy as np
from Coupling import *


class TransformCoupling(Coupling, object):
  "Class defining coupling terms for DMP transformation systems."

  def __init__(self, dmp_num_dimensions_init, name=""):
    super(TransformCoupling, self).__init__(name)
    self.dmp_num_dimensions = dmp_num_dimensions_init

  def isValid(self):
    assert (self.dmp_num_dimensions > 0), "self.dmp_num_dimensions=" + str(
        self.dmp_num_dimensions) + "<= 0 (invalid!)"
    return True

  def getValue(self):
    ct_acc = np.zeros((self.dmp_num_dimensions, 1))
    ct_vel = np.zeros((self.dmp_num_dimensions, 1))
    return ct_acc, ct_vel
