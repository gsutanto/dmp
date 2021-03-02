#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Created on Mon Oct 30 19:00:00 2017

@author: gsutanto
"""

import numpy as np
import os
import sys
import copy
sys.path.append(os.path.join(os.path.dirname(__file__), '../dmp_param/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../dmp_base/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../dmp_discrete/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../utilities/'))
from DMPDiscrete import *
from TransformSystemDiscrete import *
from utilities import *


class DMPDiscrete1D(DMPDiscrete, object):
  'Class for one-dimensional (1D) discrete DMPs.'

  def __init__(self,
               model_size_init,
               canonical_system_discrete,
               transform_couplers_list=[],
               name=''):
    self.transform_sys_discrete_1D = TransformSystemDiscrete(
        1, canonical_system_discrete, None, [True], 25.0, 25.0 / 4.0, None,
        None, None, None, transform_couplers_list)
    super(DMPDiscrete1D,
          self).__init__(1, model_size_init, canonical_system_discrete,
                         self.transform_sys_discrete_1D, name)
    print('DMPDiscrete1D is created.')

  def isValid(self):
    assert (self.transform_sys_discrete_1D.isValid())
    assert (super(DMPDiscrete1D, self).isValid())
    assert (self.transform_sys_discrete == self.transform_sys_discrete_1D)
    assert (self.dmp_num_dimensions == 1)
    return True

  def extractSetTrajectories(self,
                             training_data_dir_or_file_path,
                             start_column_idx=1,
                             time_column_idx=0):
    return extractSetNDTrajectories(training_data_dir_or_file_path, 1,
                                    start_column_idx, time_column_idx)
