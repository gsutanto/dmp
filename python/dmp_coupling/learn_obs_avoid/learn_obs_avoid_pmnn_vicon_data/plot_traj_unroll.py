#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:00:00 2017

@author: gsutanto
"""

import numpy as np
import os
import sys
import copy
import glob
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../utilities/'))
from utilities import *


global_traj_unroll_before_iter_learn_unroll = loadObj('global_traj_unroll_before_iterative_learn_unroll.pkl')
global_traj_unroll_after_iter_learn_unroll = loadObj('global_traj_unroll.pkl')