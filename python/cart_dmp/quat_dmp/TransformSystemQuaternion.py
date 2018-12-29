#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 21:00:00 2018

@author: gsutanto
"""

import numpy as np
import numpy.linalg as npla
import numpy.matlib as npma
import os
import sys
import copy
sys.path.append(os.path.join(os.path.dirname(__file__), '../../dmp_base/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../dmp_discrete/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../dmp_goal_system/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../dmp_param/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../dmp_state/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../utilities/'))
from TauSystem import *
from TransformSystemDiscrete import *
from FuncApproximatorDiscrete import *
from CanonicalSystemDiscrete import *
from QuaternionGoalSystem import *
from QuaternionDMPState import *
import utility_quaternion as util_quat

