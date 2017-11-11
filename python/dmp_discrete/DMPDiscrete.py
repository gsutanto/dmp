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
sys.path.append(os.path.join(os.path.dirname(__file__), '../dmp_param/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../dmp_base/'))
from DMP import *
from TransformSystemDiscrete import *
from FuncApproximatorDiscrete import *
from CanonicalSystemDiscrete import *
from LearningSystemDiscrete import *

class DMPDiscrete(DMP, object):
    'Class for discrete DMPs.'
    
    def __init__(self, name=""):
        