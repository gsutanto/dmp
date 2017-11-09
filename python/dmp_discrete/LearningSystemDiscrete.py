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
from LearningSystem import *
from TransformSystemDiscrete import *
from FuncApproximatorDiscrete import *

class LearningSystemDiscrete(LearningSystem, object):
    'Class for learning systems of discrete DMPs.'
    
    def __init__(self, transformation_system_discrete, name=""):
        super(LearningSystemDiscrete, self).__init__(transformation_system_discrete, name)
    
    def isValid(self):
        assert (super(LearningSystemDiscrete, self).isValid())
        return True
    
    def learnApproximator(self, set_dmptrajectory_demo_local, robot_task_servo_rate):
        assert (self.isValid())
        assert (robot_task_servo_rate > 0.0)
        
        
        return None