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
sys.path.append(os.path.join(os.path.dirname(__file__), '../dmp_discrete/'))
from DMPDiscrete import *

class DMPDiscrete1D(DMPDiscrete, object):
    'Class for one-dimensional (1D) discrete DMPs.'
    
    def __init__(self, name=""):