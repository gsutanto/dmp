#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Created on Tue Apr  4 16:09:06 2017

@author: gsutanto
"""

import numpy as np
import tensorflow as tf


class NeuralNetwork:
  'Common base class for a variety of neural network topologies.'

  def __init__(self, name):
    self.name = name
