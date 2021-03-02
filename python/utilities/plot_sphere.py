#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Created on Mon Oct 30 19:00:00 2017

@author: gsutanto
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_sphere(ax, R, Ox, Oy, Oz, color='r'):
  u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
  x = Ox + (R * np.cos(u) * np.sin(v))
  y = Oy + (R * np.sin(u) * np.sin(v))
  z = Oz + (R * np.cos(v))
  ax.plot_surface(x, y, z, color=color)
  return None


if __name__ == '__main__':
  fig = plt.figure()
  ax = fig.gca(projection='3d')
  ax.set_aspect('equal')
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.set_zlabel('z')
  plot_sphere(ax, 5, 1, 2, 3, 'b')
