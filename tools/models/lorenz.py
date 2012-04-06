#!/usr/bin/python
# -*- coding: utf-8 -*-

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import scipy



def lorenz_int(initial, t, args):
    x = initial[0]
    y = initial[1]
    z = initial[2]
    
    sigma = args['sigma']
    ro = args['ro']
    beta = args['beta']
  
    x_dot = sigma * (y - x)
    y_dot = x * (ro -z) - y
    z_dot = x * y - beta* z
    return [x_dot, y_dot, z_dot]

initial = [0, 1, 1.05]

t = scipy.arange(0, 100, 0.1)
lorenz_sol = odeint(lorenz_int, initial, t,({'sigma':0.2, 'ro':14, 'beta':8.0/3},))

x = [i[0] for i in lorenz_sol]
y = [i[1] for i in lorenz_sol]
z = [i[2] for i in lorenz_sol]

np.savetxt("lor5.dat",x)
