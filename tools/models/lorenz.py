#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.integrate import odeint
import scipy
import sys



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


#tutaj zaczyna sie petla po parametrach

try:
    sigma = sys.argv.index('--sigma')+1
    ro = sys.argv.index('--ro')+1
    beta = sys.argv.index('--beta')+1
    output = sys.argv.index('--output')+1
    
except ValueError:
    print "Wrong parameters"
    

sigma = float(sys.argv[sigma])
ro = float(sys.argv[ro])
beta = float(sys.argv[beta])
output = sys.argv[output]
output = "%s_%0.3f_%0.3f_%0.3f.dat" % (output, sigma, ro, beta)


initial = [0, 1, 1.05]

t = scipy.arange(0, 100, 0.1)
lorenz_sol = odeint(lorenz_int, initial, t,({'sigma':sigma, 'ro':ro, 'beta':beta},))

x = [i[0] for i in lorenz_sol]
y = [i[1] for i in lorenz_sol]
z = [i[2] for i in lorenz_sol]

np.savetxt(output,x)
