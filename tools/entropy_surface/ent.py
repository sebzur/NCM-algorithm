from pylab import *
from matplotlib.widgets import Slider, Button, RadioButtons
import sys
from scipy.optimize import leastsq
import os

filename = sys.argv[1]

data = loadtxt(filename)
matrix = log(data)


r = matrix[:,0]

sd = data[0,0]/5.0
ent_index = []
m = 2
for edge in [0.20, 0.15, 0.1]:
    for r_index in range(data.shape[0]):
        v = data[r_index,0]/sd
        if v < edge:
            log_cm_m = matrix[r_index, m]
            log_cm_m_1 = matrix[r_index, m + 1]
            entropy = log_cm_m - log_cm_m_1
            ent_index.append([v, entropy])
            break

#print 'Entropia', ent_index

file = open(sys.argv[2], 'a')
ff = filename.split('/')[-1]
file.write("%s\t%s\n" % (ff, '\t'.join(map(lambda x: "%.2f\t%.2f" % (x[0], x[1]), ent_index))))
file.close()



