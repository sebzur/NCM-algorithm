import argparse
from mpi4py import MPI
import ncm
import os
import numpy
import rea

# ---------------------------
# Some default MPI variables
# ---------------------------
comm = MPI.COMM_WORLD
sub_comm = MPI.COMM_WORLD.Clone()
rank = comm.Get_rank()
size = comm.Get_size()

# ---------------------------
# Some default MPI variables
# ---------------------------

class Matrix(object):
    """Arguments processing and evaluation class. 

    The instance of `Process` class is a callable object
    that gets the:
    
    - filename
    - embeding dimension 
    - time lag

    and processes the datafile to calculate the C_m(r) matrix.

    """

    def __init__(self):
        self.verbose = rank == 0

    def __call__(self, signal, max_m, tau, f_min, f_max, r_steps, normalize=True, selfmatches=False):
        if tau > 1:
            print('Sorry, at this moment the implementation only allows for tau=1. This will be changed soon. Stay tuned!')
            return
        return self.get_matrix(signal, max_m, tau, f_min=f_min, f_max=f_max, r_steps=r_steps, normalize=normalize, selfmatches=selfmatches)


    def get_treshold_range(self, signal, f_min, f_max, steps):
        """ Returns the treshold ranges.

        Having the signal provided, returns the set of 
        treshold (r-filters) values.

        """
        r_std = signal.std()
        #r_std = 1.0

        r_min =  r_std  * f_min
        r_max = r_std  * f_max
        r_step = (r_max - r_min) / steps
        if self.verbose:
            print('The corsum matrix will be created for [%.2f, %.2f] tresholds range created with %d steps. SD=%.2f' % (r_min, r_max, steps, r_std))

        #normalized_log_spaced = (numpy.logspace(0, 1) - 1)/10.0
        #delta = r_max - r_min
        #log_spaced = r_min + normalized_log_spaced * delta
        #return log_spaced[::-1]

        # the r_range array is reverted
        return numpy.arange(r_min, r_max, r_step)[::-1]


    def get_matrix(self, signal, max_m, tau, f_min, f_max, r_steps, normalize, selfmatches):
        matrix = ncm.NCMatrix(signal)
        r_range = self.get_treshold_range(signal, f_min, f_max, r_steps)
        # now, crete m_range as: [1 .. max_m]
        # we increment the second `range` argument to get the max_m processed
        m_range = range(1, max_m + 1)
        if self.verbose:
            print('Preparing for correlation sums matrix calculation. The matrix dimension will be %d x %d.' % (r_steps, max_m))
        c_m = matrix.corsum_matrix(m_range, r_range, tau, normalize, selfmatches)
        if rank == 0:
            return self.merge_data(c_m, r_range)

    def merge_data(self, c_m, r_range):
        data = []
        for i, r in enumerate(r_range):
            columns = c_m[i, :].tolist()                
            data.append([r] + columns)
        return numpy.array(data)
            
matrix = Matrix()
        
