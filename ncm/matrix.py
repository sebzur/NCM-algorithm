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

    def __call__(self, signal, max_m, tau, rmode, rdiff, normalize=True, selfmatches=False):
        if tau > 1:
            print('Sorry, at this moment the implementation only allows for tau=1. This will be changed soon. Stay tuned!')
            return
        if rmode != 'steps' and  rmode != 'diff':
            print('Sorry, You must choose the rmode: "diff" or "steps"!')
            return

        return self.get_matrix(signal, max_m, tau, rmode=rmode, rdiff=rdiff, normalize=normalize, selfmatches=selfmatches)


    def get_treshold_range(self, signal, rmode, rdiff):
        """ Returns the treshold ranges.

        Having the signal provided, returns the set of 
        treshold (r-filters) values.

        """
        r_min = 0
        r_max = 1.3 * (float(max(signal))-float(min(signal)))
        if rmode == 'steps':
            r_step = (r_max-r_min) / rdiff
        else:
            r_step = rdiff
        return numpy.arange(r_min, r_max, r_step)[::-1]

    def get_matrix(self, signal, max_m, tau, rmode, rdiff, normalize, selfmatches):
        matrix = ncm.NCMatrix(signal)
        r_range = self.get_treshold_range(signal, rmode, rdiff)
        # now, crete m_range as: [1 .. max_m]
        # we increment the second `range` argument to get the max_m processed
        m_range = range(1, max_m + 1)
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
        
