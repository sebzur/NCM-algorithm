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


class Process(object):
    """Arguments processing and evaluation class. 

    The instance of `Process` class is a callable object
    that gets the:
    
    - filename
    - embeding dimension 
    - time lag

    and processes the datafile to calculate the C_m(r) matrix.

    """

    def __call__(self, filename, max_m, tau, output, r_min, r_max, r_steps, verbose=False, normalize=True, selfmatches=False, **kwargs):
        self.verbose = verbose
        signal = self.load_rea(filename)
        if self.verbose:
            print('Signal data file %s has been loaded...' % filename)
        self.get_matrix(signal, max_m, tau, output, r_min=r_min, r_max=r_max, r_steps=r_steps, normalize=normalize, selfmatches=selfmatches)

    def load_rea(self, filename):
        return rea.REA(args.file).get_signal()

    def load_txt(self, filename):
        return numpy.loadtxt(args.file, skiprows=0, usecols=(0,))

    def get_treshold_range(self, signal, r_min=0, r_max=None, steps=1000):
        """ Returns the treshold ranges.
        Having the signal provided, returns the set of
        treshold (r-filters) values.
        """
        if r_max is None:
            r_max = signal.max() - signal.min()
        r_step = (r_max - r_min) / (steps + 0.0)
        if self.verbose:
            print('The corsum matrix will be created for [%.2f, %.2f] tresholds range created with %d steps.' % (r_min, r_max, steps))
        # the r_range array is reverted
        return numpy.arange(r_min, r_max + r_step, r_step)[::-1]


    def get_matrix(self, signal, max_m, tau, output, r_min, r_max, r_steps, normalize, selfmatches):
        matrix = ncm.NCMatrix(signal, 0, signal.size)
        r_range = self.get_treshold_range(signal, r_min, r_max, r_steps)
        # now, crete m_range as: [1 .. max_m]
        # we increment the second `range` argument to get the max_m processed
        m_range = range(1, max_m + 1)
        if self.verbose:
            print('Preparing for correlation sums matrix calculation. The matrix dimension will be %d x %d.' % (r_steps, max_m))
        c_m = matrix.corsum_matrix(m_range, r_range, tau, normalize, selfmatches)
        if rank == 0:
            self.store(c_m, r_range, output)


    def store(self, c_m, r_range, filename):
        out = open(filename, 'w')
        for i, r in enumerate(r_range):
            columns = c_m[i, :].tolist()                
            data = [r] + columns
            out.write("\t".join(["%f" % z for z in data]) + '\n')
        out.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NCM algorithm for correlation sums')
    parser.add_argument('file', type=str, help='Signal data - single column file or REA format')
    parser.add_argument('m', type=int, default=10, help='Embeding dimension (default: 10)')
    parser.add_argument('tau', type=int, default=1, help='Time lag (default: 1)')
    parser.add_argument('output', type=str, help='Output file name')
    parser.add_argument('--normalize', dest='normalize', action='store_true', help='Should the output be normalized?')
    parser.add_argument('--selfmatches', dest='selfmatches', action='store_true', help='Correlate selfmatches?')
    parser.add_argument('--rsteps', type=int, default=100, help='Number of treshold values')
    parser.add_argument('--rmin', type=float, default=0, help='Starting value for the tresholds range (default: 0)')
    parser.add_argument('--rmax', type=float, default=None, help='Upper value for tresholds range (default: sig.max - sig.min)')
    args = parser.parse_args()
    # we print out things only on rank 0 (aka master node)
    verbose = rank == 0
    if args.tau > 1:
        if verbose:
            print('Sorry, at this moment the implementation only allows for tau=1. This will be changed soon. Stay tuned!')
    else:
        Process()(args.file, args.m, args.tau, output=args.output, verbose=verbose, normalize=args.normalize,
                  selfmatches=args.selfmatches, r_min=args.rmin, r_max=args.rmax, r_steps=args.rsteps)




    



