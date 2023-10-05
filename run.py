import argparse
from mpi4py import MPI
import methods
import os
import numpy
import importlib

# ---------------------------
# Some default MPI variables
# ---------------------------
comm = MPI.COMM_WORLD
sub_comm = MPI.COMM_WORLD.Clone()
rank = comm.Get_rank()
size = comm.Get_size()

AVALIBLE_METHODS = ["ncm_basic", "ncm_mpi"]

class Process:
    """Arguments processing and evaluation class.

    The instance of `Process` class is a callable object
    that gets the:

    - filename
    - method how to calculate correlation sum matrix
    - embeding dimension
    - time lag

    and processes the datafile to calculate the C_m(r) matrix.

    """

    def __call__(self, filename, method, max_m, tau, output, usecol, f_min, f_max, r_steps, skiprows=0, verbose=False,
                 normalize=True, selfmatches=True):

        if method not in AVALIBLE_METHODS:
            raise ValueError("Method for calculating correlation matrix must be one of [npc_basic,npc_mpi]")

        self.verbose = verbose
        signal = numpy.loadtxt(args.file, skiprows=skiprows, usecols=(usecol,))
        if self.verbose:
            print('Signal data file is loaded. ')

        self.get_matrix(method, signal, max_m, tau, output, f_min=f_min, f_max=f_max, r_steps=r_steps,
                        normalize=normalize, selfmatches=selfmatches)

    def get_treshold_range(self, signal, f_min, f_max, steps):
        """ Returns the treshold ranges.

        Having the signal provided, returns the set of
        treshold (r-filters) values.

        """
        r_std = signal.std()
        # r_std = 1.0

        r_min = r_std * f_min
        r_max = r_std * f_max
        r_step = (r_max - r_min) / steps
        if self.verbose:
            print(
                'The corsum matrix will be created for [%.2f, %.2f] tresholds range created with %d steps. SD=%.2f' % (
                r_min, r_max, steps, r_std))

        # normalized_log_spaced = (numpy.logspace(0, 1) - 1)/10.0
        # delta = r_max - r_min
        # log_spaced = r_min + normalized_log_spaced * delta
        # return log_spaced[::-1]

        # the r_range array is reverted
        return numpy.arange(r_min, r_max, r_step)[::-1]

    def get_matrix(self, method, signal, max_m, tau, output, f_min, f_max, r_steps, normalize, selfmatches):
        method_module = importlib.import_module("methods."+method, ".")
        matrix = method_module.Matrix(signal, 0, signal.size)
        r_range = self.get_treshold_range(signal, f_min, f_max, r_steps)
        # now, crete m_range as: [1 .. max_m]
        # we increment the second `range` argument to get the max_m processed
        m_range = range(1, max_m + 1)
        if self.verbose:
            print('Preparing for correlation sums matrix calculation. The matrix dimension will be %d x %d.' % (
            r_steps, max_m))
        c_m = matrix.corsum_matrix(m_range, r_range, tau, normalize, selfmatches)
        if rank == 0:
            self.store(c_m, r_range, output)

    def store(self, c_m, r_range, filename, as_log=False):
        out = open(filename, 'w')
        for i, r in enumerate(r_range):
            if as_log:
                columns = numpy.log((c_m[i, :] + 0.00000000000000000001)).tolist()
            else:
                columns = c_m[i, :].tolist()
            data = [r] + columns
            out.write("\t".join(["%f" % z for z in data]) + '\n')
        out.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NCM algorithm for correlation sums')
    parser.add_argument('file', type=str, help='Signal data - single column file or REA format')
    parser.add_argument('method', type=str,
                        help='Method for calculating correlation sums (classic, NCM, NCM-MPI enhanced)')
    parser.add_argument('m', type=int, default=10, help='Embeding dimension (default: 10)')
    parser.add_argument('tau', type=int, default=1, help='Time lag (default: 1)')
    parser.add_argument('output', type=str, help='Output file name')
    parser.add_argument('usecol', type=int, help='Column with data')
    parser.add_argument('--normalize', dest='normalize', action='store_true', help='Should the output be normalized?')
    parser.add_argument('--selfmatches', dest='selfmatches', action='store_true', help='Correlate selfmatches?')
    parser.add_argument('--skiprows', type=float, default=0, dest='skiprows', help='Skip first n rows from input file')
    parser.add_argument('--rsteps', type=int, default=100, help='Number of treshold values')
    parser.add_argument('--fmin', type=float, default=0,
                        help='SD * f_min - lower value for tresholds range (default: 0.001)')
    parser.add_argument('--fmax', type=float, default=5.0,
                        help='SD * f_max - upper value for tresholds range (default: 5.0)')
    args = parser.parse_args()
    # we print out things only on rank 0 (aka master node)
    verbose = rank == 0
    if args.tau > 1:
        if verbose:
            print(
                'Sorry, at this moment the implementation only allows for tau=1. This will be changed soon. Stay tuned!')
    else:
        Process()(args.file, args.method, args.m, args.tau, output=args.output, usecol=args.usecol, verbose=verbose,
                  normalize=args.normalize, selfmatches=args.selfmatches, skiprows=args.skiprows,
                  f_min=args.fmin, f_max=args.fmax, r_steps=args.rsteps)








