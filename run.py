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

    def __call__(self, filename, max_m, tau, output, f_min=0.001, f_max=5, r_steps=100, verbose=False, normalize=True):
        self.verbose = verbose
        root, ext = os.path.splitext(filename)
        if ext.lower() == '.rea':
            if self.verbose:
                print('Guessing that the file is in REA format...')
            try:
                rea_reader = rea.REA(args.file, skiprows=0)
                signal = rea_reader.get_signal()
                if self.verbose:
                    print('OKay, REA is loaded!')
            except Exception, error:
                if self.verbose:
                    print('There is an error: %s' % error)
        else:
            signal = numpy.loadtxt(args.file)
            if self.verbose:
                print('Signal data file is loaded. ')

        print signal[:10]
        self.get_matrix(signal, max_m, tau, output, f_min=0.001, f_max=5, r_steps=100, normalize=normalize)

    def get_treshold_range(self, signal, f_min, f_max, steps):
        """ Returns the treshold ranges.

        Having the signal provided, returns the set of 
        treshold (r-filters) values.

        """
        r_std = signal.std()

        r_min =  r_std  * f_min
        r_max = r_std  * f_max
        r_step = (r_max - r_min) / steps
        if self.verbose:
            print('The corsum matrix will be created for [%.2f, %.2f] tresholds range created with %d steps.' % (r_min, r_max, steps))
        # the r_range array is reverted
        return numpy.arange(r_min, r_max, r_step)[::-1]
        

    def get_matrix(self, signal, max_m, tau, output, f_min, f_max, r_steps, normalize):
        matrix = ncm.NCMatrix(signal, 0, signal.size)
        r_range = self.get_treshold_range(signal, f_min, f_max, r_steps)
        # now, crete m_range as: [1 .. max_m]
        # we increment the second `range` argument to get the max_m processed
        m_range = range(1, max_m + 1)
        if self.verbose:
            print('Preparing for correlation sums matrix calculation. The matrix dimension will be %d x %d.' % (r_steps, max_m))
        c_m = matrix.corsum_matrix(m_range, r_range, tau, normalize)
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
    parser.add_argument('m', type=int, default=10, help='Embeding dimension (default: 10)')
    parser.add_argument('tau', type=int, default=1, help='Time lag (default: 1)')
    parser.add_argument('output', type=str, help='Output file name')
    parser.add_argument('--normalize', dest='normalize', action='store_true', help='Should the output be normalized? (default: true)')
    args = parser.parse_args()
    # we print out things only on rank 0 (aka master node)
    verbose = rank == 0
    if args.tau > 1:
        if verbose:
            print('Sorry, at this moment the implementation only allows for tau=1. This will be changed soon. Stay tuned!')
    else:
        Process()(args.file, args.m, args.tau, output=args.output, verbose=verbose, normalize=args.normalize)




    



