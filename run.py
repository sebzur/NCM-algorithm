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

    def __call__(self, filename, max_m, tau, output, usecol, rmode, rdiff, wsize, wstep, verbose=False, normalize=True, selfmatches=False):
        self.verbose = verbose
        root, ext = os.path.splitext(filename)
        if ext.lower() == '.rea':
            if self.verbose:
                print('Guessing that the file is in REA format...')
            try:
                # we're skipping the first row - this is almost always
                # the row with column names
                #rea_reader = rea.REA(args.file, skiprows=1, usecols=(1,), only_valid=True)
                rea_reader = rea.REA(args.file, skiprows=6, usecols=(usecol,), only_valid=True)
                signal = rea_reader.get_signal()
                if self.verbose:
                    print('OKay, REA is loaded!')
            except Exception, error:
                if self.verbose:
                    print('There is an error: %s' % error)
        else:
            #signal = numpy.loadtxt(args.file)
            signal = numpy.loadtxt(args.file, skiprows=0, usecols=(usecol,))
            if self.verbose:
                print('Signal data file is loaded. ')

        r_range = self.get_treshold_range(signal, rmode, rdiff)
        number_ordinal = 1
        for window in self.get_windows(signal, wsize or len(signal), wstep):
            self.get_matrix(window, max_m, tau, output, r_range, normalize=normalize, selfmatches=selfmatches, number_ordinal=number_ordinal)
            number_ordinal += 1

    def get_treshold_range(self, signal, rmode, rdiff):
        """ Returns the treshold ranges.

        Having the signal provided, returns the set of 
        treshold (r-filters) values.

        """
        r_min = 0
        r_max = 1.5 * (float(max(signal)) - float(min(signal)))
        print rmode
        if rmode == 'step':
            r_step = (r_max - r_min) / rdiff
            if self.verbose:
                print('The corsum matrix will be created for [%.2f, %.2f],' % (r_min, r_max))

        else:
            r_step = rdiff
        return numpy.arange(r_min, r_max, r_step)[::-1]

    def get_windows(self, signal, w_size, w_step):
        w_start = 0
        while w_start + w_size <= len(signal):
            yield signal[w_start:w_start + w_size]
            w_start += w_step
    
    def get_matrix(self, signal, max_m, tau, output, r_range, normalize, selfmatches, number_ordinal):
        matrix = ncm.NCMatrix(signal, 0, signal.size)
        # now, crete m_range as: [1 .. max_m]
        # we increment the second `range` argument to get the max_m processed
        m_range = range(1, max_m + 1)
        if self.verbose:
            print('Preparing for correlation sums matrix calculation.')
        c_m = matrix.corsum_matrix(m_range, r_range, tau, normalize, selfmatches)
        if rank == 0:
            self.store(c_m, r_range, output, number_ordinal)


    def store(self, c_m, r_range, filename, number_ordinal, as_log=False):
        filename = filename + str(number_ordinal).zfill(6)
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
    parser.add_argument('usecol', type=int, help='Column with data')
    parser.add_argument('--normalize', dest='normalize', action='store_true', help='Should the output be normalized?')
    parser.add_argument('--selfmatches', dest='selfmatches', action='store_true', help='Correlate selfmatches?')
    parser.add_argument('--rmode', type=str, default='steps', help='(step|diff)')
    parser.add_argument('--rdiff', type=float, default=100, help='Number of treshold values')
    parser.add_argument('--wsize', type=int, default=None, help='Window size')
    parser.add_argument('--wstep', type=int, default=1, help='Window step')
    args = parser.parse_args()
    # we print out things only on rank 0 (aka master node)
    verbose = rank == 0
    if args.tau > 1:
        if verbose:
            print('Sorry, at this moment the implementation only allows for tau=1. This will be changed soon. Stay tuned!')
    else:
        Process()(args.file, args.m, args.tau, output=args.output, usecol=args.usecol, verbose=verbose, normalize=args.normalize,
                  selfmatches=args.selfmatches, rmode=args.rmode, rdiff=args.rdiff, wsize=args.wsize, wstep=args.wstep)




    



