from mpi4py import MPI
import numpy
import importlib



# ---------------------------
# Some default MPI variables
# ---------------------------
comm = MPI.COMM_WORLD
sub_comm = MPI.COMM_WORLD.Clone()
rank = comm.Get_rank()
size = comm.Get_size()

AVALIBLE_METHODS = ["ncm_plain", "ncm_mpi", "bruteforce"]

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

    def __init__(self, method):
        if method not in AVALIBLE_METHODS:
            raise ValueError("Method for calculating correlation matrix must be one of [ncm_plain,ncm_mpi]")
        self.method = method


    def __call__(self, filename, method, max_m, tau, output, usecol, f_min, f_max, r_steps, wsize, wstep, skiprows=0, verbose=False,
                 normalize=True, selfmatches=True, precision=6):



        self.verbose = verbose
        signal = numpy.loadtxt(filename, skiprows=skiprows, usecols=(usecol,))
        if self.verbose:
            print('Signal data file is loaded. ')
        number_ordinal = 1
        for window in self.get_windows(signal, wsize or len(signal), wstep):
            self.get_matrix(window, max_m, tau, output, f_min=f_min, f_max=f_max, r_steps=r_steps,
                            normalize=normalize, selfmatches=selfmatches, number_ordinal=number_ordinal, precision=precision)
            number_ordinal += 1


    def get_treshold_range(self, signal, f_min, f_max=None, steps=None, use_std=False):
        """ Returns the treshold ranges.

        Having the signal provided, returns the set ofssdwa
        treshold (r-filters) values.

        """
        r_std = signal.std()
        # r_std = 1.0

        r_min = r_std * f_min
        if f_max is None:
            return numpy.array([f_min])
        if steps is None:
            raise ValueError("Steps has to be provided when f_max is defined")
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

    def get_windows(self, signal, w_size, w_step):
        w_start = 0
        while w_start + w_size <= len(signal):
            yield signal[w_start:w_start + w_size]
            w_start += w_step

    def get_matrix(self, signal, max_m, tau, output, f_min, f_max, r_steps, normalize, selfmatches, number_ordinal, precision):
        method_module = importlib.import_module(f"ncm.methods.{self.method}", ".")
        matrix = method_module.Matrix(signal)
        r_range = self.get_treshold_range(signal, f_min, f_max, r_steps)
        # now, crete m_range as: [1 .. max_m]
        # we increment the second `range` argument to get the max_m processed
        m_range = range(1, max_m + 1)
        if self.verbose:
            print(f'Preparing for correlation sums matrix calculation. The matrix dimension will be {r_steps} x {max_m}.')
        c_m = matrix.corsum_matrix(m_range, r_range, tau, normalize, selfmatches, precision)
        if rank == 0:
            self.store(c_m, r_range, output, number_ordinal)

    def store(self, c_m, r_range, filename,number_ordinal, as_log=False):
        #filename = f"{filename}_{str(number_ordinal).zfill(6)}.txt"
        filename = f"{filename}.txt"
        out = open(filename, 'w')
        for i, r in enumerate(r_range):
            if as_log:
                columns = numpy.log((c_m[i, :] + 0.00000000000000000001)).tolist()
            else:
                columns = c_m[i, :].tolist()
            data = [r] + columns
            out.write("\t".join(["%f" % z for z in data]) + '\n')
        out.close()
