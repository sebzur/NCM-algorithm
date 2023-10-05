import numpy
import sys

from mpi4py import MPI
comm = MPI.COMM_WORLD
sub_comm = MPI.COMM_WORLD.Clone()
rank = comm.Get_rank()
size = comm.Get_size()


class Matrix:

    def __init__(self,signal, start, stop):
        self.signal = signal

    def corsum_matrix(self, m_range, r_range, tau, normalize, selfmatches):
        """ Caclulates correlaction sum for a signal.

        For the provided embeding space, tau and treshold
        calculates correlation sum for a signal entered and
        returns the number of counts

        """
        signal = self.signal

        total = numpy.zeros(len(r_range))
        templates_count = len(signal) - (m -1) * tau
        ext_factor = 1.0 / templates_count
        int_factor = 1.0 / (templates_count - 1) # -1 due to self-matches exclusion
        for i in range(rank, templates_count, size):
            template = numpy.array(signal[i:i + m:tau])
            for j in range(len(signal) - (m - 1) * tau):
                if i == j:
                    continue
                checked = numpy.array(signal[j:j+m:tau])
                total += (numpy.abs((template - checked)).max() <= r_range)

        results = comm.gather(total, root=0)
        if rank == 0:
            summed = numpy.array(results).sum(axis=0)
            return summed * ext_factor * int_factor