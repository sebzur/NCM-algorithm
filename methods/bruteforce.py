import numpy
import sys

from mpi4py import MPI
comm = MPI.COMM_WORLD
sub_comm = MPI.COMM_WORLD.Clone()
rank = comm.Get_rank()
size = comm.Get_size()

NumberTypes = (int, float)

class Matrix:

    def __init__(self,signal, start, stop):
        self.signal = signal

    def corsum_matrix(self, m, r_range, tau, normalize=True, selfmatches=True):
        """ Caclulates correlaction sum for a signal.

        For the provided embeding space, tau and treshold
        calculates correlation sum for a signal entered and
        returns the number of counts

        """
        signal = self.signal
        if not isinstance(r_range,numpy.ndarray):
            r_range = [r_range]

        m = m[0] +1

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

class Matrix:
    def __init__(self,signal, start, stop):
        self.__class__.__name__ = "Brutefoce_correlation_sums"
        self.signal = signal

    @staticmethod
    def heaviside(x):
        if x >= 0:
            return 1
        else:
            return 0

    @staticmethod
    def max_coordinate_diff(v1, v2):
        return max(abs(v1-v2))


    def corsum_matrix(self, m, r_range, tau=1):
        results = []
        m = m[0] + 1
        Lm = len(self.signal) - (m - 1) * tau
        Corr_sum = 0
        for r in r_range:
            for i in range(Lm):
                v_i = numpy.array(self.signal[i:i + m:tau])
                for j in range(Lm):
                    if i == j:
                        continue
                    v_j = numpy.array(self.signal[j:j+m:tau])
                    Corr_sum += self.heaviside(r-self.max_coordinate_diff(v_i, v_j))
            sum_norm = Corr_sum * (1/Lm) * (1/(Lm-1))
            results.append(sum_norm)
        return results

    def corsum_matrix(self, m, r_range, tau=1 ):
        """ Caclulates correlaction sum for a signal.
        For the provided embeding space, tau and treshold
        calculates correlation sum for a signal entered and
        returns the number of counts
        """
        m = m[0]
        signal = self.signal
        total = 0
        templates_count = len(signal) - (m -1) * tau
        ext_factor = 1.0 / templates_count
        int_factor = 1.0 / (templates_count - 1)  # -1 due to self-matches exclusion
        for i in range(templates_count):
            template = numpy.array(signal[i:i + m:tau])
            for j in range(len(signal) - (m - 1) * tau):
                if i == j:
                    continue
                checked = numpy.array(signal[j:j+m:tau])
                total += (numpy.abs((template - checked)).max() <= r_range)
        results = total
        summed = numpy.array(results).sum(axis=0)
        return summed * ext_factor * int_factor

    def corsum_matrix(self, m, r_range, tau = 1): # bf_correlation
        """ Caclulates correlaction sum for a signal.
        For the provided embeding space, tau and treshold
        calculates correlation sum for a signal entered and
        returns the number of counts
        """
        res = []
        m = m[0]
        signal = self.signal
        total = 0
        templates_count = len(signal) - (m -1) * tau
        ext_factor = 1.0 / templates_count
        int_factor = 1.0 / (templates_count - 1) # -1 due to self-matches exclusion

        r_range = [r_range] if isinstance(r_range, NumberTypes) else r_range
        for r in r_range:
            for i in range(templates_count):
                template = numpy.array(signal[i:i + m:tau])
                for j in range(len(signal) - (m - 1) * tau):
                    if i == j:
                        continue
                    checked = numpy.array(signal[j:j+m:tau])
                    total += (numpy.abs((template - checked)).max() <= r)
            results = total
            summed = numpy.array(results).sum(axis=0)
            summed * ext_factor * int_factor
            res.append(summed)
        return res



