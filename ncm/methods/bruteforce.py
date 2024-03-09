import numpy

from mpi4py import MPI
comm = MPI.COMM_WORLD
sub_comm = MPI.COMM_WORLD.Clone()
rank = comm.Get_rank()
size = comm.Get_size()

NumberTypes = (int, float)


class Matrix:
    def __init__(self,signal):
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

    def corsum_matrix(self, m, r_range, tau=1, precision=6):
        results = []
        m = m[0]
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
            result = Corr_sum * (1 / Lm) * (1 / (Lm - 1))
            results.append(result)
        return numpy.round(results, precision)

