# -*- coding: utf-8
# 17 Feb. 2011, 22:52:27 CET
# Leiden, The Netherlands

import numpy
import itertools

from mpi4py import MPI
comm = MPI.COMM_WORLD
sub_comm = MPI.COMM_WORLD.Clone()
rank = comm.Get_rank()
size = comm.Get_size()
NumberTypes = (int, float)

class Matrix:
    """ Norm Component Matrix """

    def __init__(self, series, start, stop):
        self.series = series
        self.max_diff = self.series.max() - self.series.min()
        self.N = len(self.series)
        self._rows = {}

    def windups(self, m_counts, row, tau):
        """ A generator that iterates over m_counts successive rows,
        starting from the now noumber `row` in the base norm-component
        matrix.

        """
        current_row = self.get_row(row, tau)
        for m in range(m_counts):
            if row + m * tau == self.N - 1:
                break
            v = self.get_row(row + m * tau, tau)
            current_row = numpy.vstack((current_row[:v.size], v)).max(axis=0)
            yield current_row

    def get_row(self, row, tau):
        """ Builds up the matrix. """
        # The matrix is generated with the successive rows defined below, where
        # each row is the difference of the [i+1:] series subset with the i-th series element
        # *BUT* the i goes from [0, N -1], the last point is excluded

        # W should check:
        # if row < self.N:
        # and if not, than: 
        # raise ValueError("The row index must by in [0, N-1]")
        # but it gives another check, and we're greedy on resources
        try:
            return self._rows[row]
        except KeyError:
            self._rows[row] = self._calc_row(row, tau)
            return self._rows[row]

    def _calc_row(self, row, tau):
        """ Returns NCM row.

        Arguments:
        - row: the row index to calculate
        - tau: time lag value to use
        
        The row is calculated with the matrix definition given by:

        $n_{i,j} = |\vec v_{0,\tau}(i) - \vec v_{0,\tau}(i + 1 + j \cdot \tau)|$

        so we do n_{i,:} here, using numpy vectorized way to process the whole
        array in one readable code line.
        """
        return numpy.abs(self.series[row] - self.series[row + tau::tau])

    def generate_rows(self, tau):
        """ Yields all rows in the NCM matrix. """
        # The matrix is generated with the successive rows defined below, where
        # each row is the difference of the [i+1:] series subset with the i-th series element
        # *BUT* the i goes from [0, N -1], the last point is excluded
        #for i in xrange(0, self.N - tau, tau):
        for i in range(0, self.N - tau, 1):
            yield self.get_row(i, tau)

    def _r_range_filter(self, row, r_range, a, b):
        # Iterates over the al row elements and check the
        # number of the existance od the elements lower than
        # the filter value applied.
        counts = []
        for r in r_range: # r_range have to be sorted!
            row = row.compress(row <= r)
            counts.append(len(row)) 
        return numpy.array(counts)


    def r_range_filter(self, row, r_range_size, a, b):
        # Iterates over the al row elements and check the
        # number of the existance od the elements lower than
        # the filter value applied.
        # --------
        v = ((row - b)/a).astype('int')
        if numpy.any(v<0):
            raise ValueError
        v.sort()
        z = numpy.zeros(r_range_size)
        for k, v in itertools.groupby(v):
            z[:k+1] += len(list(v))
        return z

    def _multi(self, matrix, tresholds):
        # depraceted, do not use
        for row in range(matrix.shape[0]):
            for k, element in enumerate(matrix[row,:]):
                matrix[row,:k] += element

    def corsum_matrix(self, m_range, r_range, tau, normalize=True, selfmatches=False):

        m_counts = max(m_range)

        if isinstance(r_range, NumberTypes):
            r_range = [r_range]
            corsum_matrix = numpy.zeros((m_counts, 1))
            a = -1
            b = r_range[0]
            r_range_size = 1
            r_range_max = r_range[0]
        elif isinstance(r_range, numpy.ndarray):
            corsum_matrix = numpy.zeros((m_counts, len(r_range)))
            a = -(r_range[0] - r_range[-1]) / (r_range.size - 1)
            b = r_range[0]
            r_range_size = r_range.size
            r_range_max = r_range.max()
        else:
            raise ValueError("r_range must be number or numpy array of numbers")


        if r_range_max < self.max_diff:
            raise ValueError("R range maximum has to be greater then the signal max diff")

        # when parallel aprochach is in the case:
        # We can scan thourgh the matrix in parallel

        for row_n in range(rank, self.N - 1, size):
            for m_index, row in enumerate(self.windups(m_counts, row_n, tau)):
                corsum_matrix[m_index] += self.r_range_filter(row, r_range_size, a, b)
            for key in [k for k in self._rows.keys() if k < row_n]:
                self._rows.pop(key)
#        self.multi(corsum_matrix, r_range)


        results = comm.gather(corsum_matrix, root=0)
        if rank == 0:

            summed = corsum_matrix = numpy.zeros((m_counts, len(r_range)))
            for r in results:
                # we scale up the results by a factor 2 due to
                # triangularity of the NCM matrix 
                summed += r * 2

            if selfmatches:
                for row in range(m_counts):
                    m = row + 1
                    offset = (self.N - (m - 1) * tau)
                    summed[row] += offset
                
            if normalize:
                for row in range(m_counts):
                    m = row + 1
                    factorA = (self.N - (m - 1) * tau)
                    if selfmatches:
                        factorB = (self.N  - (m - 1) * tau)
                    else:
                        factorB = (self.N - 1 - (m - 1) * tau)
                    factor = (factorA * factorB)
                    summed[row] /= factor

            return summed.transpose()

        else:
            return None



