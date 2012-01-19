# -*- coding: utf-8 -*-
# 17 Feb. 2011, 22:52:27 CET
# Leiden, The Netherlands

import numpy
import sys
import os
import time
#import matplotlib.pyplot as plt
import rea
import itertools
#from pylab. import import 

def logistic_map(x, A, length):
    wynik=[]
    for i in range(length):
        x=A*x*(1-x)
        wynik.append(x)
    return wynik[200:]


from mpi4py import MPI
comm = MPI.COMM_WORLD
sub_comm = MPI.COMM_WORLD.Clone()
rank = comm.Get_rank()
size = comm.Get_size()


def max_norm(iterable):
    return max(iterable)

class FNCMatrix(object):
    """ Norm Component Matrix """

    def __init__(self, series, start, stop):
        self.series = series
        self.N = len(self.series)
        self.start = start
        self.stop = stop
        self._rows = {}

    def windups(self, m_counts, row, tau, norm=max_norm):
        """ A generator that iterates over m_counts successive rows,
        starting from the now noumber `row` in the base norm-component
        matrix.

        """
        current_row = self.get_row(row, tau)
        for m in xrange(m_counts):
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
        for i in xrange(0, self.N - tau, 1):
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

    def r_range_filter(self, row, r_range, a, b):
        # Iterates over the al row elements and check the
        # number of the existance od the elements lower than
        # the filter value applied.
        v = ((row - b)/a).astype('int')
        v.sort()
        z = numpy.zeros(r_range.size)
        for k, v in itertools.groupby(v):
            #z[:k] += len(list(v))
            try:
                z[k-1] += len(list(v))
            except IndexError:
                indeks = 0 if k < 0 else r_range.size - 1
                z[indeks] += len(list(v))
        return z

    def multi(self, matrix, tresholds):
        for row in range(matrix.shape[0]):
            for k, element in enumerate(matrix[row,:]):
                matrix[row,:k] += element

    def corsum_matrix(self, m_range, r_range, tau):
        #factor = 1.0 / ((self.N - m + 1) * self.N)
        # corsum_row = numpy.zeros(len(r_range))

        # when parallel aprochach is in the case:
        # We can scan thourgh the matrix in parallel
        m_counts = max(m_range)
        max_m = m_counts

        corsum_matrix = numpy.zeros((m_counts, len(r_range)))

 

        a = -(r_range[0] - r_range[-1])/(r_range.size)
        b = r_range[0]

        for row_n in range(rank, self.N - 1, size):
            for m_index, row in enumerate(self.windups(m_counts, row_n, tau)):
                # tutaj na mnożeniu możemy zystać..
                corsum_matrix[m_index] += self.r_range_filter(row, r_range, a,b)# * 2# * factor

            for key in [k for k in self._rows.keys() if k < row_n]:
                self._rows.pop(key)

        self.multi(corsum_matrix, r_range)


        results = comm.gather(corsum_matrix, root=0)
        if rank == 0:
            summed = corsum_matrix = numpy.zeros((m_counts, len(r_range)))
            for r in results:
                summed += r

            if 0:
                for row in range(m_counts):
                    m = row + 1
                    factorA = (self.N - (m - 1) * tau)
                    factorB = (self.N - 1 - (m - 1) * tau)
                    factor = (factorA * factorB)/2.0
                    summed[row] /= factor
            return summed.transpose()

        else:
            return None
#        return corsum_matrix.transpose()

def test(signal, m, tau, r):
    total = 0
    factorA = len(signal) - (m -1) * tau
    factorB = len(signal) - (m -1) * tau - 1
    for i in range(len(signal) - (m -1) * tau):
        template = numpy.array(signal[i:i + m:tau])
        for j in range(len(signal) - (m -1) * tau):
            if i == j:
                continue
            checked = numpy.array(signal[j:j+m:tau])
            total += numpy.abs((template - checked)).max() <= r
    return total, 1.0/ (factorA * factorB), total * 1.0/ (factorA * factorB)

if __name__ == "__main__":

    if rank == 0:
        ts = time.time()


    signal_data = sys.argv[1]
    tau = int(sys.argv[2])
    m = int(sys.argv[3])

    
    #print rea.REA('simple_test_2.txt', skiprows=0).get_signal()
    #print numpy.loadtxt('simple_test_3.txt')

    rea_reader = rea.REA(signal_data, skiprows=0)
    signal = rea_reader.get_signal()

    #N = 1000
    #signal = logistic_map(0.4, 3.99, N)
    #signal = numpy.array([1.2, 1.3, 1.1, 1.2, 1.3])


    #signal = numpy.loadtxt(signal_data)
    #print signal

    #print 'LENGTH:', len(signal)

    ncm = FNCMatrix(signal, 0, signal.size)
    #ncm = NCMatrix(signal)

    r_std = signal.std()

    r_min =  r_std * 0.001
    r_max = r_std * 5.0
    r_step = (r_max - r_min)/100
    print 'R_min', r_min, 'R_max', r_max
    
    r_range = numpy.arange(r_min, r_max, r_step)[::-1]
    #r_range = numpy.arange(r_std * 0.1, r_std, r_std * 0.1)
    #print 'RANGE', r_range
#    for row in ncm.generate_rows(tau):
#        print row

    m_range = range(1, m + 1) # zawsze prawa strona nie wchodzi


#    for r in  r_range:
#        d = test(signal, 1, 1, r)
#        print r, d[0]


    #print r_range
    c_m = ncm.corsum_matrix(m_range, r_range, tau)

    if rank == 0 and 1:
        path = 'tmp/RR/tau_%d/' % tau
        if not os.path.exists(path):
            os.makedirs(path)
        head, tail = os.path.split(signal_data)
        out = open('tmp/RR/tau_%d/%s' % (tau, tail), 'w')
        for i,r in enumerate(r_range):
            #data = [r] + (numpy.log((c_m[i, :] + 0.00000000000000000001))).tolist()
            data = [r] + c_m[i, :].tolist()
            out.write("\t".join(["%f" % z for z in data]) + '\n')


        time_out = open('tmp/RR/time', 'a')
        splitted = sys.argv[1].split('_')[1]
        splitted = splitted.split('.')[0]
        time_out.write('%s\t%s\t%.10f\n' % (sys.argv[1], splitted, time.time() - ts))
        time_out.close()

