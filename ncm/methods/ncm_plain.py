import numpy as np
from typing import Union
import itertools

NumberTypes = (int, float)


class Matrix:

    def __init__(self, signal):
        self.__class__.__name__ = "NCM_plain"
        self.signal = signal

    def corsum_matrix(self, m_range, r_range, tau: int = 1, normalize=None, selfmatches=None, precision=6):
        assert tau == 1
        signal = self.signal
        m_counts = max(m_range)
        m_range = range(m_counts)

        # find linear coefficient for solving algebraic equation
        corsum_matrix = np.zeros((m_counts, len(r_range)))
        if r_range.size > 1:
            a = -(r_range[0] - r_range[-1]) / (r_range.size - 1)
            b = r_range[0]

        # define size of Norm component matrix
        size_X = (len(signal) - (1 - 1) * tau) - 1
        size_Y = (len(signal) - 1 - (1 - 1) * tau)
        # create triangular NCM matrix
        NCM = np.zeros((size_X, size_Y))
        for i_row in range(size_X):
            for j_column in range(size_Y):
                if i_row + (j_column + 1) * tau <= len(signal) - 1:
                    NCM[i_row, j_column] = np.abs(signal[i_row] - signal[i_row + (j_column + 1) * tau])

        # calculate correlation sum for embedded dimension m
        for m in m_range:
            for current_row_idx in range(
                    len(NCM) - m):  # Careful - we need to iterate over len(NCM)-m rows of NCM matrix, becouse we cant take more than one row on last row
                current_row = NCM[current_row_idx:current_row_idx + (m + 1)]
                current_row = current_row[:, :size_Y - current_row_idx - m].max(axis=0)
                # Solve for v using the given coefficients a and b
                z = np.zeros(r_range.size)
                if r_range.size == 1:
                    z[0] += (current_row <= r_range[0]).sum()
                else:
                    v = ((current_row - b) / a).astype('int')
                    v = np.sort(v)
                    for k, v in itertools.groupby(v):
                        z[:k + 1] += len(list(v))
                corsum_matrix[m] += z

        # normalize correlation sum, multiply by 2 due to property of triangular matrix and exclude duplicates
        for m in m_range:
            factorA = (len(signal) - m * tau)
            factorB = (len(signal) - 1 - m * tau)
            factor = factorA * factorB
            corsum_matrix[m] = corsum_matrix[m] * 2 * 1 / factor
        return np.round(corsum_matrix.T, precision)
