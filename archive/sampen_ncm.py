import numpy as np
from typing import List

NumberTypes = (int, float)


def calc_correletion_sums(signal: np.ndarray, m: int, r_range) -> np.ndarray:
    tau = 1
    m_range = range(m)
    m_counts = max(m_range) + 1

    if isinstance(r_range, NumberTypes):
        corsum_matrix = np.zeros((m_counts, 1))
        a = -1
        b = r_range - 1 if r_range != 0.0 else r_range
        r_range_size = 1
    elif isinstance(r_range, np.ndarray):
        corsum_matrix = np.zeros((m_counts, len(r_range)))
        a = -(r_range[0] - r_range[-1]) / (r_range.size - 1)
        b = r_range[0]
        r_range_size = r_range.size
    else:
        raise ValueError("r_range must be number or numpy array of numbers")


    # define size of Norm component matrix
    size_X = (len(signal) - (1 - 1) * tau)-1
    size_Y = (len(signal) - 1 - (1 - 1) * tau)

    # create triangular NCM matrix
    NCM = np.zeros((size_X, size_Y))
    for i_row in range(size_X):
        for j_column in range(size_Y):
            if i_row + (j_column + 1) * tau <= len(signal)-1:
                NCM[i_row,j_column] = np.abs(signal[i_row] - signal[i_row + (j_column + 1) * tau])

    # calculate correlation sum for embedded dimension m
    for m in m_range:
        for current_row_idx in range(len(NCM)-m): # Careful - we need to iterate over len(NCM)-m rows of NCM matrix, because we cant take more than one row on last row
            current_row = NCM[current_row_idx:current_row_idx + (m+1)]
            current_row = current_row[:,:size_Y-current_row_idx-m].max(axis=0)
            # Solve for v using the given coefficients a and b
            v = ((current_row - b) / a).astype('int')
            v = np.sort(v)
            z = np.zeros(r_range_size)
            for idx in v:
                z[:idx+1] += 1
            corsum_matrix[m] += z

    # normalize correlation sum, multiply by 2 due to property of triangular matrix and exclude duplicates
    for m in m_range:
        factorA = (len(signal) - (m) * tau)
        factorB = (len(signal) - 1 - (m) * tau)
        factor = factorA * factorB
        corsum_matrix[m] = corsum_matrix[m] * 2 * 1/factor
    return corsum_matrix.T


def calc_samp_en(signal: np.ndarray,r: float) -> float:
    """
    calculate Sample entropy for single r value
    """
    cm = calc_correletion_sums(signal, 2, r)[0]
    sampen = np.log(cm[0]) - np.log(cm[1])
    return sampen
