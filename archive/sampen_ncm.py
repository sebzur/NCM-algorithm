import numpy as np


def calc_correletion_sums(signal: np.ndarray, m: int, r) -> np.ndarray:
    tau = 1
    m_range = range(m)
    m_counts = max(m_range) + 1

    corsum_matrix = np.zeros(m_counts)

    # define size of Norm component matrix
    size_X = len(signal) - 1
    size_Y = len(signal) - 1

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
            # counting max norm here
            current_row = current_row[:,:size_Y-current_row_idx-m].max(axis=0)
            # sum correlated entries
            corsum_matrix[m] += (current_row <= r).sum()


    # normalize correlation sum, multiply by 2 due to property of triangular matrix and exclude duplicates
    for m in m_range:
        factorA = (len(signal) - (m) * tau)
        factorB = (len(signal) - 1 - (m) * tau)
        factor = factorA * factorB
        corsum_matrix[m] = corsum_matrix[m] * 2 * 1/factor
    return np.round(corsum_matrix.T,6)


def calc_samp_en(signal: np.ndarray,r: float) -> float:
    """
    calculate Sample entropy for single r value
    """
    cm = calc_correletion_sums(signal, 2, r)
    sampen = np.log(cm[0]) - np.log(cm[1])
    return sampen


def heaviside(x):
    if x >= 0:
        return 1
    else:
        return 0


def max_coordinate_diff(v1, v2):
    return max(abs(v1-v2))


def correlation_sum(signal, r, m, tau):
    Lm = len(signal) - (m - 1) * tau
    Corr_sum = 0
    for i in range(Lm):
        v_i = np.array(signal[i:i + m:tau])
        for j in range(Lm):
            if i == j:
                continue
            v_j = np.array(signal[j:j+m:tau])
            Corr_sum += heaviside(r-max_coordinate_diff(v_i, v_j))
    return np.round(Corr_sum * (1/Lm) * (1/(Lm-1)),6)


if __name__ == "__main__":
    test_rr = np.loadtxt("100_nsr_800_5.csv")
    validation_data = np.loadtxt("SZ_out")

    idx = 0
    for r in validation_data[:, 0]:
        cor_sum = calc_correletion_sums(test_rr,m=2,r=r)
        corr_sum_bf_m1 = correlation_sum(test_rr,r=r,m=1,tau=1)
        #corr_sum_bf_m2 = correlation_sum(test_rr,r=r,m=2,tau=1)
        print(r,cor_sum[0] - validation_data[idx,1],cor_sum[0],validation_data[idx,1])
        print(r,corr_sum_bf_m1 - validation_data[idx,1],corr_sum_bf_m1,validation_data[idx,1])
        print("next")

        idx +=1