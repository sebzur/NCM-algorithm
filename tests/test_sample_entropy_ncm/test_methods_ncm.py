import unittest
import numpy as np
import ncm.methods.ncm_plain as plain
import ncm.methods.ncm_mpi as mpi
import ncm.methods.bruteforce as bf
# const for logistic map
r_value = 100  # You can experiment with different values of r
initial_value = 0.5
iterations = 50

# const for sinus signal and signal with white noise
fs = 1000  # Sampling frequency
f = 5      # Frequency of the sine wave
t = np.arange(0, 1, 1/fs)  # Time vector
signal = np.sin(2*np.pi*f*t)

# Generating white noise
noise_amplitude = 0.2
white_noise = np.random.normal(0, noise_amplitude, len(t))

# Adding white noise to the signal
noisy_signal = signal + white_noise


class TestSampleEntropySyntheticData(unittest.TestCase):

    def test_sinus_signal(self):
        tested_r = 0.1
        corr_sum_bruteforce = bf.Matrix(signal).corsum_matrix(m=[1], r_range=np.array([tested_r]), tau=1)
        ncm_plain = plain.Matrix(signal).corsum_matrix(m_range=[2], r_range=np.array([tested_r]))
        ncm_mpi = mpi.Matrix(signal).corsum_matrix(m_range=[2], r_range=np.array([tested_r]), tau=1)

        assert corr_sum_bruteforce == ncm_plain[0][0] == ncm_mpi[0][0]

    def test_sinus_with_noise_signal(self):
        tested_r = 0.1
        corr_sum_bruteforce = bf.Matrix(noisy_signal).corsum_matrix(m=[1], r_range=np.array([tested_r]), tau=1)
        ncm_plain = plain.Matrix(noisy_signal).corsum_matrix(m_range=[2], r_range=np.array([tested_r]))
        ncm_mpi = mpi.Matrix(noisy_signal).corsum_matrix(m_range=[2], r_range=np.array([tested_r]), tau=1)

        assert corr_sum_bruteforce == ncm_plain[0][0] == ncm_mpi[0][0]


class TestSampleEntropyRRdata(unittest.TestCase):

    def setUp(self) -> None:
        self.short_signal = np.loadtxt("100_nsr_800_5.csv")
        self.longer_singal = np.loadtxt("00_af_17_0.csv")

    def test_sinus_signal(self):
        tested_r = 0.1
        corr_sum_bruteforce = bf.Matrix(self.short_signal).corsum_matrix(m=[1], r_range=np.array([tested_r]), tau=1)
        ncm_plain = plain.Matrix(self.short_signal).corsum_matrix(m_range=[2], r_range=np.array([tested_r]))
        ncm_mpi = mpi.Matrix(self.short_signal).corsum_matrix(m_range=[2], r_range=np.array([tested_r]), tau=1)

        assert corr_sum_bruteforce == ncm_plain[0][0] == ncm_mpi[0][0]

    def test_sinus_with_noise_signal(self):
        tested_r = 0.1
        corr_sum_bruteforce = bf.Matrix(self.longer_singal).corsum_matrix(m=[1], r_range=np.array([tested_r]), tau=1)
        ncm_plain = plain.Matrix(self.longer_singal).corsum_matrix(m_range=[2], r_range=np.array([tested_r]))
        ncm_mpi = mpi.Matrix(self.longer_singal).corsum_matrix(m_range=[2], r_range=np.array([tested_r]), tau=1)

        assert corr_sum_bruteforce == ncm_plain[0][0] == ncm_mpi[0][0]
