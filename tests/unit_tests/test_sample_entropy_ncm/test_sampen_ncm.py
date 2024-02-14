import unittest
import numpy as np
import archive.sampen_ncm as sampen


class TestSampleEntropySingleR(unittest.TestCase):
    def setUp(self):
        self.test_rr = np.loadtxt("100_nsr_800_5.csv")
        self.validation_data = np.loadtxt("SZ_out")

    def test_correlation_sum_matrix(self):
        idx = 0
        for r in self.validation_data[:, 0]:
            cm = sampen.calc_correletion_sums(self.test_rr, 2, r)
            self.assertAlmostEqual(cm[0], self.validation_data[idx, 1], places=5)
            self.assertAlmostEqual(cm[1], self.validation_data[idx, 2], places=5)
            idx += 1

    def test_sample_entropy(self):
        idx = 0
        for r in self.validation_data[:, 0]:
            se = sampen.calc_samp_en(self.test_rr, r)
            se_test = np.log(self.validation_data[idx, 1]) - np.log(self.validation_data[idx, 2])
            self.assertAlmostEqual(se, se_test, places=4)
            idx += 1

    def test_bruteforce_comparison(self):
        idx = 0
        for r in self.validation_data[:, 0]:
            cm = sampen.calc_correletion_sums(self.test_rr, 2, r)
            cm_bf_m1 = sampen.correlation_sum(self.test_rr,r,m=1,tau=1)
            self.assertAlmostEqual(cm[0], cm_bf_m1, places=5)
            idx += 1

if __name__ == '__main__':
    unittest.main()
