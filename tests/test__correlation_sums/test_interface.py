import os
import unittest
import numpy as np

import ncm.methods.bruteforce as bf
from ncm.api import Process


class TestProcessInterface(unittest.TestCase):
    """
    Test suit prepered for testing wheter using NCM Process class we will create correlation sums matrix for given data
    that has the same values as correlation sums calculated using classic approach for the same data.
    """
    def setUp(self) -> None:
        self.signal = np.loadtxt("100_nsr_800_5.csv")

    def test_plain_m1(self):
        """
        This test compared values calculated via NCM_plain method for 1000 diffrent r values and m = 1 for given data
        with correlation sums calculated using classic approach.
        """
        Process("ncm_plain")(
            "100_nsr_800_5.csv",
            "ncm_plain",
            10,
            1,
            "test_res_plain",
            0,
            0,
            15,
            1000,
            None,
            1,
            0,
            False,
            True,
        )
        test_data = np.loadtxt("test_res_plain.txt")
        idx = 0
        for r in test_data[:, 0]:
            bruteforce_corr_sum = bf.Matrix(self.signal).corsum_matrix(
                m=[1], r_range=np.array([r]), tau=1
            )[0]
            self.assertAlmostEqual(bruteforce_corr_sum, test_data[idx, 1], places=5)
            idx += 1
        os.remove("test_res_plain.txt")

    def test_mpi_m1(self):
        """
        This test compared values calculated via NCM_mpi method for 1000 diffrent r values and m = 1 for given data
        with correlation sums calculated using classic approach.
        """
        Process("ncm_mpi")(
            "100_nsr_800_5.csv",
            "ncm_mpi",
            10,
            1,
            "test_res_mpi",
            0,
            0,
            15,
            1000,
            None,
            1,
            0,
            False,
            True,
            False,
        )
        test_data = np.loadtxt("test_res_mpi.txt")
        idx = 0
        for r in test_data[:, 0]:
            bruteforce_corr_sum = bf.Matrix(self.signal).corsum_matrix(
                m=[1], r_range=np.array([r]), tau=1
            )[0]
            self.assertAlmostEqual(bruteforce_corr_sum, test_data[idx, 1], places=5)
            idx += 1
        os.remove("test_res_mpi.txt")

    def test_plain_m2(self):
        """
        This test compared values calculated via NCM_plain method for 1000 diffrent r values and m = 2 for given data
        with correlation sums calculated using classic approach.
        """
        Process("ncm_plain")(
            "100_nsr_800_5.csv",
            "ncm_plain",
            10,
            1,
            "test_res_plain",
            0,
            0,
            15,
            1000,
            None,
            1,
            0,
            False,
            True,
        )
        test_data = np.loadtxt("test_res_plain.txt")
        idx = 0
        for r in test_data[:, 0]:
            bruteforce_corr_sum = bf.Matrix(self.signal).corsum_matrix(
                m=[2], r_range=np.array([r]), tau=1
            )[0]
            self.assertAlmostEqual(bruteforce_corr_sum, test_data[idx, 2], places=5)
            idx += 1
        os.remove("test_res_plain.txt")

    def test_mpi_m2(self):
        """
        This test compared values calculated via NCM_mpi method for 1000 diffrent r values and m = 2 for given data
        with correlation sums calculated using classic approach.
        """
        Process("ncm_mpi")(
            "100_nsr_800_5.csv",
            "ncm_mpi",
            10,
            1,
            "test_res_mpi",
            0,
            0,
            15,
            1000,
            None,
            1,
            0,
            False,
            True,
            False,
        )
        test_data = np.loadtxt("test_res_mpi.txt")
        idx = 0
        for r in test_data[:, 0]:
            bruteforce_corr_sum = bf.Matrix(self.signal).corsum_matrix(
                m=[2], r_range=np.array([r]), tau=1
            )[0]
            self.assertAlmostEqual(bruteforce_corr_sum, test_data[idx, 2], places=5)
            idx += 1
        os.remove("test_res_mpi.txt")
