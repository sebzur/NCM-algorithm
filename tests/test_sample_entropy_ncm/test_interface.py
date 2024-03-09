import os
import unittest
import numpy as np

import ncm.methods.bruteforce as bf
from ncm.api import Process


class TestProcessInterface(unittest.TestCase):
    # python ncm./run.py testdata/dane.csv ncm_plain 10 1 test1.txt 0 --fmax=15  --rsteps=1000 --normalize
    def setUp(self) -> None:
        self.signal = np.loadtxt("100_nsr_800_5.csv")

    def test_plain_m1(self):
        Process("ncm_plain")("100_nsr_800_5.csv", "ncm_plain",10,1,"test_res_plain",0,0,15,1000,None,1,0,False,True)
        test_data = np.loadtxt("test_res_plain.txt")
        idx = 0
        for r in test_data[:, 0]:
            bruteforce_corr_sum = bf.Matrix(self.signal).corsum_matrix(m=[1], r_range=np.array([r]), tau=1)[0]
            self.assertAlmostEqual(bruteforce_corr_sum, test_data[idx, 1], places=5)
            idx += 1
        os.remove("test_res_plain.txt")

    def test_mpi_m1(self):
        Process("ncm_mpi")("100_nsr_800_5.csv", "ncm_mpi",10,1,"test_res_mpi",0,0,15,1000,None,1,0,False,True,False)
        test_data = np.loadtxt("test_res_mpi.txt")
        idx = 0
        for r in test_data[:, 0]:
            bruteforce_corr_sum = bf.Matrix(self.signal).corsum_matrix(m=[1], r_range=np.array([r]), tau=1)[0]
            self.assertAlmostEqual(bruteforce_corr_sum, test_data[idx, 1], places=5)
            idx += 1
        os.remove("test_res_mpi.txt")

    def test_plain_m2(self):
        Process("ncm_plain")("100_nsr_800_5.csv", "ncm_plain",10,1,"test_res_plain",0,0,15,1000,None,1,0,False,True)
        test_data = np.loadtxt("test_res_plain.txt")
        idx = 0
        for r in test_data[:, 0]:
            bruteforce_corr_sum = bf.Matrix(self.signal).corsum_matrix(m=[2], r_range=np.array([r]), tau=1)[0]
            self.assertAlmostEqual(bruteforce_corr_sum, test_data[idx, 2], places=5)
            idx += 1
        os.remove("test_res_plain.txt")

    def test_mpi_m2(self):
        Process("ncm_mpi")("100_nsr_800_5.csv", "ncm_mpi",10,1,"test_res_mpi",0,0,15,1000,None,1,0,False,True,False)
        test_data = np.loadtxt("test_res_mpi.txt")
        idx = 0
        for r in test_data[:, 0]:
            bruteforce_corr_sum = bf.Matrix(self.signal).corsum_matrix(m=[2], r_range=np.array([r]), tau=1)[0]
            self.assertAlmostEqual(bruteforce_corr_sum, test_data[idx, 2], places=5)
            idx += 1
        os.remove("test_res_mpi.txt")
