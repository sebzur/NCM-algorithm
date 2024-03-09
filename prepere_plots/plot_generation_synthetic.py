from typing import List
import time
import numpy as np
import matplotlib.pyplot as plt

from ncm.methods.bruteforce import Matrix as Brutforce
from ncm.methods.ncm_plain import Matrix as NCM_Plain
from ncm.methods.ncm_mpi import Matrix as NCM_Enchanced


def logistic_map(num_iterations, r=3.8, x0=0.5):
    result = []
    x = x0
    for _ in range(num_iterations):
        result.append(x)
        x = r * x * (1 - x)
    return np.array(result)


def gen_elapsed_time_for_diffrent_n(cls, n_point, m, r_range):

    lorenz_data = []
    for n in n_point:
        data = logistic_map(n)
        lorenz_data.append(data)

    times = []
    for data in lorenz_data:
        start = time.perf_counter()
        res = cls(data, 0, 1).corsum_matrix(m, r_range, tau=1)
        end = time.perf_counter()
        t = end - start
        times.append(t)
    return times


def gen_elapsed_time_for_diffrent_r(cls, n,m , r_ranges):
    data = logistic_map(n)
    times = []
    for r_range in r_ranges:
        start = time.perf_counter()
        res = cls(data, 0, 1).corsum_matrix(m,r_range, tau=1)
        end = time.perf_counter()
        t = end - start
        times.append(t)
    return times


def gen_elapsed_time_for_diffrent_m(cls, n, m_range, r):
    data = logistic_map(n)
    times = []
    for m in m_range:
        start = time.perf_counter()
        res = cls(data, 0, 1).corsum_matrix([m], r, tau=1)
        end = time.perf_counter()
        t = end - start
        times.append(t)
    return times


def gen_elapsed_time_for_bruteforce_for_m(n,m_range,r):
    data = logistic_map(n)
    times = []
    corsum =[]
    for m in m_range:
        start = time.perf_counter()
        m_list = range(1,m+1)
        for m_i in m_list:
            res = Brutforce(data, 0, 1).corsum_matrix([m_i], r, tau=1)
            corsum.append(res)
        end = time.perf_counter()
        t = end - start
        times.append(t)
    return times


def gen_plots(N_points: np.ndarray, r_ranges: List, m_ranges: List, fixed_n:int, fixed_m:int,fixed_r:int):
    bf_time = gen_elapsed_time_for_diffrent_n(Brutforce, N_points, [fixed_m], fixed_r)
    plain_time = gen_elapsed_time_for_diffrent_n(NCM_Plain, N_points, [fixed_m], fixed_r)
    ncm_time = gen_elapsed_time_for_diffrent_n(NCM_Enchanced, N_points, [fixed_m], fixed_r)

    plt.plot(N_points, plain_time, label="NCM_plain")
    plt.plot(N_points, ncm_time, label="NCM_Enchanced")
    plt.plot(N_points, bf_time, label="Bruteforce")
    plt.title("Time of calculating correlation sums for data of different length")
    plt.xlabel("Length of data")
    plt.ylabel("Time [s]")
    plt.legend()
    #plt.xscale("log")
    #plt.yscale("log")

    #plt.savefig(f"synthetic_diffrent_N_fixed_r-{fixed_r}_fixed_m-{fixed_m}.png")
    plt.show()



    bf_time = gen_elapsed_time_for_diffrent_r(Brutforce, fixed_n, [fixed_m], r_ranges)
    plain_time = gen_elapsed_time_for_diffrent_r(NCM_Plain, fixed_n, [fixed_m], r_ranges)
    ncm_time = gen_elapsed_time_for_diffrent_r(NCM_Enchanced, fixed_n, [fixed_m], r_ranges)

    r_ranges_plot = [len(r) if isinstance(r, (list, np.ndarray)) else 1 for r in r_ranges]
    plt.plot(r_ranges_plot, plain_time, label="NCM_plain")
    plt.plot(r_ranges_plot, ncm_time, label="NCM_Enchanced")
    plt.plot(r_ranges_plot, bf_time, label="Bruteforce")
    plt.title(f"Time of calculating correlation sums for diffrent number of r values, signal length = {fixed_n}")
    plt.xlabel("Number of r")
    plt.ylabel("Time [s]")
    plt.legend()

    #plt.savefig(f"synthetic_diffrent_r_fixed_n-{fixed_n}_fixed_m-{fixed_m}.png")
    plt.show()

    bf_time = gen_elapsed_time_for_bruteforce_for_m( fixed_n, m_ranges, fixed_r)
    plain_time = gen_elapsed_time_for_diffrent_m(NCM_Plain, fixed_n, m_ranges, fixed_r)
    ncm_time = gen_elapsed_time_for_diffrent_m(NCM_Enchanced, fixed_n, m_ranges, fixed_r)

    plt.plot(m_ranges, plain_time, label="NCM_plain")
    plt.plot(m_ranges, ncm_time, label="NCM_Enchanced")
    plt.plot(m_ranges, bf_time, label="Bruteforce")
    plt.title(f"Time of calculating correlation sums for  diffrent number of m values, signal length ={fixed_n}")
    plt.xlabel("Value of m")
    plt.ylabel("Time [s]")
    plt.legend()

    #plt.savefig(f"synthetic_diffrent_m_fixed_n-{fixed_n}_fixed_r-{fixed_r}.png")
    plt.show()


if __name__ == "__main__":

    # to są wartości r potrzebne do wykresu  czas/liczba parametrów r
    r_ranges = [100, np.array([100, 200, 300]), np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])]
    # to są długości sygnału z układu Lorenza które chcemy wykorzystać do wykresu czas/długośc nagrania
    N_points = np.array([10,100,1000,10000])
    # to są wartości m ktore chcemy wykorzystać do wykresu czas/liczba parametrów m
    m_range = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    fixed_n = 100 # to jest długość sygnału wykorzystywana dla wykresów czas/ liczba r i czas/ liczbba m
    fixed_m = 1 # to jest wartośc m wykorzystywana dla wykresów czas/ długość nagrania i czas/ liczba r
    fixed_r = 100 # to jest wartość r wykorzystywana dla wyrkesów czas/ długość nagrania i czas/ liczba m

    gen_plots(N_points, r_ranges, m_range, fixed_n, fixed_m, fixed_r)

