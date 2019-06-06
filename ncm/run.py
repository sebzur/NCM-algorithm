import argparse
import os
from mpi4py import MPI
import numpy

from matrix import matrix

def load_data(filename, usecol):
    root, ext = os.path.splitext(filename)
    signal = numpy.loadtxt(filename, skiprows=0, usecols=(usecol,))
    return signal

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NCM algorithm for correlation sums')
    parser.add_argument('file', type=str, help='Signal data - single column file or REA format')
    parser.add_argument('m', type=int, default=10, help='Embeding dimension (default: 10)')
    parser.add_argument('tau', type=int, default=1, help='Time lag (default: 1)')
    parser.add_argument('output', type=str, help='Output file name')
    parser.add_argument('usecol', type=int, help='Column with data')
    parser.add_argument('--normalize', dest='normalize', action='store_true', help='Should the output be normalized?')
    parser.add_argument('--selfmatches', dest='selfmatches', action='store_true', help='Correlate selfmatches?')
    parser.add_argument('--rmode', type=str, default='steps', help='(steps|diff)')
    parser.add_argument('--rdiff', type=float, default=100, help='Number of treshold values')
    parser.add_argument('--wsize', type=int, default=None, help='Windows size')
    parser.add_argument('--wstep', type=int, default=1, help='Windows steps')
    args = parser.parse_args()
    # we print out things only on rank 0 (aka master node)
    signal = load_data(args.file, usecol=args.usecol)
    c_m = matrix(signal, args.m, args.tau, rmode=args.rmode, rdiff=args.rdiff, normalize=args.normalize, selfmatches=args.selfmatches)
    numpy.savetxt(args.output, c_m)
