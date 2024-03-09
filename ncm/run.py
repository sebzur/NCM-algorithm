import argparse
from api import Process,rank


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NCM algorithm for correlation sums')
    parser.add_argument('file', type=str, help='Signal data - single column file or REA format')
    parser.add_argument('method', type=str,
                        help='Method for calculating correlation sums (classic, NCM, NCM-MPI enhanced)')
    parser.add_argument('m', type=int, default=10, help='Embeding dimension (default: 10)')
    parser.add_argument('tau', type=int, default=1, help='Time lag (default: 1)')
    parser.add_argument('output', type=str, help='Output file name')
    parser.add_argument('usecol', type=int, help='Column with data')
    parser.add_argument('--normalize', dest='normalize', action='store_true', help='Should the output be normalized?')
    parser.add_argument('--selfmatches', dest='selfmatches', action='store_true', help='Correlate selfmatches?')
    parser.add_argument('--skiprows', type=int, default=0, dest='skiprows', help='Skip first n rows from input file')
    parser.add_argument('--rsteps', type=int, default=None, help='Number of treshold values')
    parser.add_argument('--fmin', type=float, default=0,
                        help='SD * f_min - lower value for tresholds range (default: 0.001)')
    parser.add_argument('--fmax', type=float, default=None,
                        help='SD * f_max - upper value for tresholds range (default: 5.0)')
    parser.add_argument('--wsize', type=int, default=None, help='Window size')
    parser.add_argument('--wstep', type=int, default=1, help='Window step')
    parser.add_argument('--precision', type=int, default=6, help='calculation precision,number of digits after coma')

    args = parser.parse_args()
    # we print out things only on rank 0 (aka master node)
    verbose = rank == 0
    if args.tau > 1:
        if verbose:
            print(
                'Sorry, at this moment the implementation only allows for tau=1. This will be changed soon. Stay tuned!')
    else:
        Process(args.method)(args.file, args.method, args.m, args.tau, output=args.output, usecol=args.usecol, verbose=verbose,
                  normalize=args.normalize, selfmatches=args.selfmatches, skiprows=args.skiprows,
                  f_min=args.fmin, f_max=args.fmax, r_steps=args.rsteps, wstep=args.wstep, wsize=args.wsize)








