import argparse
import os
import numpy

def logistic_map(initial_rate, A, years):
    """

    Arguments:

     - initial_rate: a number between zero and one, represents the initial ratio of population to max. population (at year 0)
     - A: is a positive number, and represents a combined rate for reproduction and starvation
     - years: a positive integer tells us for for how many years should the process continue

     """
    x = initial_rate
    for i in xrange(years):
        x = A * x * (1 - x)
        yield x

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Logistic map generator')
    parser.add_argument('A', type=float, default=3.99, help='')
    parser.add_argument('initial_rate', type=float, default=0.4, help='Initial x value')
    parser.add_argument('years', type=int, default=1000, help='Number of years')
    args = parser.parse_args()

    for value in logistic_map(args.initial_rate, args.A, args.years):
        print value
    
