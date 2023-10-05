import sys
import numpy

if __name__ == "__main__":
    """ Merge two files with (x,y) structure, assuming that both 
    have the same x columns.
    """
    a = numpy.loadtxt(sys.argv[1])
    b = numpy.loadtxt(sys.argv[2])
    numpy.savetxt(sys.argv[3], [[el_a[0], el_a[1], el_b[1]] for el_a, el_b in zip(a,b)])
        
    
