import numpy
import sys

def surface(filename):
    #delta = 0.00000000000000000000000000000000000001
    data = numpy.loadtxt(filename)
    for column_id in range(2, data.shape[1]):
        log_cm_m = numpy.log(data[:,column_id - 1])
        log_cm_m_1 = numpy.log(data[:,column_id])
        entropy = log_cm_m - log_cm_m_1
        for r_index in range(data.shape[0]):
    	    if not numpy.isnan(entropy[r_index]):
        	print numpy.log(data[r_index, 0]), column_id - 1, entropy[r_index]
        print 

if __name__ == "__main__":
    surface(sys.argv[1])
    
