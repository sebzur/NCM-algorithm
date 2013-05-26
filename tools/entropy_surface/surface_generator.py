import numpy
import sys

def surface(filename):
    #delta = 0.00000000000000000000000000000000000001
    data = numpy.loadtxt(filename)
    sd = data[0,0]/5.0
    for column_id in range(2, data.shape[1]):


        try:
            log_cm_m = numpy.log(data[:,column_id - 1])
            log_cm_m_1 = numpy.log(data[:,column_id])
            #log_cm_m = data[:,column_id - 1]
            #log_cm_m_1 = data[:,column_id]
            entropy = log_cm_m - log_cm_m_1
        except:
            entropy = 0

        for r_index in range(data.shape[0]):
    	    if not numpy.isnan(entropy[r_index]):
                if entropy[r_index] >= 0:
                    #print numpy.log(data[r_index, 0]), column_id - 1, entropy[r_index]
                    print data[r_index, 0]/sd, column_id - 1, entropy[r_index]
                    #print data[r_index, 0], column_id - 1, entropy[r_index]
        print 

if __name__ == "__main__":
    surface(sys.argv[1])
    
