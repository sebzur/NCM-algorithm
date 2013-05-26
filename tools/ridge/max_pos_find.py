import numpy
import sys
import os

def finde(filename):
    data = numpy.loadtxt(filename)
    m_values = numpy.arange(1, data[:,1].max() + 1)
    for m in m_values:
        r_data = numpy.compress(data[:,1] == m, data[:,0])
        sampen_data = numpy.compress(data[:,1] == m, data[:,2])
        max_ind = sampen_data.argmax(0)
        yield m, r_data[max_ind], sampen_data[max_ind]


if __name__ == "__main__":
    f_name = "%s.max" % os.path.splitext(sys.argv[1])[0]
    numpy.savetxt(f_name, numpy.array(list(finde(sys.argv[1]))))

